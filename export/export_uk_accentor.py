"""Build the ``ukr_accentor/`` bundle from the upstream ukrainian-accentor-transformer model.

Upstream (https://github.com/Theodotus1243/ukrainian-accentor-transformer,
MIT) ships its Transformer only as a CTranslate2 ``model.bin`` on the Hugging
Face Hub, with a SentencePiece tokenizer.  This script reads that binary
directly, rebuilds the network in torch with the same arithmetic CTranslate2
uses (sinusoidal positions counted from 1, sin and cos halves concatenated,
pre-norm layers, ReLU, embeddings scaled by sqrt(256)) and exports two ONNX
graphs: ``encoder.onnx`` (source ids to memory) and ``decoder.onnx`` (target
prefix and memory to the logits of the next token).  The SentencePiece model
becomes a ``tokenizer.json`` (Unigram model, the same precompiled nmt_nfkc
normalisation, Metaspace pre-tokenizer) for the ``tokenizers`` runtime.

Beam search (beam 2, repetition penalty 1.2, length penalty 1) is not part of
the graphs: ``stressonnx/backends/uk_accentor.py`` runs it in numpy, step by
step, the way CTranslate2 does.

Usage::

    uv venv --python 3.11 .venv-export
    uv pip install --python .venv-export/bin/python torch onnx onnxruntime \\
        sentencepiece huggingface_hub numpy
    .venv-export/bin/python export/export_uk_accentor.py <out_dir>

writes ``<out_dir>/ukr_accentor/{encoder.onnx, decoder.onnx, tokenizer.json,
vocab.txt, meta.json, LICENSE.ukrainian-accentor-transformer}`` and prints
each file's size and sha256.  The model files are fetched from the pinned
Hugging Face revision; the licence text is fetched from the pinned commit of
the upstream GitHub repository.
"""
import base64
import hashlib
import json
import math
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as spm_pb2

UPSTREAM_URL = "https://github.com/Theodotus1243/ukrainian-accentor-transformer"
UPSTREAM_COMMIT = "1af138199c61d996094575fb0efe34a0b2e85adc"
HF_REPO = "theodotus/ukrainian-accentor-transformer"
HF_REVISION = "5289b00e07f0aa8678825cca59684b1fb0ec8c3c"  # tag v0.1, branch main

D_MODEL = 256
HEADS = 8
LAYERS = 4
LN_EPS = 1e-5  # CTranslate2 default for LayerNorm with beta and no config.json

_DTYPES = {0: np.float32, 1: np.int8, 2: np.int16, 3: np.int32, 4: np.float16}


def read_ct2_model(path: Path) -> dict:
    """Variables of a CTranslate2 ``model.bin`` (binary version 5)."""
    f = path.open("rb")

    def read_string() -> str:
        n = struct.unpack("H", f.read(2))[0]
        return f.read(n)[:-1].decode()

    version = struct.unpack("I", f.read(4))[0]
    spec = read_string()
    struct.unpack("I", f.read(4))  # spec revision
    if version != 5 or spec != "TransformerSpec":
        raise SystemExit(f"unexpected model.bin: binary version {version}, spec {spec}")
    variables = {}
    for _ in range(struct.unpack("I", f.read(4))[0]):
        name = read_string()
        rank = f.read(1)[0]
        dims = struct.unpack("I" * rank, f.read(4 * rank))
        dtype = f.read(1)[0]
        data = f.read(struct.unpack("I", f.read(4))[0])
        if dtype == 1 and rank == 1 and name.endswith("_token"):
            variables[name] = data.decode()
        else:
            variables[name] = np.frombuffer(data, _DTYPES[dtype]).reshape(dims).copy()
    return variables


def check_spec(v: dict) -> None:
    expected = {
        "activation": 0, "pre_norm": 1, "num_heads": HEADS, "with_relative_position": 0,
        "layernorm_embedding": 0, "encoder/scale_embeddings": 1, "decoder/scale_embeddings": 1,
        "decoder/start_from_zero_embedding": 0, "with_source_bos": 0, "with_source_eos": 0,
        "with_target_bos": 1, "user_decoder_start_tokens": 0,
    }
    for name, value in expected.items():
        got = int(v[name].reshape(-1)[0])
        if got != value:
            raise SystemExit(f"model.bin {name}={got}, this exporter handles {value}")
    if v["bos_token"] != "<s>" or v["eos_token"] != "</s>" or v["unk_token"] != "<unk>":
        raise SystemExit("unexpected special tokens")
    if "encoder/position_encodings/encodings" in v:
        raise SystemExit("learned position encodings are not handled")


def sinusoidal(max_time: int) -> torch.Tensor:
    """CTranslate2's table: positions 1..max_time, sin half then cos half."""
    half = D_MODEL // 2
    increment = math.log(10000.0) / (half - 1)
    timescales = torch.exp(-increment * torch.arange(half, dtype=torch.float32))
    scaled = torch.arange(1, max_time + 1, dtype=torch.float32)[:, None] * timescales[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class Linear(torch.nn.Module):
    def __init__(self, v, scope):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.from_numpy(v[f"{scope}/weight"]))
        self.bias = torch.nn.Parameter(torch.from_numpy(v[f"{scope}/bias"]))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Norm(torch.nn.Module):
    def __init__(self, v, scope):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.from_numpy(v[f"{scope}/gamma"]))
        self.beta = torch.nn.Parameter(torch.from_numpy(v[f"{scope}/beta"]))

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (D_MODEL,), self.gamma, self.beta, LN_EPS)


def split_heads(x):
    b, t, _ = x.shape
    return x.reshape(b, t, HEADS, D_MODEL // HEADS).transpose(1, 2)


def merge_heads(x):
    b, _, t, _ = x.shape
    return x.transpose(1, 2).reshape(b, t, D_MODEL)


def attend(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(D_MODEL // HEADS))
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    return merge_heads(torch.matmul(torch.softmax(scores, dim=-1), v))


class FeedForward(torch.nn.Module):
    def __init__(self, v, scope):
        super().__init__()
        self.norm = Norm(v, f"{scope}/layer_norm")
        self.inner = Linear(v, f"{scope}/linear_0")
        self.outer = Linear(v, f"{scope}/linear_1")

    def forward(self, x):
        return x + self.outer(torch.relu(self.inner(self.norm(x))))


class SelfAttention(torch.nn.Module):
    def __init__(self, v, scope):
        super().__init__()
        self.norm = Norm(v, f"{scope}/layer_norm")
        self.qkv = Linear(v, f"{scope}/linear_0")
        self.out = Linear(v, f"{scope}/linear_1")

    def forward(self, x, mask=None):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        return x + self.out(attend(split_heads(q), split_heads(k), split_heads(v), mask))


class CrossAttention(torch.nn.Module):
    def __init__(self, v, scope):
        super().__init__()
        self.norm = Norm(v, f"{scope}/layer_norm")
        self.q = Linear(v, f"{scope}/linear_0")
        self.kv = Linear(v, f"{scope}/linear_1")
        self.out = Linear(v, f"{scope}/linear_2")

    def forward(self, x, memory):
        q = self.q(self.norm(x))
        k, v = self.kv(memory).chunk(2, dim=-1)
        return x + self.out(attend(split_heads(q), split_heads(k), split_heads(v)))


class Encoder(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.from_numpy(v["encoder/embeddings_0/weight"]))
        self.positions = torch.nn.Parameter(sinusoidal(1024), requires_grad=False)
        self.attention = torch.nn.ModuleList(
            SelfAttention(v, f"encoder/layer_{i}/self_attention") for i in range(LAYERS))
        self.ffn = torch.nn.ModuleList(FeedForward(v, f"encoder/layer_{i}/ffn") for i in range(LAYERS))
        self.norm = Norm(v, "encoder/layer_norm")

    def forward(self, ids):
        x = self.embeddings[ids] * math.sqrt(D_MODEL) + self.positions[: ids.shape[1]]
        for attention, ffn in zip(self.attention, self.ffn):
            x = ffn(attention(x))
        return self.norm(x)


class Decoder(torch.nn.Module):
    """Logits of the token after *prefix*; the prefix starts with ``<s>``."""

    def __init__(self, v):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.from_numpy(v["decoder/embeddings/weight"]))
        self.positions = torch.nn.Parameter(sinusoidal(1024), requires_grad=False)
        self.self_attention = torch.nn.ModuleList(
            SelfAttention(v, f"decoder/layer_{i}/self_attention") for i in range(LAYERS))
        self.cross_attention = torch.nn.ModuleList(
            CrossAttention(v, f"decoder/layer_{i}/attention") for i in range(LAYERS))
        self.ffn = torch.nn.ModuleList(FeedForward(v, f"decoder/layer_{i}/ffn") for i in range(LAYERS))
        self.norm = Norm(v, "decoder/layer_norm")
        self.projection = Linear(v, "decoder/projection")

    def forward(self, prefix, memory):
        t = prefix.shape[1]
        x = self.embeddings[prefix] * math.sqrt(D_MODEL) + self.positions[:t]
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool))[None, None]
        for self_attention, cross_attention, ffn in zip(
                self.self_attention, self.cross_attention, self.ffn):
            x = ffn(cross_attention(self_attention(x, mask), memory))
        return self.projection(self.norm(x[:, -1]))


def export_graph(module, args, names, dynamic, path: Path) -> None:
    torch.onnx.export(
        module, args, str(path), input_names=names[0], output_names=names[1],
        dynamic_axes=dynamic, opset_version=17, dynamo=False, do_constant_folding=True,
    )
    onnx.checker.check_model(str(path))


def build_tokenizer(spm_path: Path) -> dict:
    """``tokenizer.json`` equivalent to the SentencePiece model: nmt_nfkc
    precompiled normalisation, extra spaces collapsed and stripped, Metaspace
    with a dummy prefix, Unigram vocabulary with the same ids."""
    model = spm_pb2.ModelProto()
    model.ParseFromString(spm_path.read_bytes())
    if model.trainer_spec.model_type != 1 or model.trainer_spec.byte_fallback:
        raise SystemExit("tokenizer.model is not a plain Unigram model")
    spec = model.normalizer_spec
    if not (spec.add_dummy_prefix and spec.remove_extra_whitespaces and spec.escape_whitespaces):
        raise SystemExit("unexpected SentencePiece normalizer options")
    charsmap = base64.b64encode(spec.precompiled_charsmap).decode()
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": i, "content": p.piece, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for i, p in enumerate(model.pieces) if p.type in (2, 3)
        ],
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Precompiled", "precompiled_charsmap": charsmap},
            {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "},
            {"type": "Strip", "strip_left": True, "strip_right": True},
        ]},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True},
        "post_processor": None,
        "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True},
        "model": {"type": "Unigram", "unk_id": model.trainer_spec.unk_id,
                  "vocab": [[p.piece, p.score] for p in model.pieces], "byte_fallback": False},
    }


def main(out_dir: Path) -> None:
    files = {name: Path(hf_hub_download(HF_REPO, name, revision=HF_REVISION))
             for name in ("ctranslate2/model.bin", "ctranslate2/shared_vocabulary.txt", "tokenizer.model")}
    v = read_ct2_model(files["ctranslate2/model.bin"])
    check_spec(v)

    bundle = out_dir / "ukr_accentor"
    bundle.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    encoder, decoder = Encoder(v).eval(), Decoder(v).eval()
    with torch.no_grad():
        export_graph(encoder, (torch.tensor([[7, 8, 9, 10]]),), (["src_ids"], ["memory"]),
                     {"src_ids": {0: "batch", 1: "src_len"}, "memory": {0: "batch", 1: "src_len"}},
                     bundle / "encoder.onnx")
        memory = encoder(torch.tensor([[7, 8, 9, 10]]))
        export_graph(decoder, (torch.tensor([[1, 7, 8], [1, 9, 10]]), memory.repeat(2, 1, 1)),
                     (["prefix_ids", "memory"], ["logits"]),
                     {"prefix_ids": {0: "beam", 1: "tgt_len"}, "memory": {0: "beam", 1: "src_len"},
                      "logits": {0: "beam"}},
                     bundle / "decoder.onnx")

    # the CTranslate2 vocabulary (``<blank>`` first, ``<unk>`` last) orders the
    # decoder's rows; the tokenizer vocabulary (``<unk>`` first) is the
    # SentencePiece one.  vocab.txt maps decoder row to piece.
    (bundle / "vocab.txt").write_bytes(files["ctranslate2/shared_vocabulary.txt"].read_bytes())
    tokenizer = build_tokenizer(files["tokenizer.model"])
    (bundle / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")

    licence = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/Theodotus1243/ukrainian-accentor-transformer/{UPSTREAM_COMMIT}/LICENSE.md"
    ).read()
    (bundle / "LICENSE.ukrainian-accentor-transformer").write_bytes(licence)

    meta = {
        "source": {
            "package": "ukrainian-accentor-transformer",
            "version": "0.1.0",
            "license": "MIT",
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "license_url": f"{UPSTREAM_URL}/blob/{UPSTREAM_COMMIT}/LICENSE.md",
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "model_origin": "CTranslate2 TransformerSpec (4+4 layers, d 256, 8 heads, ffn 1024, "
                            "shared SentencePiece vocabulary of 4095 pieces) read from "
                            "ctranslate2/model.bin and re-exported to ONNX",
            "training_data": "lang-uk News corpus, stressed with ukrainian-word-stress "
                             "(per the upstream README)",
            "training_data_license": "ukrainian-word-stress labels: MIT. lang-uk News corpus: "
                                     "a row of the UberText section of "
                                     "https://lang.org.ua/en/corpora/, which states the "
                                     "texts are distributed on condition of Fair Use for "
                                     "statistical and scientific analysis; the row itself "
                                     "carries no licence field",
        },
        "backend": "uk_accentor",
        "lang": "uk",
        "max_chunk_tokens": 30,
        "split_tokens": [".", ",", "!", "?"],
        "beam_size": 2,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "max_decoding_length": 256,
        "deps": ["onnxruntime", "numpy", "tokenizers"],
        "no_torch": True,
    }
    (bundle / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    for path in sorted(bundle.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(out_dir)}\t{path.stat().st_size}\t{digest}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
