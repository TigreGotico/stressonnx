"""Export kubataba Russian stress Transformer to two ONNX subgraphs.

Source model: https://github.com/kubataba/Russian-Stress-Accent-Predictor
License: MIT

The model is an encoder-decoder Transformer with autoregressive decoding.
The generate() loop uses data-dependent early exit, so we export two graphs:

  encoder.onnx       — src (1, 256) int64 → memory (1, 256, d_model) float32
  decoder_step.onnx  — memory (1, 256, d_model), tgt (1, t) int64 → logits (1, vocab) float32

The inference loop stays in Python using onnxruntime only (no torch at runtime).

Usage (run on a machine with torch installed, e.g. ser9):
    pip install ruaccent-predictor
    python export_kubataba.py --out /tmp/ru_kubataba
    # then upload:
    huggingface-cli upload TigreGotico/stressonnx-models /tmp/ru_kubataba ru_kubataba
"""
import argparse
import json
import math
import os
import shutil

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Replicate model architecture (matches kubataba source exactly)
# ---------------------------------------------------------------------------

class CharacterEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pos_encoding", self._make_pe(max_len, d_model))
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _make_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.dropout(self.char_embed(x) * math.sqrt(self.d_model) + self.pos_encoding[:, :seq_len, :])


class StressAccentTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embed = CharacterEmbedding(vocab_size, d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        src_emb = self.embed(src)
        tgt_emb = self.embed(tgt)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------

class _EncoderWrapper(nn.Module):
    def __init__(self, model: StressAccentTransformer):
        super().__init__()
        self.embed = model.embed
        self.encoder = model.transformer.encoder

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(src))


def _extract_mha_weights(mha: nn.MultiheadAttention):
    """Return (in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)."""
    return (
        mha.in_proj_weight.detach(),   # (3*d, d)
        mha.in_proj_bias.detach(),     # (3*d,)
        mha.out_proj.weight.detach(),  # (d, d)
        mha.out_proj.bias.detach(),    # (d,)
    )


def _sdpa_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              in_w: torch.Tensor, in_b: torch.Tensor,
              out_w: torch.Tensor, out_b: torch.Tensor,
              n_heads: int, is_causal: bool = False) -> torch.Tensor:
    """Multi-head attention using F.scaled_dot_product_attention (ONNX-friendly)."""
    B, T, D = q.shape
    S = k.shape[1]
    head_dim = D // n_heads

    d = D
    # Project Q, K, V from separate in_proj weights
    wq, wk, wv = in_w[:d], in_w[d:2*d], in_w[2*d:]
    bq, bk, bv = in_b[:d], in_b[d:2*d], in_b[2*d:]

    Q = (q @ wq.T + bq).view(B, T, n_heads, head_dim).transpose(1, 2)  # (B,H,T,hd)
    K = (k @ wk.T + bk).view(B, S, n_heads, head_dim).transpose(1, 2)  # (B,H,S,hd)
    V = (v @ wv.T + bv).view(B, S, n_heads, head_dim).transpose(1, 2)  # (B,H,S,hd)

    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
    attn = attn.transpose(1, 2).contiguous().view(B, T, D)
    return attn @ out_w.T + out_b


class _DecoderStepWrapper(nn.Module):
    """Single autoregressive decoder step using F.scaled_dot_product_attention.

    Bypasses nn.Transformer internals to avoid the _detect_is_causal_mask
    data-dependent guard that blocks torch.export with dynamic sequence lengths.
    """

    def __init__(self, model: StressAccentTransformer):
        super().__init__()
        self.embed = model.embed
        self.output_proj = model.output_proj
        n_heads = model.transformer.decoder.layers[0].self_attn.num_heads
        self.n_heads = n_heads

        # Extract weights from each decoder layer
        layers = model.transformer.decoder.layers
        self.n_layers = len(layers)

        for i, layer in enumerate(layers):
            # Self-attention
            sa_iw, sa_ib, sa_ow, sa_ob = _extract_mha_weights(layer.self_attn)
            self.register_buffer(f"sa_iw_{i}", sa_iw)
            self.register_buffer(f"sa_ib_{i}", sa_ib)
            self.register_buffer(f"sa_ow_{i}", sa_ow)
            self.register_buffer(f"sa_ob_{i}", sa_ob)
            # Cross-attention
            ca_iw, ca_ib, ca_ow, ca_ob = _extract_mha_weights(layer.multihead_attn)
            self.register_buffer(f"ca_iw_{i}", ca_iw)
            self.register_buffer(f"ca_ib_{i}", ca_ib)
            self.register_buffer(f"ca_ow_{i}", ca_ow)
            self.register_buffer(f"ca_ob_{i}", ca_ob)
            # FFN
            self.register_buffer(f"ff1_w_{i}", layer.linear1.weight.detach())
            self.register_buffer(f"ff1_b_{i}", layer.linear1.bias.detach())
            self.register_buffer(f"ff2_w_{i}", layer.linear2.weight.detach())
            self.register_buffer(f"ff2_b_{i}", layer.linear2.bias.detach())
            # LayerNorms
            self.register_buffer(f"ln1_w_{i}", layer.norm1.weight.detach())
            self.register_buffer(f"ln1_b_{i}", layer.norm1.bias.detach())
            self.register_buffer(f"ln2_w_{i}", layer.norm2.weight.detach())
            self.register_buffer(f"ln2_b_{i}", layer.norm2.bias.detach())
            self.register_buffer(f"ln3_w_{i}", layer.norm3.weight.detach())
            self.register_buffer(f"ln3_b_{i}", layer.norm3.bias.detach())

        # Final norm (if present)
        if model.transformer.decoder.norm is not None:
            self.register_buffer("final_ln_w", model.transformer.decoder.norm.weight.detach())
            self.register_buffer("final_ln_b", model.transformer.decoder.norm.bias.detach())
            self._has_final_norm = True
        else:
            self._has_final_norm = False

    def _layer(self, x: torch.Tensor, memory: torch.Tensor, i: int) -> torch.Tensor:
        # Post-norm (norm_first=False): attention/FFN first, then layer norm.
        H = self.n_heads
        sa_iw = getattr(self, f"sa_iw_{i}")
        sa_ib = getattr(self, f"sa_ib_{i}")
        sa_ow = getattr(self, f"sa_ow_{i}")
        sa_ob = getattr(self, f"sa_ob_{i}")
        ca_iw = getattr(self, f"ca_iw_{i}")
        ca_ib = getattr(self, f"ca_ib_{i}")
        ca_ow = getattr(self, f"ca_ow_{i}")
        ca_ob = getattr(self, f"ca_ob_{i}")

        # Self-attention + add + norm
        x = torch.nn.functional.layer_norm(
            x + _sdpa_mha(x, x, x, sa_iw, sa_ib, sa_ow, sa_ob, H, is_causal=True),
            [x.shape[-1]], getattr(self, f"ln1_w_{i}"), getattr(self, f"ln1_b_{i}"),
        )

        # Cross-attention + add + norm
        x = torch.nn.functional.layer_norm(
            x + _sdpa_mha(x, memory, memory, ca_iw, ca_ib, ca_ow, ca_ob, H, is_causal=False),
            [x.shape[-1]], getattr(self, f"ln2_w_{i}"), getattr(self, f"ln2_b_{i}"),
        )

        # FFN + add + norm
        ffn = torch.nn.functional.linear(x, getattr(self, f"ff1_w_{i}"), getattr(self, f"ff1_b_{i}"))
        ffn = torch.nn.functional.relu(ffn)
        ffn = torch.nn.functional.linear(ffn, getattr(self, f"ff2_w_{i}"), getattr(self, f"ff2_b_{i}"))
        return torch.nn.functional.layer_norm(
            x + ffn, [x.shape[-1]],
            getattr(self, f"ln3_w_{i}"), getattr(self, f"ln3_b_{i}"),
        )

    def forward(self, memory: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        x = self.embed(tgt)
        for i in range(self.n_layers):
            x = self._layer(x, memory, i)
        if self._has_final_norm:
            x = torch.nn.functional.layer_norm(
                x, [x.shape[-1]], self.final_ln_w, self.final_ln_b,
            )
        return self.output_proj(x[:, -1, :])  # (batch, vocab)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _run_ort(enc_path: str, dec_path: str, vocab: dict, text: str) -> str:
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    idx2char = {v: k for k, v in vocab.items()}

    pad, bos, eos, unk = vocab["<pad>"], vocab["<s>"], vocab["</s>"], vocab.get("<unk>", 3)
    indices = [bos] + [vocab.get(c, unk) for c in text[:254]] + [eos]
    indices += [pad] * (256 - len(indices))
    src = np.array([indices[:256]], dtype=np.int64)

    memory = enc.run(["memory"], {"src": src})[0]
    tgt = np.array([[bos]], dtype=np.int64)
    for _ in range(256):
        logits = dec.run(["logits"], {"memory": memory, "tgt": tgt})[0]
        next_tok = int(np.argmax(logits[0]))
        tgt = np.concatenate([tgt, [[next_tok]]], axis=1)
        if next_tok == eos:
            break

    chars = []
    for idx in tgt[0, 1:].tolist():
        if idx == eos:
            break
        chars.append(idx2char.get(idx, ""))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="Path to acc_model.pt (default: auto-download via pip package)")
    ap.add_argument("--vocab", default=None,
                    help="Path to vocab.json (default: auto-locate from pip package)")
    ap.add_argument("--out", default="/tmp/ru_kubataba",
                    help="Output directory (default: /tmp/ru_kubataba)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Locate weights + vocab ------------------------------------------------
    if args.weights is None or args.vocab is None:
        try:
            import ruaccent as _pkg  # pip package name is ruaccent-predictor
            pkg_dir = os.path.dirname(_pkg.__file__)
        except ImportError:
            raise SystemExit(
                "ruaccent-predictor not installed and no --weights/--vocab given.\n"
                "  pip install ruaccent-predictor\nor pass explicit paths."
            )
        weights_path = args.weights or os.path.join(pkg_dir, "model", "acc_model.pt")
        vocab_path = args.vocab or os.path.join(pkg_dir, "model", "vocab.json")
    else:
        weights_path, vocab_path = args.weights, args.vocab

    with open(vocab_path) as fh:
        vocab = json.load(fh)
    vocab_size = max(vocab.values()) + 1

    print(f"Vocab size: {vocab_size}")
    print(f"Loading weights from {weights_path} ...")

    model = StressAccentTransformer(vocab_size=vocab_size)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Copy vocab.json -------------------------------------------------------
    dest_vocab = os.path.join(args.out, "vocab.json")
    shutil.copy2(vocab_path, dest_vocab)
    print(f"Copied vocab → {dest_vocab}")

    # Export encoder --------------------------------------------------------
    enc_wrapper = _EncoderWrapper(model)
    enc_wrapper.eval()
    dummy_src = torch.zeros(1, 256, dtype=torch.long)
    enc_path = os.path.join(args.out, "encoder.onnx")
    torch.onnx.export(
        enc_wrapper, dummy_src, enc_path,
        input_names=["src"], output_names=["memory"],
        dynamic_axes={"src": {0: "batch"}, "memory": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported encoder  → {enc_path}")

    # Export decoder step ---------------------------------------------------
    dec_wrapper = _DecoderStepWrapper(model)
    dec_wrapper.eval()
    d_model = model.d_model
    dummy_mem = torch.zeros(1, 256, d_model)
    dummy_tgt = torch.zeros(1, 2, dtype=torch.long)
    dec_path = os.path.join(args.out, "decoder_step.onnx")
    from torch.export import Dim as _Dim
    _tgt_seq = _Dim("tgt_seq", min=1, max=256)
    torch.onnx.export(
        dec_wrapper, (dummy_mem, dummy_tgt), dec_path,
        input_names=["memory", "tgt"], output_names=["logits"],
        dynamic_shapes=(
            {0: _Dim("batch")},
            {0: _Dim("batch2"), 1: _tgt_seq},
        ),
        opset_version=17,
    )
    print(f"Exported decoder  → {dec_path}")

    # Merge external data into the .onnx file so it's self-contained
    import onnx as _onnx
    _m = _onnx.load(dec_path, load_external_data=True)
    _onnx.save_model(_m, dec_path, save_as_external_data=False)
    data_file = dec_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    print(f"Merged external data into {dec_path}")

    # Validate with onnxruntime ---------------------------------------------
    print("Validating with onnxruntime ...")
    result = _run_ort(enc_path, dec_path, vocab, "привет")
    print(f"  'привет' → {result!r}")
    assert "'" in result or any(c in result for c in "аоуыэиеяёю"), \
        f"Expected stress marker in output, got: {result!r}"
    print("Validation passed.")
    print(f"\nFiles ready in {args.out}/")
    print("Upload with:")
    print(f"  huggingface-cli upload TigreGotico/stressonnx-models {args.out} ru_kubataba")


if __name__ == "__main__":
    main()
