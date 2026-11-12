"""Export the silero_stress 'ru' accentor and homosolver to ONNX + data files.

torch is used ONLY here at export time. Runtime is onnxruntime + numpy.

Usage::

    python export/export_ru.py --out_dir /tmp/ru

Accentor files: ``accentor.onnx`` (MLP heads over a pooled n-gram embedding),
``embedding.npy``, ``ngram_dict.txt.gz``, ``exceptions.txt.gz``.

Homosolver files: ``homosolver.onnx`` (BERT encoder + classifier; the int8
word-embedding table and its dequantization stay inside the graph),
``homosolver_vocab.txt.gz`` (WordPiece vocabulary, one token per line in id
order), ``homodict.json.gz`` (homograph → its two stressed variants) and
``homophrases.json.gz`` (homograph → upstream phrase regex, applied before
the model).

``LICENSE.silero-stress`` carries the upstream MIT notice, and ``meta.json``
names the source package and version.
"""
import argparse
import gzip
import importlib.metadata
import json
import math
import os

import numpy as np
import torch

from silero_stress import load_accentor

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
args = parser.parse_args()
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

acc = load_accentor("ru")
m = acc.accentor.model
m.eval()

W = m.embedding.weight.detach().cpu().numpy().astype(np.float32)  # [126523, 16]
ngram_dict = dict(m.embedding.ngram_dict)
exceptions = dict(acc.accentor.exceptions)           # word -> [stress_char_idx, yo_char_idx]
emb_dim = W.shape[1]


def rebuild_mlp(scripted_clf):
    """Rebuild a scripted Linear/ReLU MLP as a fresh nn.Sequential copying weights."""
    layers = []
    for layer in scripted_clf.children():
        if layer.original_name == "Linear":
            w = layer.weight.detach().clone()
            b = layer.bias.detach().clone()
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                lin.weight.copy_(w)
                lin.bias.copy_(b)
            layers.append(lin)
        elif layer.original_name == "ReLU":
            layers.append(torch.nn.ReLU())
        else:
            raise RuntimeError(f"unexpected layer {layer.original_name}")
    return torch.nn.Sequential(*layers)


class Classifiers(torch.nn.Module):
    """Takes pooled (mean) embeddings [N, 16] -> (stress_logits, yo_logits)."""
    def __init__(self, model):
        super().__init__()
        self.stress_clf = rebuild_mlp(model.stress_clf)
        self.yo_clf = rebuild_mlp(model.yo_clf)

    def forward(self, pooled):
        return self.stress_clf(pooled), self.yo_clf(pooled)


clf = Classifiers(m)
clf.eval()

dummy = torch.zeros(3, emb_dim, dtype=torch.float32)
onnx_path = os.path.join(out_dir, "accentor.onnx")
torch.onnx.export(
    clf,
    (dummy,),
    onnx_path,
    input_names=["pooled"],
    output_names=["stress_logits", "yo_logits"],
    dynamic_axes={"pooled": {0: "batch"},
                  "stress_logits": {0: "batch"},
                  "yo_logits": {0: "batch"}},
    opset_version=14,
    dynamo=False,
)
print("exported", onnx_path, os.path.getsize(onnx_path))

np.save(os.path.join(out_dir, "embedding.npy"), W)

# ngram dict (gram -> id): gzip text, "gram\tid" per line.
with gzip.open(os.path.join(out_dir, "ngram_dict.txt.gz"), "wt", encoding="utf-8") as f:
    for gram, idx in ngram_dict.items():
        f.write(f"{gram}\t{idx}\n")

# exceptions: word\tstress\tyo
with gzip.open(os.path.join(out_dir, "exceptions.txt.gz"), "wt", encoding="utf-8") as f:
    for w, (s, y) in exceptions.items():
        f.write(f"{w}\t{s}\t{y}\n")


# ---------------------------------------------------------------------------
# Homosolver: BERT encoder + classifier over the [HOMO] marker and the word
# ---------------------------------------------------------------------------
homo = acc.homosolver
hm = homo.model
bert = hm.bert

# load_accentor() dequantizes the word embeddings in place; the packaged
# int8 table is reloaded so the graph can carry it at a quarter of the size.
with open(os.path.join(os.path.dirname(__import__("silero_stress").__file__),
                       "data", "accentor.pt"), "rb") as fh:
    raw = torch.package.PackageImporter(fh).load_pickle("accentor_models", "accentor")
q_weight = raw.homosolver.model.bert.embeddings.word_embeddings.weight.detach().clone()
assert q_weight.dtype == torch.int8
restored = bert.scale * (q_weight - bert.zero_point)
assert torch.equal(restored, bert.embeddings.word_embeddings.weight), "dequantization mismatch"


def copy_linear(scripted):
    lin = torch.nn.Linear(scripted.weight.shape[1], scripted.weight.shape[0])
    with torch.no_grad():
        lin.weight.copy_(scripted.weight)
        lin.bias.copy_(scripted.bias)
    return lin


def copy_norm(scripted):
    norm = torch.nn.LayerNorm(scripted.weight.shape[0], eps=1e-12)
    with torch.no_grad():
        norm.weight.copy_(scripted.weight)
        norm.bias.copy_(scripted.bias)
    return norm


class Layer(torch.nn.Module):
    def __init__(self, scripted, n_heads):
        super().__init__()
        att = scripted.attention
        self.n_heads = n_heads
        self.head_size = att.self.attention_head_size
        self.query = copy_linear(att.self.query)
        self.key = copy_linear(att.self.key)
        self.value = copy_linear(att.self.value)
        self.att_out = copy_linear(att.output.dense)
        self.att_norm = copy_norm(att.output.LayerNorm)
        self.inter = copy_linear(scripted.intermediate.dense)
        self.out = copy_linear(scripted.output.dense)
        self.out_norm = copy_norm(scripted.output.LayerNorm)

    def heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x):
        q, k, v = self.heads(self.query(x)), self.heads(self.key(x)), self.heads(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v).permute(0, 2, 1, 3)
        ctx = ctx.reshape(x.shape[0], x.shape[1], -1)
        x = self.att_norm(self.att_out(ctx) + x)
        return self.out_norm(self.out(torch.nn.functional.gelu(self.inter(x))) + x)


class HomoSolverONNX(torch.nn.Module):
    """input_ids [B, T], homo_start [B], homo_end [B] -> logits [B, 1].

    No attention mask, as upstream: rows padded to the batch length attend
    to the padding too.
    """

    def __init__(self):
        super().__init__()
        emb = bert.embeddings
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", bert.scale.detach().clone())
        self.register_buffer("zero_point", bert.zero_point.detach().clone())
        self.register_buffer("position", emb.position_embeddings.weight.detach().clone())
        self.register_buffer("token_type", emb.token_type_embeddings.weight[0].detach().clone())
        self.emb_norm = copy_norm(emb.LayerNorm)
        layers = [getattr(bert.encoder.layer, str(i)) for i in range(bert.num_hidden_layers)]
        n_heads = layers[0].attention.self.num_attention_heads
        self.layers = torch.nn.ModuleList(Layer(layer, n_heads) for layer in layers)
        clf = hm.homo_clf
        self.clf_in = copy_linear(getattr(clf, "0"))
        self.clf_out = copy_linear(getattr(clf, "3"))

    def forward(self, input_ids, homo_start, homo_end):
        t = input_ids.shape[1]
        words = self.scale * (self.q_weight[input_ids].float() - self.zero_point)
        x = (words + self.token_type) + self.position[:t]
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer(x)
        steps = torch.arange(t).unsqueeze(0)
        marker = torch.gather(x, 1, homo_start.view(-1, 1, 1).expand(-1, 1, x.shape[2])).squeeze(1)
        inside = ((steps > homo_start.unsqueeze(1)) & (steps < homo_end.unsqueeze(1))).float()
        word = (x * inside.unsqueeze(2)).sum(1) / inside.sum(1, keepdim=True)
        return self.clf_out(torch.relu(self.clf_in(torch.cat([marker, word], dim=1))))


homo_onnx = HomoSolverONNX().eval()
tok = homo.tokenizer
homo_path = os.path.join(out_dir, "homosolver.onnx")


def encode(sentences_and_words):
    """Upstream batching: mark each homograph, tokenize, pad with [PAD]."""
    rows = []
    for sent, word in sentences_and_words:
        i = sent.index(word)
        tagged = homo._find_and_tag_homos(sent)
        raw_mark = next(r for s, _, _, _, r, _ in tagged if s == i)
        rows.append(tok(raw_mark))
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r in rows])
    starts = torch.tensor([r.index(tok.homo_start_id) for r in rows])
    ends = torch.tensor([r.index(tok.homo_end_id) for r in rows])
    return ids, starts, ends


sample = encode([("старинный замок стоит на горе", "замок"),
                 ("все еще идет дождь", "все"),
                 ("он открыл замок ключом и вошел в дом", "замок")])
torch.onnx.export(
    homo_onnx,
    sample,
    homo_path,
    input_names=["input_ids", "homo_start", "homo_end"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                  "homo_start": {0: "batch"},
                  "homo_end": {0: "batch"},
                  "logits": {0: "batch"}},
    opset_version=14,
    dynamo=False,
)
print("exported", homo_path, os.path.getsize(homo_path))

vocab = sorted(tok.vocab.items(), key=lambda kv: kv[1])
assert [i for _, i in vocab] == list(range(len(vocab)))
with gzip.open(os.path.join(out_dir, "homosolver_vocab.txt.gz"), "wt", encoding="utf-8") as f:
    f.write("\n".join(t for t, _ in vocab) + "\n")
with gzip.open(os.path.join(out_dir, "homodict.json.gz"), "wt", encoding="utf-8") as f:
    json.dump(homo.homodict, f, ensure_ascii=False, sort_keys=True)
with gzip.open(os.path.join(out_dir, "homophrases.json.gz"), "wt", encoding="utf-8") as f:
    json.dump({w: p.pattern for w, p in homo.compiled_phrases.items()}, f,
              ensure_ascii=False, sort_keys=True)
assert all(p.flags & __import__("re").IGNORECASE for p in homo.compiled_phrases.values())

dist = importlib.metadata.distribution("silero-stress")
licence = next(f for f in dist.files if f.name == "LICENSE")
with open(os.path.join(out_dir, "LICENSE.silero-stress"), "w", encoding="utf-8") as f:
    f.write(licence.read_text(encoding="utf-8"))

meta = {
    "source": {"package": "silero-stress", "version": dist.version, "license": "MIT",
               "url": "https://github.com/snakers4/silero-stress"},
    "emb_dim": int(emb_dim), "n_ngrams": len(ngram_dict),
    "n_exceptions": len(exceptions),
    "unk_id": int(ngram_dict["UNK"]),
    "homosolver": {
        "window": homo.window_size,
        "n_homographs": len(homo.homodict),
        "n_phrase_words": len(homo.compiled_phrases),
        "pad_id": tok.pad_token_id, "unk_id": tok.unk_token_id,
        "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id,
        "homo_start_id": tok.homo_start_id, "homo_end_id": tok.homo_end_id,
    },
}
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(meta)

# ---- numeric verification: ONNX vs torch ----
import onnxruntime as ort


def word_ngrams(text):
    grams = []
    t = "<" + text + ">"
    for i in range(1, len(text) + 2 + 1):
        for j in range(len(t) - i + 1):
            grams.append(t[j:j + i])
    if len(text) < 1:
        grams.append(text)
    return grams


def pool(words):
    out = np.zeros((len(words), emb_dim), dtype=np.float32)
    for k, w in enumerate(words):
        ids = [ngram_dict[g] for g in word_ngrams(w) if g in ngram_dict]
        if not ids:
            ids = [ngram_dict["UNK"]]
        out[k] = W[ids].mean(0)
    return out


sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test_words = ["привет", "мир", "замок", "молоко", "съешь", "а", "приветствую", "котёнок", "капуста"]
pooled = pool(test_words)
o_stress, o_yo = sess.run(None, {"pooled": pooled})

# (1) classifiers must match torch classifiers on identical pooled input
with torch.no_grad():
    t_stress_clf = m.stress_clf(torch.tensor(pooled)).numpy()
    t_yo_clf = m.yo_clf(torch.tensor(pooled)).numpy()
# Tolerance relative to the largest logit (yo logits reach ~2e5): GEMM
# kernels differ per CPU, and 2 * 128 terms * 2**-23 ≈ 3e-5 bounds the
# float32 reordering error of the 128-wide layers.
d_stress = np.abs(o_stress - t_stress_clf).max() / np.abs(t_stress_clf).max()
d_yo = np.abs(o_yo - t_yo_clf).max() / np.abs(t_yo_clf).max()
print("classifier stress max diff / max logit:", d_stress)
print("classifier yo max diff / max logit:", d_yo)
assert d_stress < 3e-5
assert d_yo < 3e-5

# (2) end-to-end argmax (the decode-relevant quantity) must match the full torch model
with torch.no_grad():
    f_stress, f_yo = m(test_words)
assert (o_stress.argmax(1) == f_stress.numpy().argmax(1)).all()
assert (o_yo.argmax(1) == f_yo.numpy().argmax(1)).all()

# (3) homosolver: ONNX logits vs the scripted upstream model on a padded batch
homo_sess = ort.InferenceSession(homo_path, providers=["CPUExecutionProvider"])
(o_logits,) = homo_sess.run(None, {"input_ids": sample[0].numpy(),
                                   "homo_start": sample[1].numpy(),
                                   "homo_end": sample[2].numpy()})
with torch.no_grad():
    t_logits = hm(*sample).numpy()
d_homo = np.abs(o_logits - t_logits).max() / max(1.0, np.abs(t_logits).max())
print("homosolver logits max diff / max logit:", d_homo, o_logits.ravel(), t_logits.ravel())
assert d_homo < 1e-4
assert ((o_logits > 0) == (t_logits > 0)).all()
print("NUMERIC VERIFY OK (accentor heads, full-path argmax, homosolver logits)")
