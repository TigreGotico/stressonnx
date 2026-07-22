"""Export silero_stress main accentors ('ukr', 'bel') to ONNX + data files.

torch is used ONLY here at export time.  Runtime is onnxruntime + numpy.
Run once per language, then upload the output directory to HF.

Usage::

    python stressonnx/export/export_main_accentors.py --lang ukr
    python stressonnx/export/export_main_accentors.py --lang bel
"""
import argparse
import gzip
import json
import os

import numpy as np
import torch

from silero_stress import load_accentor

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lang", required=True, choices=["ukr", "bel", "ru"])
parser.add_argument("--out_dir", default=None,
                    help="Output directory (default: same dir as this script)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
out_dir = args.out_dir or HERE
lang = args.lang

# ---------------------------------------------------------------------------
acc = load_accentor(lang)
accentor = acc.accentor
m = accentor.model
m.eval()

W = m.embedding.weight.detach().cpu().numpy().astype(np.float32)
ngram_dict = dict(m.embedding.ngram_dict)
exceptions = dict(accentor.exceptions)   # word -> (stress_char_idx, yo_char_idx)
emb_dim = W.shape[1]


# ---------------------------------------------------------------------------
# Rebuild the stress_clf MLP as a plain nn.Sequential

def rebuild_mlp(scripted_clf):
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


class StressOnly(torch.nn.Module):
    """stress_clf only — ukr/bel have no yo_clf."""
    def __init__(self, model):
        super().__init__()
        self.stress_clf = rebuild_mlp(model.stress_clf)

    def forward(self, pooled):
        return self.stress_clf(pooled)


clf = StressOnly(m)
clf.eval()

dummy = torch.zeros(3, emb_dim, dtype=torch.float32)
onnx_path = os.path.join(out_dir, "accentor.onnx")
torch.onnx.export(
    clf,
    (dummy,),
    onnx_path,
    input_names=["pooled"],
    output_names=["stress_logits"],
    dynamic_axes={"pooled": {0: "batch"}, "stress_logits": {0: "batch"}},
    opset_version=14,
    dynamo=False,
)
print("exported", onnx_path, os.path.getsize(onnx_path))

# Save embedding weight
np.save(os.path.join(out_dir, "embedding.npy"), W)

# ngram dict
with gzip.open(os.path.join(out_dir, "ngram_dict.txt.gz"), "wt", encoding="utf-8") as f:
    for gram, idx in ngram_dict.items():
        f.write(f"{gram}\t{idx}\n")

# exceptions  (word -> stress_char_idx, yo_char_idx)  [yo always -1 for these langs]
with gzip.open(os.path.join(out_dir, "exceptions.txt.gz"), "wt", encoding="utf-8") as f:
    for w, (s, y) in exceptions.items():
        f.write(f"{w}\t{s}\t{y}\n")

# skip_stress / skip_yo  — these langs have no homosolver → empty
with gzip.open(os.path.join(out_dir, "skip_stress_words.txt.gz"), "wt", encoding="utf-8") as f:
    f.write("")
with gzip.open(os.path.join(out_dir, "skip_yo_words.txt.gz"), "wt", encoding="utf-8") as f:
    f.write("")

# re_cond / vowels for the runtime.
# We reconstruct the alphabet from the silero re_cond pattern by parsing its
# character class, then store it as a plain string so the runtime can rebuild
# a valid regex without depending on silero's internal string format.
import re as _re
_alpha_raw = _re.sub(r"\[|\]|'|\s", "", accentor.re_cond.replace("[^", ""))
# Split on commas and take unique non-empty chars
_alpha_chars = sorted(set(c for c in _alpha_raw if c not in (",",)))
_alpha_str = "".join(_alpha_chars)

meta = {
    "family": "main_accentor",
    "has_yo": False,
    "alpha": _alpha_str,
    "vowels": accentor.vowels,
    "emb_dim": int(emb_dim),
    "n_ngrams": len(ngram_dict),
    "n_exceptions": len(exceptions),
    "n_skip_stress": 0,
    "n_skip_yo": 0,
    "unk_id": int(ngram_dict["UNK"]),
}
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(meta)

# ---------------------------------------------------------------------------
# Numeric verification
import onnxruntime as ort


def word_ngrams(text):
    grams = []
    t = "<" + text + ">"
    for i in range(1, len(text) + 3):
        for j in range(len(t) - i + 1):
            grams.append(t[j: j + i])
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

if lang == "ukr":
    test_words = ["привіт", "світ", "молоко", "замок", "голова", "вода"]
elif lang == "bel":
    test_words = ["прывітанне", "свет", "малако", "замак", "галава", "вада"]
else:  # ru
    test_words = ["привет", "свет", "молоко", "замок", "голова", "вода"]

pooled = pool(test_words)
(o_stress,) = sess.run(None, {"pooled": pooled})

with torch.no_grad():
    t_stress = m.stress_clf(torch.tensor(pooled)).numpy()

d_stress = np.abs(o_stress - t_stress).max()
print(f"stress max abs diff: {d_stress}")
assert d_stress < 1e-3

with torch.no_grad():
    f_stress, _ = m(test_words)
assert (o_stress.argmax(1) == f_stress.numpy().argmax(1)).all()
print("NUMERIC VERIFY OK")
