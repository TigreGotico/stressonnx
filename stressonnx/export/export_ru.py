"""Export silero_stress 'ru' accentor to ONNX (numeric core) + dump data files.

torch is used ONLY here at export time. Runtime is onnxruntime + numpy.
"""
import os
import gzip
import json
import numpy as np
import torch

from silero_stress import load_accentor

HERE = os.path.dirname(os.path.abspath(__file__))

acc = load_accentor("ru")
m = acc.accentor.model
m.eval()

W = m.embedding.weight.detach().cpu().numpy().astype(np.float32)  # [126523, 16]
ngram_dict = dict(m.embedding.ngram_dict)
exceptions = dict(acc.accentor.exceptions)           # word -> [stress_char_idx, yo_char_idx]
homodict_keys = sorted(acc.homosolver.homodict.keys())     # skip_stress_words
yohomodict_keys = sorted(acc.homosolver.yohomodict.keys())  # skip_yo_words

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
onnx_path = os.path.join(HERE, "accentor.onnx")
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

# Save embedding weight (fp32) compressed.
np.save(os.path.join(HERE, "embedding.npy"), W)

# Save ngram dict (gram -> id). Use gzip text, "gram\tid" per line.
with gzip.open(os.path.join(HERE, "ngram_dict.txt.gz"), "wt", encoding="utf-8") as f:
    for gram, idx in ngram_dict.items():
        f.write(f"{gram}\t{idx}\n")

# Save exceptions: word\tstress\tyo
with gzip.open(os.path.join(HERE, "exceptions.txt.gz"), "wt", encoding="utf-8") as f:
    for w, (s, y) in exceptions.items():
        f.write(f"{w}\t{s}\t{y}\n")

# Save skip-word sets (homograph keys) one per line.
with gzip.open(os.path.join(HERE, "skip_stress_words.txt.gz"), "wt", encoding="utf-8") as f:
    f.write("\n".join(homodict_keys))
with gzip.open(os.path.join(HERE, "skip_yo_words.txt.gz"), "wt", encoding="utf-8") as f:
    f.write("\n".join(yohomodict_keys))

meta = {"emb_dim": int(emb_dim), "n_ngrams": len(ngram_dict),
        "n_exceptions": len(exceptions),
        "n_skip_stress": len(homodict_keys), "n_skip_yo": len(yohomodict_keys),
        "unk_id": int(ngram_dict["UNK"])}
with open(os.path.join(HERE, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(meta)

# ---- numeric verification: ONNX classifiers vs torch full model ----
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

# (1) classifiers must match torch classifiers EXACTLY on identical pooled input
with torch.no_grad():
    t_stress_clf = m.stress_clf(torch.tensor(pooled)).numpy()
    t_yo_clf = m.yo_clf(torch.tensor(pooled)).numpy()
d_stress = np.abs(o_stress - t_stress_clf).max()
d_yo = np.abs(o_yo - t_yo_clf).max()
print("classifier stress max abs diff:", d_stress)
print("classifier yo max abs diff:", d_yo)
assert d_stress < 1e-3
assert d_yo < 1e-3

# (2) end-to-end argmax (the decode-relevant quantity) must match the full torch model
with torch.no_grad():
    f_stress, f_yo = m(test_words)
assert (o_stress.argmax(1) == f_stress.numpy().argmax(1)).all()
assert (o_yo.argmax(1) == f_yo.numpy().argmax(1)).all()
print("NUMERIC VERIFY OK (classifiers exact, full-path argmax exact)")
