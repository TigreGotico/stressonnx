"""Build the ``ru_stressrnn/`` bundle from an upstream StressRNN checkout.

No torch and no export step: upstream ships ``models/model.onnx`` (keras2onnx
export of the RusStress BiLSTM) and the bundle takes it verbatim.  The
exception dictionary is gzipped; ``meta.json`` records the source commit and
licence; the upstream LICENSE travels with the bundle.

Usage::

    git clone https://github.com/dbklim/StressRNN /tmp/StressRNN
    git -C /tmp/StressRNN checkout 0afefac146033b41ef558ac356989c87ef764c79
    python export/export_stressrnn.py /tmp/StressRNN <out_dir>

writes ``<out_dir>/ru_stressrnn/{model.onnx, exception_dictionary.txt.gz,
meta.json, LICENSE.stressrnn}`` and prints each file's size and sha256.
"""
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/dbklim/StressRNN"
UPSTREAM_COMMIT = "0afefac146033b41ef558ac356989c87ef764c79"


def main(upstream: Path, out_dir: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"{upstream} is at {head}, expected {UPSTREAM_COMMIT}")

    bundle = out_dir / "ru_stressrnn"
    bundle.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(upstream / "stressrnn/models/model.onnx", bundle / "model.onnx")
    shutil.copyfile(upstream / "LICENSE", bundle / "LICENSE.stressrnn")

    text = (upstream / "stressrnn/dicts/exception_dictionary.txt").read_bytes()
    with gzip.GzipFile(bundle / "exception_dictionary.txt.gz", "wb", mtime=0) as fh:
        fh.write(text)

    meta = {
        "source": {
            "package": "stressrnn",
            "version": "0.2.2",
            "license": "Apache-2.0",
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "license_url": f"{UPSTREAM_URL}/blob/{UPSTREAM_COMMIT}/LICENSE",
            "model_origin": "RusStress BiLSTM (Ponomareva et al. 2017), "
                            "keras2onnx export shipped upstream as models/model.onnx",
            "dictionary_origin": "A. A. Zaliznyak, Grammatical Dictionary, via "
                                 "http://odict.ru/ (upstream dicts/exception_dictionary.txt)",
        },
        "backend": "stressrnn",
        "lang": "ru",
        "max_input_len": 40,
        "chars": "'-_абвгдежзийклмнопрстуфхцчшщъыьэюяё",
        "accuracy_threshold": 0.75,
        "deps": ["onnxruntime", "numpy"],
        "no_torch": True,
    }
    (bundle / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    for path in sorted(bundle.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(out_dir)}\t{path.stat().st_size}\t{digest}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
