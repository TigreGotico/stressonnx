"""Reference JSON-lines stress server for ``run.py --external``.

Serves one stressonnx model over the protocol in ``benchmarks/external.py``::

    python benchmarks/run.py --lang uk --external "silero-jsonl=python benchmarks/jsonl_stressonnx.py --model silero"

``--cache-dir`` points the model at a local bundle tree (``<dir>/<hf
subdir>/<file>``), so a bundle can be scored before it is published.  A
competitor writes the same loop around its own stress function.
"""
import argparse
import json
import sys

from stressonnx import make_stressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    stressors = {}
    for line in sys.stdin:
        req = json.loads(line)
        try:
            lang = req["lang"]
            if lang not in stressors:
                stressors[lang] = make_stressor(model=args.model, lang=lang,
                                                 cache_dir=args.cache_dir)
            reply = {"text": stressors[lang](req["text"])}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
