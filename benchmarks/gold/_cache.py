"""Download cache for evaluation data: ``~/.cache/stressonnx-bench/``.

Evaluation data is fetched at run time and never committed.  Set
``STRESSONNX_BENCH_CACHE`` to use another directory.
"""
import os
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("STRESSONNX_BENCH_CACHE",
                            Path.home() / ".cache" / "stressonnx-bench"))


def cached(url: str, name: str) -> Path:
    """Local copy of *url* stored as *name*; downloaded once."""
    path = CACHE / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    part.rename(path)
    return path


def stream_lines(url: str):
    """Decoded lines of *url*, streamed without a local copy."""
    with urllib.request.urlopen(url) as resp:
        for raw in resp:
            yield raw.decode("utf-8")
