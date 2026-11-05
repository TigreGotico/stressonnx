"""Stress function served by an external command over JSON lines.

A competitor can run in its own virtual environment: the harness starts the
command once and talks to it through stdin and stdout.  For each input the
harness writes one line ``{"lang": "<tag>", "text": "<plain text>"}`` and
reads one line ``{"text": "<text with U+0301 after each stressed vowel>"}``.
Replies come in request order.  A reply ``{"error": "<message>"}`` fails
that input only.  The command exits when its stdin closes.

``benchmarks/jsonl_stressonnx.py`` is a reference server built on stressonnx.
"""
import json
import shlex
import subprocess


class JsonlCommand:
    def __init__(self, command: str, lang: str):
        self.command = command
        self.lang = lang
        self._proc = subprocess.Popen(
            shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )

    def __call__(self, text: str) -> str:
        self._proc.stdin.write(json.dumps({"lang": self.lang, "text": text}, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.command!r} exited with code {self._proc.wait()}")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["text"]

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait(timeout=60)
