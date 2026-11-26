"""Accuracy of every ru model on the russtress GOLD set of benchmarks/RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import save
from results import gold_accuracy

from stressonnx import DEFAULT_MODEL

data = dict(sorted(gold_accuracy()[("ru", "russtress")].items(), key=lambda r: r[1]))


def draw(fig, theme, data):
    fig.get_layout_engine().set(w_pad=0.15)
    ax = fig.subplots()
    # russtress is the training corpus of StressRNN, so its score is not held out.
    labels = [f"{m} (training data)" if m == "stressrnn" else m for m in data]
    colors = [theme["series"][0] if m == DEFAULT_MODEL["ru"] else theme["grid"] for m in data]
    ax.barh(labels, list(data.values()), color=colors)
    for y, v in enumerate(data.values()):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=theme["text"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("word accuracy, ru russtress (GOLD), default model in colour")
    ax.tick_params(axis="y", labelsize=8)


save(str(Path(__file__).parent / "ru-models"), draw, data)
