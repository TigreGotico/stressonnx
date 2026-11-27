"""Default-model accuracy on each GOLD set of benchmarks/RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import save
from results import gold_accuracy

from stressonnx import DEFAULT_MODEL

data = dict(sorted(((f"{lang} {name} ({DEFAULT_MODEL[lang]})", acc[DEFAULT_MODEL[lang]])
                    for (lang, name), acc in gold_accuracy().items()), key=lambda r: r[1]))


def draw(fig, theme, data):
    fig.get_layout_engine().set(w_pad=0.15)
    ax = fig.subplots()
    ax.barh(list(data), list(data.values()), color=theme["series"][0])
    for y, v in enumerate(data.values()):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=theme["text"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("word accuracy, default model")
    ax.tick_params(axis="y", labelsize=8)


save(str(Path(__file__).parent / "gold-accuracy"), draw, data, width=7.6)
