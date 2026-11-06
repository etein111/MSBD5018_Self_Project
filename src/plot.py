"""Generate report figures (report/figures/*.png) from the final results.

Numbers mirror results/*.json (regenerate via the run_phase1_*.sh /
run_phase2_finetune.sh scripts). Model colors are fixed across every figure:

    gpt2 = blue, flan-t5 = orange, bert = aqua, roberta = violet
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FIG_DIR = Path("report/figures")

# Validated categorical palette (light mode), fixed per model.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
GRAY = "#c3c2b7"  # baseline axis


def _style(ax, ymax=1.0, ystep=0.1):
    ax.set_ylim(0, ymax)
    ax.set_yticks([i * ystep for i in range(int(ymax / ystep) + 1)])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("left",):
        ax.spines[s].set_visible(False)


def _bar(ax, xs, heights, color, label=None, width=0.62):
    ys = [0.0 if h is None else h for h in heights]
    bars = ax.bar(xs, ys, width=width, color=color, label=label)
    for b, h in zip(bars, heights):
        if h is not None:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.02, "—",
                    ha="center", va="bottom", fontsize=8, color=MUTED)
    return bars


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelcolor": INK,
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig1_nli_prompt_vs_finetune():
    methods = ["gpt2\nzero-shot", "flan-t5\nzero-shot", "flan-t5\nfew-shot k=4",
               "flan-t5 few-shot\n+yes/maybe/no", "BERT\nfine-tune", "RoBERTa\nfine-tune"]
    matched = [0.356, 0.365, 0.450, 0.633, 0.750, 0.831]
    mismatched = [0.352, 0.367, 0.450, None, 0.766, 0.828]
    colors = [BLUE, ORANGE, ORANGE, ORANGE, AQUA, VIOLET]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, data, title in ((axes[0], matched, "matched"), (axes[1], mismatched, "mismatched")):
        _style(ax)
        _bar(ax, range(len(methods)), data, colors, width=0.62)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=7.5, color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_ylabel("accuracy" if ax is axes[0] else "")
    fig.suptitle("NLI: prompting vs fine-tuning (MNLI dev, 4924 examples)",
                 fontsize=11, color=INK, y=1.02)
    fig.savefig(FIG_DIR / "fig1_nli_prompt_vs_finetune.png")
    plt.close(fig)


def fig2_verbalizer():
    verbalizers = ["label_words\n(multi-subword)", "A/B/C\n(single-token)",
                   "yes/maybe/no\n(single-token)"]
    flan = [0.450, 0.493, 0.633]
    gpt2 = [0.315, 0.363, 0.366]
    x = range(len(verbalizers))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _style(ax)
    b1 = _bar(ax, [i - w / 2 for i in x], flan, ORANGE, width=w)
    b2 = _bar(ax, [i + w / 2 for i in x], gpt2, BLUE, width=w)
    ax.set_xticks(list(x))
    ax.set_xticklabels(verbalizers, fontsize=8.5, color=INK)
    ax.set_ylabel("accuracy")
    ax.set_title("Verbalizer ablation (few-shot k=4, matched)", fontsize=11, color=INK)
    ax.legend([b1[0], b2[0]], ["flan-t5-base", "gpt2"], frameon=False,
              loc="upper left", fontsize=9, labelcolor=INK)
    fig.savefig(FIG_DIR / "fig2_verbalizer.png")
    plt.close(fig)


def fig3_hallucination():
    configs = ["gpt2\nlabel_words", "flan-t5\nlabel_words", "flan-t5\nyes/maybe/no"]
    f1 = [0.678, 0.000, 0.877]
    acc = [0.559, 0.270, 0.801]
    colors = [BLUE, ORANGE, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, data, title in ((axes[0], acc, "accuracy"), (axes[1], f1, "F1")):
        _style(ax)
        _bar(ax, range(len(configs)), data, colors)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=8, color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_ylabel(title if ax is axes[0] else "")
    fig.suptitle("Hallucination detection (entailment_factual rule)", fontsize=11,
                 color=INK, y=1.02)
    fig.savefig(FIG_DIR / "fig3_hallucination.png")
    plt.close(fig)


def fig4_bias():
    models = ["bert-base\nuncased", "roberta-base"]
    ratio = [0.625, 0.638]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    _style(ax, ymax=1.0)
    _bar(ax, range(len(models)), ratio, [AQUA, VIOLET])
    ax.axhline(0.5, color=GRAY, linewidth=1.2, linestyle="--")
    ax.text(1.45, 0.505, "no bias (0.5)", fontsize=8, color=MUTED, va="bottom")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9, color=INK)
    ax.set_ylabel("stereotypical-preference ratio")
    ax.set_title("Race bias (CrowS-Pairs race-color, 80 pairs)", fontsize=11, color=INK)
    fig.savefig(FIG_DIR / "fig4_bias.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_nli_prompt_vs_finetune()
    fig2_verbalizer()
    fig3_hallucination()
    fig4_bias()
    print("figures written to", FIG_DIR)
