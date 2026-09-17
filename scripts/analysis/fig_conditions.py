"""Plot RQ2 SPARQL-agent metrics (schema F1, data F1) for lexical configuration.

Two side-by-side panels: schema F1 (left), data F1 (right).
Each panel: x = ambiguity category, grouped bars for the three LLMs.
Error bars are per-query standard errors (mean over runs first, then SE
across queries within the cell).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canonical import load_traces

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "rq2_grep_sparql_f1.pdf"

LLMS = [
    ("DeepSeek-V3.2", "deepseek"),
    ("GPT-5.4",       "gpt5"),
    ("Qwen3.5-27B",   "qwen"),
]
# UNDER is excluded: it is scored behaviorally, not by result quality.
CATS = ["BASE", "SYN", "TYPO"]
COLORS = {"DeepSeek-V3.2": "#1f77b4", "GPT-5.4": "#2ca02c", "Qwen3.5-27B": "#d62728"}


def per_query_means(traces: list[dict], key: str) -> list[float]:
    qm: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        qm[t["query_id"]].append(t["adaptive_metrics"][key])
    return [statistics.mean(v) for v in qm.values()]


def cell_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return m, se


def collect() -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    out: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for name, folder in LLMS:
        records = load_traces(folder)
        out[name] = {}
        for c in CATS:
            traces = [
                t for t in records
                if t.get("approach") == "agentic_grep" and t.get("query_set") == c
            ]
            out[name][c] = {
                "schema_f1": cell_stats(per_query_means(traces, "schema_f1")),
                "data_f1":   cell_stats(per_query_means(traces, "best_f1")),
            }
    return out


def plot(data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0), sharey=True)
    n_llm = len(LLMS)
    width = 0.55
    x = np.arange(len(CATS)) * 2.0

    for ax, metric, title in (
        (axes[0], "schema_f1", r"Schema $F_1$"),
        (axes[1], "data_f1",   r"Data $F_1$"),
    ):
        for i, (name, _) in enumerate(LLMS):
            cells = [data[name][c][metric] for c in CATS]
            vals = [m for m, _ in cells]
            errs = [se for _, se in cells]
            offset = (i - (n_llm - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals, width,
                yerr=errs, capsize=2.5,
                error_kw={"elinewidth": 0.8, "ecolor": "black"},
                label=name, color=COLORS[name],
                edgecolor="black", linewidth=0.4,
            )
            for b, v, se in zip(bars, vals, errs):
                label = f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.2f}"
                ax.text(b.get_x() + b.get_width() / 2,
                        v + se + 0.02,
                        label, ha="left", va="bottom",
                        fontsize=16, rotation=30,
                        rotation_mode="anchor")
        ax.set_xticks(x)
        ax.set_xticklabels(CATS)
        ax.set_ylim(0, 1.35)
        ax.set_title(title, fontsize=20)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="both", labelsize=17)

    axes[0].set_ylabel(r"$F_1$", fontsize=19)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    plot(collect())