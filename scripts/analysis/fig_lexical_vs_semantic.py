"""Plot RQ3: lexical vs. semantic, schema and data F1 in a single row.

Three panels in a 1x3 grid (one per LLM). Each panel shows the four cells
schema and data F1 per condition on the x-axis, with paired
bars for the lexical (grep) and semantic configurations per cell. Error
bars are per-query standard errors.
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
OUT = ROOT / "figures" / "rq3_grep_vs_semantic_f1.pdf"

LLMS = [
    ("DeepSeek-V3.2", "deepseek"),
    ("GPT-5.4",       "gpt5"),
    ("Qwen3.5-27B",   "qwen"),
]
# Lexical vs. semantic across the three conditions with a defined answer.
# UNDER is excluded: it is scored behaviorally, not by answer quality.
CATS = ["BASE", "SYN", "TYPO"]
CONFIG_COLORS = {"grep": "#9aa0a6", "semantic": "#1f77b4"}
CONFIG_LABELS = {"grep": "lexical", "semantic": "semantic"}


def per_query_means(traces: list[dict], key: str) -> dict[str, float]:
    qm: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        qm[t["query_id"]].append(t["adaptive_metrics"][key])
    return {q: statistics.mean(v) for q, v in qm.items()}


def cell_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return m, se




def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


METRIC_KEYS = {"schema_f1": "schema_f1", "data_f1": "best_f1"}


def collect():
    out = {}
    for name, folder in LLMS:
        records = load_traces(folder)
        out[name] = {}
        for c in CATS:
            out[name][c] = {}
            grep_traces = [
                t for t in records
                if t.get("approach") == "agentic_grep" and t.get("query_set") == c
            ]
            sem_traces = [
                t for t in records
                if t.get("approach") == "agentic_semantic" and t.get("query_set") == c
            ]
            for label, key in METRIC_KEYS.items():
                gq = per_query_means(grep_traces, key)
                sq = per_query_means(sem_traces, key)
                shared = sorted(set(gq) & set(sq))
                g = [gq[q] for q in shared]
                s = [sq[q] for q in shared]
                out[name][c][label] = {
                    "grep": cell_stats(g),
                    "semantic": cell_stats(s),
                }
    return out


def plot(data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    width = 0.55

    cells = [(c, m) for c in CATS for m in ("schema_f1", "data_f1")]
    x = np.arange(len(cells)) * 1.6
    xlabels = [
        f"{c}\n{'schema' if m == 'schema_f1' else 'data'}"
        for c, m in cells
    ]

    for col, (name, _) in enumerate(LLMS):
        ax = axes[col]
        for i, key in enumerate(("grep", "semantic")):
            vals = [data[name][c][m][key][0] for c, m in cells]
            errs = [data[name][c][m][key][1] for c, m in cells]
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                yerr=errs, capsize=2.0,
                error_kw={"elinewidth": 0.7, "ecolor": "black"},
                label=CONFIG_LABELS[key],
                color=CONFIG_COLORS[key],
                edgecolor="black", linewidth=0.4,
            )
            for b, v, se in zip(bars, vals, errs):
                label = f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.2f}"
                ax.text(b.get_x() + b.get_width() / 2, v + se + 0.02,
                        label, ha="center", va="bottom",
                        fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=15)
        ax.set_ylim(0, 1.30)
        ax.set_title(name, fontsize=20)
        if col == 0:
            ax.set_ylabel(r"$F_1$", fontsize=20)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="y", labelsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    plot(collect())
