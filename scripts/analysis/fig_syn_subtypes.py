"""Figure 5: the three groups of SYN questions (reachable target, displaced target, renamed entity).

  (a) data F1 per group for the four cells BASE/SYN x lexical/semantic, mean of per-question means
      over the three backbones, SE over questions; single questions as dots where a group has
      <= 3 members; annotated: SYN semantic - BASE semantic.
  (b) per question: rank of the lowest-ranking required schema element under its SYN wording in the
      tool's embedding index (x, log) against the semantic delta (y), SE over the three backbones.

Inputs (run syn_group_scores.py and syn_target_ranks.py first):
  results/analysis/syn_target_ranks.json, results/analysis/syn_group_scores.json,
  results/paper/<model>/adaptive_evaluation.json
Output: figures/rq2_syn_subtypes.pdf (+ .png)

Run from the repository root:  python scripts/analysis/fig_syn_subtypes.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import MODELS, ROOT, load_traces  # noqa: E402

ANALYSIS = ROOT / "results" / "analysis"
OUT = ROOT / "figures" / "rq2_syn_subtypes.pdf"
RANK_CUT = 50

A = {r["n"]: r for r in json.load(open(ANALYSIS / "syn_target_ranks.json", encoding="utf-8"))}
R = {r["n"]: r for r in json.load(open(ANALYSIS / "syn_group_scores.json", encoding="utf-8"))}


def feature(n):
    a = A[n]
    if a["entity"] and not all(e["shared_prefix"] for e in a["entity"]):
        return "entity"
    return "displaced" if a["worst_rank"] > RANK_CUT else "bridged"


GROUPS = [("bridged", "reachable target"), ("displaced", "displaced target"), ("entity", "renamed entity")]
members = {g: sorted(n for n in R if feature(n) == g) for g, _ in GROUPS}
labels = [f'{lbl}\n({", ".join(members[g]) if len(members[g]) <= 3 else "n=" + str(len(members[g]))})' for g, lbl in GROUPS]

SERIES = [("BASE, lexical", "bl", "#d3d6d9"), ("BASE, semantic", "bs", "#a6c8e3"),
          ("SYN, lexical", "sl", "#9aa0a6"), ("SYN, semantic", "ss", "#1f77b4")]
GCOL = {"bridged": "#1f77b4", "displaced": "#b2182b", "entity": "#e08214"}
GMK = {"bridged": "o", "displaced": "s", "entity": "D"}

# ---- per-backbone semantic deltas for panel (b) ----
base, syn = {}, {}
for folder, label in MODELS:
    per_b, per_s = {}, {}
    for t in load_traces(folder):
        if t.get("approach") != "agentic_semantic":
            continue
        f1 = (t.get("adaptive_metrics") or {}).get("best_f1") or 0
        if t.get("query_set") == "BASE":
            per_b.setdefault(t["query_id"][4:], []).append(f1)
        elif t.get("query_set") == "SYN":
            per_s.setdefault(t["query_id"][3:], []).append(f1)
    for n, v in per_b.items():
        base[(label, n)] = statistics.mean(v)
    for n, v in per_s.items():
        syn[(label, n)] = statistics.mean(v)

# ---- figure ----
plt.rcParams.update({"font.size": 13})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.0, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

x = np.arange(len(GROUPS)); w = 0.2
for i, (lbl, tag, color) in enumerate(SERIES):
    m, e = [], []
    for g, _ in GROUPS:
        vals = [R[n][f"{tag}_f1"] for n in members[g]]
        m.append(statistics.mean(vals)); e.append(statistics.stdev(vals) / len(vals) ** .5 if len(vals) > 1 else 0)
    ax.bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=2.5, error_kw={"elinewidth": 0.8, "ecolor": "black"},
           label=lbl, color=color, edgecolor="black", linewidth=0.4)
    for j, (g, _) in enumerate(GROUPS):
        if len(members[g]) <= 3:
            vals = [R[n][f"{tag}_f1"] for n in members[g]]
            ax.scatter([x[j] + (i - 1.5) * w] * len(vals), vals, s=16, color="black", zorder=3)
for j, (g, _) in enumerate(GROUPS):
    bs = statistics.mean(R[n]["bs_f1"] for n in members[g]); ss = statistics.mean(R[n]["ss_f1"] for n in members[g])
    top = max([bs, ss] + ([R[n]["ss_f1"] for n in members[g]] if len(members[g]) <= 3 else []))
    ax.annotate(f"{ss - bs:+.2f}", (x[j] + .5 * w, top + .05), ha="center", fontsize=14, weight="bold",
                color=("#b2182b" if ss - bs < -.1 else "#444444"))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(0, 1.34); ax.set_ylabel("data $F_1$", fontsize=15)
ax.set_title("(a) Answer quality per SYN subtype", fontsize=15)
ax.set_axisbelow(True); ax.yaxis.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=11.5, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.4)

for g, lbl in GROUPS:
    first = True
    for n in members[g]:
        deltas = [syn[(m, n)] - base[(m, n)] for _f, m in MODELS if (m, n) in syn and (m, n) in base]
        y = statistics.mean(deltas); ye = statistics.stdev(deltas) / len(deltas) ** .5 if len(deltas) > 1 else 0
        xv = max(A[n]["worst_rank"], 1)
        bx.errorbar(xv, y, yerr=ye, fmt=GMK[g], ms=8, color=GCOL[g], ecolor=GCOL[g], elinewidth=1, capsize=2.5,
                    markeredgecolor="black", markeredgewidth=.5, zorder=3, label=lbl if first else None)
        first = False
        bx.annotate(n, (xv, y), textcoords="offset points", xytext=(7, -11) if n == "18" else (7, 4), fontsize=11)
bx.axvline(RANK_CUT, color="grey", linestyle="--", linewidth=.9)
bx.axhline(0, color="grey", linewidth=.7)
bx.set_xscale("log")
bx.set_xlabel("rank of the lowest-ranking required schema element\nunder the SYN wording in the embedding index", fontsize=13)
bx.set_ylabel("data $F_1$: SYN semantic $-$ BASE semantic", fontsize=14)
bx.set_title("(b) Rank of the target vs. semantic drop", fontsize=15)
bx.grid(alpha=.35, linestyle=":"); bx.legend(fontsize=11.5, loc="lower left", frameon=False)
bx.tick_params(labelsize=12)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
print("wrote", OUT)
print({g: members[g] for g, _ in GROUPS})
