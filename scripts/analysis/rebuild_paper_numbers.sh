#!/usr/bin/env bash
# Rebuild every number and figure of the paper from the evaluated runs. Run from the repository root:
#   bash scripts/analysis/rebuild_paper_numbers.sh
# Prerequisites: the raw traces unpacked into results/experiments/<run>/traces/ (see README, "Experiment
# traces"); syn_target_ranks.py additionally needs the embedding index under data/cache/embeddings/ and an
# OPENAI_API_KEY only for phrases missing from results/analysis/syn_term_embeddings.json.
set -eo pipefail
export PYTHONUTF8=1
OUT=${1:-results/analysis/log}
mkdir -p "$OUT"

echo "[1/6] snapshot of the canonical runs"
python scripts/analysis/build_snapshot.py | tee "$OUT/build_snapshot.txt"

echo "[2/6] Table 2 and the F1 values of Figures 3 and 4"
python scripts/analysis/table_conditions.py | tee "$OUT/table_conditions.txt"
python scripts/analysis/table_retry.py | tee "$OUT/table_retry.txt"
python scripts/analysis/handback_outcome.py | tee "$OUT/handback_outcome.txt"

echo "[3/6] Table 3"
python scripts/analysis/table_under.py | tee "$OUT/table_under.txt"

echo "[4/6] Figures 3 and 4"
python scripts/analysis/fig_conditions.py
python scripts/analysis/fig_lexical_vs_semantic.py

echo "[5/6] SYN subtype analysis and Figure 5"
python scripts/analysis/syn_group_scores.py | tee "$OUT/syn_group_scores.txt"
python scripts/analysis/syn_target_ranks.py | tee "$OUT/syn_target_ranks.txt"
python scripts/analysis/fig_syn_subtypes.py | tee "$OUT/fig_syn_subtypes.txt"
python scripts/analysis/syn_search_stats.py | tee "$OUT/syn_search_stats.txt"

echo "[6/6] run time, tool time and TYPO token statistics"
python scripts/analysis/tool_time.py | tee "$OUT/tool_time.txt"
python scripts/analysis/typo_tokens.py | tee "$OUT/typo_tokens.txt"
python scripts/analysis/runtime_stats.py | tee "$OUT/runtime_stats.txt"
echo "done -> $OUT"
