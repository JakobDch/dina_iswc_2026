# Results

## Layout

```
results/
├── experiments/<run>/adaptive_evaluation.json   evaluated records of every run of the run directory
├── experiments/<run>/traces/*.json              raw traces (from the trace archive, see the main README)
├── paper/<backbone>/adaptive_evaluation.json    snapshot of the canonical runs read by scripts/analysis/
└── analysis/                                    derived files of the SYN subtype analysis and the analysis log
```

## Run directories

`<backbone>` is `deepseek` (DeepSeek-V3.2), `gpt5` (GPT-5.4) or `qwen` (Qwen3.5-27B). Every run
directory contains both retrieval configurations (`agentic_grep` = lexical, `agentic_semantic` =
semantic in the `approach` field).

| Run | Content | Runs |
|---|---|---|
| `<backbone>_base` | BASE condition, 17 questions × 6 replicates × 2 configurations | 204 |
| `<backbone>_syn` | SYN condition, same design | 204 |
| `<backbone>_typo` | TYPO condition, same design | 204 |
| `<backbone>_under` | UNDER condition, default configuration, 5 questions × 6 replicates × 2 configurations | 60 |
| `<backbone>_under_instructed` | UNDER condition with the prompt rule of Section 3.2 (`UNDER_INSTRUCTED=1`) | 60 |
| `<backbone>_base_instructed` | false-alarm control: BASE questions under the instructed prompt, 17 × 3, semantic retrieval | 51 |
| `chess_heterogeneous` | CHESS on the 17 SYN questions, heterogeneous regime (Appendix C) | 17 |
| `chess_consolidated` | CHESS on the 17 SYN questions, consolidated regime | 17 |
| `chess_heterogeneous_sql_on_consolidated` | the SQL of the heterogeneous run re-executed on the consolidated snapshot | 17 |

Where a run was restarted, the directory can hold more than one trace for the same
(question, configuration, replicate); the loader in `scripts/analysis/canonical.py` keeps the latest.

## Evaluated records

`adaptive_evaluation.json` maps every trace id to its evaluation record: the question (`query_id`,
`query_set`, `query_text`), the configuration (`approach`), the query the system committed to
(`final_sparql`), the matching ground-truth reading (`gt_sparql`), the column pairing (`column_mapping`)
and the metrics in `adaptive_metrics` (`schema_precision`, `schema_recall`, `schema_f1`, data
`best_precision`, `best_recall`, `best_f1`, `retrieval_triple_f1`, `retrieval_path_coherence`, ...).
Records are produced by `scripts/reevaluate_tiered.py <run> --adaptive` and
`scripts/reevaluate_retrieval.py <run>`.

## Raw traces

One JSON file per run, named `<question>_<configuration>_<id>.json`. The main fields:

| Field | Content |
|---|---|
| `query_metadata`, `approach`, `llm_model`, `run_number`, `timestamp` | what was run |
| `retrieval_phases`, `generation_phases` | the agents' turns with messages, declared status signals and candidate queries |
| `tool_invocations` | every tool call with tool name, input summary, output size and latency |
| `all_retrieved_triples`, `final_schema_triples` | schema triples the retrieval agent surfaced and committed to |
| `final_sparql`, `final_results`, `final_result_count` | the committed query and its result |
| `retrieval_iterations`, `generation_iterations` | phase counts (a retry is a run with more than one retrieval iteration) |
| `is_unanswerable`, `unanswerable_reason`, `stop_kind` | runs that ended without a query and why (`missing_schema`, `underspecified`) |
| `timing`, `agent_timing_details`, `token_usage` | run time per agent and tool, token counts where the provider reports them |

## Snapshot and analysis files

`results/paper/<backbone>/adaptive_evaluation.json` concatenates the records of the four condition runs of
one backbone; every record carries `_source_experiment` (the run directory) and `_trace_id`. It is
rebuilt with `scripts/analysis/build_snapshot.py`.

| File | Content | Produced by |
|---|---|---|
| `analysis/syn_group_scores.json` | per SYN question: data, schema and decoy-aware retrieval F1 for BASE/SYN × lexical/semantic | `scripts/analysis/syn_group_scores.py` |
| `analysis/syn_target_ranks.json` | per SYN question: rank of every substituted schema element under the SYN wording, entity mention similarity | `scripts/analysis/syn_target_ranks.py` |
| `analysis/syn_term_embeddings.json` | embedding cache of the substitution phrases | `scripts/analysis/syn_target_ranks.py` |
| `analysis/under_recognition.json` | Table 3 with per-question counts and the reasons the agents gave | `scripts/analysis/table_under.py` |
| `analysis/runtime_stats.json` | latency, tool-call and token statistics per backbone | `scripts/analysis/runtime_stats.py` |
