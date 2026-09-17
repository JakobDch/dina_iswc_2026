# Vocabulary mismatch and underspecification in tool-based KGQA over virtual knowledge graphs

Code, question corpus, ground truth, evaluation framework and experiment results for the paper

> Jakob Deich, Tobias Meisen, André Pomp. *When does the user's wording break knowledge graph question
> answering? Vocabulary mismatch and underspecification in tool-based systems.* Submitted to the
> Semantic Web Journal, 2026.

The paper studies how far a tool-using KGQA system tolerates questions whose wording departs from the
schema vocabulary. A two-agent system (retrieval agent, SPARQL generation agent) answers questions over
virtual knowledge graphs defined by R2RML/RML mappings. Its three linking tools can be swapped between a
**lexical** (substring) and a **semantic** (embedding-based) matching mechanism while everything else
stays fixed. Four input conditions (BASE, SYN, TYPO, UNDER) are run over five datasets and three LLM
backbones, and every run is scored against an adaptive ground truth that admits every faithful reading
of a question.

## Where to find what

| Paper | Content | Path |
|---|---|---|
| Section 3.1–3.2 | Agents, router, status signals, prompts | [src/agents/](src/agents/) (`orchestrator.py`, `retrieval/grep_agent.py` = lexical, `retrieval/semantic_agent.py` = semantic, `generation/sparql_agent.py`); prompt rule of the instructed configuration in [src/config.py](src/config.py) (`UNDERSPECIFIED_INSTRUCTION_*`) |
| Section 3.3, Appendix B | Schema extraction from mappings by minimal materialisation | [scripts/rml_schema_extractor.py](scripts/rml_schema_extractor.py); extracted schemas in `data/datasets/<DATASET>/class_semantic_models/` |
| Section 3.4 | Linking and navigation tools, instance-value sampler | [src/tools/retrieval_tools.py](src/tools/retrieval_tools.py), [src/tools/data_instance_tools.py](src/tools/data_instance_tools.py), [src/tools/sparql_tools.py](src/tools/sparql_tools.py) |
| Section 4.1 | Result evaluation: adaptive ground truth, column matching, schema and data metrics | [src/evaluation/tiered_metrics_tuples.py](src/evaluation/tiered_metrics_tuples.py), [src/evaluation/column_signatures.py](src/evaluation/column_signatures.py) |
| Section 4.2 | Retrieval evaluation: retrieval ground truth, triple F1, path coherence | [src/evaluation/retrieval_ground_truth.py](src/evaluation/retrieval_ground_truth.py), [src/evaluation/retrieval_metrics.py](src/evaluation/retrieval_metrics.py) |
| Section 4.3 | Search count and search-failure rate | [scripts/analysis/table_conditions.py](scripts/analysis/table_conditions.py) |
| Section 5.1 | Datasets, mappings, ontologies | [mappings/](mappings/), see [Datasets](#datasets) |
| Section 5.2 | Question corpus (56 formulations) with all readings and column labels | [data/queries/](data/queries/README.md) |
| Section 5.2 | Rename table (controlling for memorisation) | [scripts/rename_mapping.json](scripts/rename_mapping.json), [scripts/rename_ontology.py](scripts/rename_ontology.py) |
| Section 5.3 | Backbones, configuration, run protocol | [src/config.py](src/config.py), [scripts/run_full_experiment.py](scripts/run_full_experiment.py) |
| Section 6, Tables 2–3, Figures 3–5 | Evaluated runs and the scripts that produce every number | [results/](results/README.md), [scripts/analysis/](scripts/analysis/) |
| Appendix C | Text2SQL reference point (CHESS) | [tools/chess/](tools/chess/), `scripts/create_chess_*.py`, `scripts/evaluate_chess_*.py` |

## Repository layout

```
src/                     agentic pipeline: agents, tools, router, tracing, evaluation
scripts/                 experiment runner, re-evaluation, dataset and schema preparation, CHESS scripts
scripts/analysis/        the scripts behind every table and figure of the paper
data/queries/            question corpus and adaptive ground truth (Python module + JSON/CSV export)
data/datasets/<DS>/      class semantic models (one Turtle file per class) read by the retrieval agents
data/cache/              ground-truth result cache and the embedding index of the semantic tools
mappings/<dataset>/      R2RML/RML mapping, ontology and Ontop configuration per dataset
results/experiments/     one directory per run (evaluated records; raw traces from the archive, see below)
results/paper/           per-backbone snapshot of the canonical runs read by the analysis scripts
results/analysis/        derived analysis files (SYN subtype grouping, term embeddings)
figures/                 figures of the paper
tools/chess/             vendored CHESS Text2SQL pipeline (Apache 2.0)
tests/                   unit tests of the evaluation framework and the router
```

## Setup

Python 3.11 or newer and Docker (for the Ontop SPARQL endpoints).

```bash
pip install -e .
cp .env.example .env   # API keys of the providers you use; OPENAI_API_KEY is also needed for the embeddings
```

Supported backbones are listed in `LLM_MODELS` in [scripts/run_full_experiment.py](scripts/run_full_experiment.py).
The paper uses `gpt5` (GPT-5.4, reasoning effort medium), `deepseek-or` (DeepSeek-V3.2 via OpenRouter) and
`qwen35-or` (Qwen3.5-27B-Instruct via OpenRouter); sampling temperature 0.1 for the two open-weight models,
retrieval and generation phase limits of 3 iterations each, embeddings with `text-embedding-3-large`.

### SPARQL endpoints

Every dataset is exposed as a virtual knowledge graph through its own Ontop endpoint over a MySQL
instance. The runner distributes concurrent runs over replicated endpoint *slots* so that runs never
interfere with each other.

```bash
python scripts/initialize_slot_mappings.py --slots 5 --clean      # per-slot copies of the mappings
docker compose -f docker-compose.yml -f docker-compose.slots.yml up -d
python scripts/preindex_embeddings.py                             # embedding index of the semantic tools (optional, cached)
```

Endpoint URLs default to `http://localhost:8080..8085/sparql` (EDU, TRN, NRG, BSBM, LCA); the slot
variants use the neighbouring ports defined in [docker-compose.slots.yml](docker-compose.slots.yml).

### Datasets

| Code | Domain | Source | Mapping and ontology |
|---|---|---|---|
| EDU | University | LUBM (Guo, Pan, Heflin 2005) | `mappings/edu-small/` |
| TRN | Public transit | GTFS-Madrid-Bench (Chaves-Fraga et al. 2020) | `mappings/trn-small/` |
| NRG | Oil and gas | NPD benchmark (Lanti et al. 2015) | `mappings/nrg-small/` |
| BSBM | E-commerce | Berlin SPARQL Benchmark (Bizer, Schultz 2009) | `mappings/bsbm/` |
| LCA | Life-cycle assessment | ORIENTING (Baumanns et al. 2026) | `mappings/lca/` |

The raw data is not part of this repository. Download each dataset from its upstream distribution and
convert it into the relational form the mappings expect with the converters under `scripts/`
(`convert_lubm_owl_to_sql.py`, `convert_lubm_to_bulk.py`, `csv_to_sql.py`, `convert_to_bulk.py`). Table 1 of
the paper reports the schema, mapping and source statistics per dataset.

For EDU, TRN and NRG the identifiers of the ontologies, mappings and gold queries were rewritten
(Section 5.2, "Controlling for memorization"): [scripts/rename_mapping.json](scripts/rename_mapping.json)
lists every original class and property name and its replacement (45 classes, 97 properties);
[scripts/rename_ontology.py](scripts/rename_ontology.py) applies the table. The files in this repository
already carry the rewritten names.

### Class semantic models (Section 3.3, Appendix B)

The retrieval agents do not search the mapping documents. They search one short Turtle file per class
that lists the class's data properties with datatypes and its object properties with target classes:

```turtle
eduo:Learner eduo:label xsd:string ;
    eduo:belongsTo eduo:Division ;
    eduo:enrolledIn eduo:Course ;
    eduo:mentor eduo:Educator .
```

These files ship under `data/datasets/<DATASET>/class_semantic_models/`. They were produced with
[scripts/rml_schema_extractor.py](scripts/rml_schema_extractor.py), which loads a bounded sample of the
source rows (at most 100 per table) into an SQLite stub, materialises the unmodified mapping with
Morph-KGC, reads the schema off the resulting RDF and adds entries for the superclasses declared in the
ontology:

```bash
python scripts/rml_schema_extractor.py mappings/edu-small/mapping.ttl mappings/edu-small/ontology.owl \
    <EDU SQL dump or CSV folder> data/datasets/EDU/class_semantic_models
```

## Question corpus and ground truth (Sections 4.1 and 5.2)

The corpus and its adaptive ground truth live in [data/queries/](data/queries/README.md): 17 paired
questions in the conditions BASE, SYN and TYPO plus 5 UNDER questions, every question with all admissible
SPARQL readings and the REQUIRED/ACCEPTABLE labels of the result columns. The README there lists every
question. `corpus.json` and `corpus.csv` are exports of the Python module for reading without code.

## Running the experiments (Section 5.3)

```bash
# both retrieval configurations, all four conditions, six replicate runs per cell
python scripts/run_full_experiment.py --models gpt5 --approaches grep semantic \
    --query-sets BASE SYN TYPO UNDER --runs 6 --serialize-datasets --name gpt5_all

# instructed configuration (RQ3): the prompt rule of Section 3.2 is appended to every agent
UNDER_INSTRUCTED=1 python scripts/run_full_experiment.py --models gpt5 --approaches grep semantic \
    --query-sets UNDER --runs 6 --name gpt5_under_instructed
```

`--approaches grep` is the lexical configuration, `semantic` the embedding-based one. Runs land in
`results/experiments/<name>/`: one JSON trace per run under `traces/` (message history, tool-call log
with inputs and output sizes, candidate SPARQL queries, execution results, status signals, timing and token
counts), a `checkpoint.json` for `--resume`, and after evaluation `adaptive_evaluation.json`.

### Evaluation (Section 4)

```bash
python scripts/reevaluate_tiered.py <run> --adaptive     # result-level metrics against the adaptive ground truth
python scripts/reevaluate_retrieval.py <run>            # retrieval ground truth, triple F1, path coherence
```

`reevaluate_tiered.py` executes every admissible reading against the endpoints, or reads their results
from `data/cache/ground_truth_full.json` (`scripts/extract_ground_truth_cache.py`) when no endpoint is
running, matches the system's result columns to the ground-truth columns by their values, selects the
best-matching reading and writes schema and data precision, recall and F1 per run into
`adaptive_evaluation.json`. The implementation is in [src/evaluation/](src/evaluation/); the unit tests
under [tests/](tests/) cover the column matching and the retrieval metrics.

## Experiment traces and results (Section 6)

[results/README.md](results/README.md) describes every run directory, the per-backbone snapshot in
`results/paper/` and the derived files in `results/analysis/`. The evaluated records of all runs are in
the repository. The raw traces (2,398 files, 630 MB unpacked, 17 MB as zip) are distributed as a separate archive
(`vkgqa_experiment_traces.zip`, see the data availability statement of the paper); unpack it in the
repository root and every trace lands under `results/experiments/<run>/traces/`.

### Reproducing the tables and figures

With the traces in place:

```bash
bash scripts/analysis/rebuild_paper_numbers.sh
```

| Output | Script |
|---|---|
| Table 2 (triple F1, path coherence, searches, failure rate) and the F1 values quoted in Section 6 | `scripts/analysis/table_conditions.py` |
| Table 2, Retry column | `scripts/analysis/table_retry.py` |
| Outcome of hand-back runs (Section 6.1) | `scripts/analysis/handback_outcome.py` |
| Table 3 (UNDER recognition, false alarms) | `scripts/analysis/table_under.py` |
| Figure 3, Figure 4 | `scripts/analysis/fig_conditions.py`, `scripts/analysis/fig_lexical_vs_semantic.py` |
| Figure 5 and the SYN subtypes of Section 6.2 | `scripts/analysis/syn_group_scores.py`, `scripts/analysis/syn_target_ranks.py`, `scripts/analysis/fig_syn_subtypes.py` |
| Search-call statistics of Section 6.2 (calls, raised k) | `scripts/analysis/syn_search_stats.py` |
| Misspelled tokens reaching the linking tools (Section 6.1) | `scripts/analysis/typo_tokens.py` |
| Tool execution time and run time per configuration (Section 6.2) | `scripts/analysis/tool_time.py` |
| Latency, tool-call and token statistics per backbone | `scripts/analysis/runtime_stats.py` |

All scripts read `results/paper/<backbone>/adaptive_evaluation.json` through
[scripts/analysis/canonical.py](scripts/analysis/canonical.py) and resolve the raw trace of every record
from `results/experiments/<run>/traces/`. `syn_target_ranks.py` needs the embedding index under
`data/cache/embeddings/_global/` (included) and an `OPENAI_API_KEY` only for phrases missing from the
embedding cache `results/analysis/syn_term_embeddings.json` (all phrases of the corpus are cached).

## Text2SQL reference point (Appendix C)

CHESS is vendored under [tools/chess/](tools/chess/) with its Apache 2.0 licence and small compatibility
fixes (UTF-8 template loading, Windows file locking). The comparison runs CHESS over SQLite snapshots of
the same datasets in two regimes:

```bash
python scripts/export_mysql_to_sqlite.py                 # SQLite snapshots of the relational sources
python scripts/create_chess_metadata.py                  # BIRD-style dev.json from the corpus
python scripts/create_chess_db_descriptions.py           # column descriptions CHESS indexes
python scripts/create_chess_heterogeneous_db.py          # heterogeneous regime: every table split into two differently named tables
python scripts/create_chess_full_consolidated_db.py      # consolidated regime: one canonical table per concept
python scripts/run_chess_experiment.py --models gpt54    # runs CHESS and converts its output into traces
python scripts/evaluate_chess_syn.py                     # heterogeneous regime -> results/experiments/chess_heterogeneous
python scripts/evaluate_chess_syn_homogeneous_full.py    # consolidated regime  -> results/experiments/chess_consolidated
python scripts/evaluate_het_sql_on_hom_data.py           # CHESS's heterogeneous SQL re-executed on the consolidated snapshot
```

## Licence

The code in this repository is released under the MIT licence (see [LICENSE](LICENSE)).
[tools/chess/](tools/chess/) is from <https://github.com/ShayanTalaei/CHESS> and remains under the Apache
2.0 licence (see [tools/chess/LICENSE](tools/chess/LICENSE)). The mappings and ontologies derive from the
public benchmarks named above and remain under their respective licences.
