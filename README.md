# Agentic Text-to-SPARQL Experiment (ISWC 2026)

This repository contains the source code, configuration, evaluation
queries and result artefacts that accompany the ISWC 2026 paper on
agentic Text-to-SPARQL generation in dataspace settings.

The codebase implements three pipelines that are compared on the same
natural-language query corpus:

1. **Agentic-Grep** — a tool-using agent that retrieves schema fragments
   via lexical/grep search over OBDA mappings.
2. **Agentic-Semantic** — a tool-using agent that retrieves schema
   fragments via dense embedding search.
3. **Text2SQL (CHESS)** — Stanford's CHESS pipeline (vendored under
   [tools/chess/](tools/chess/)) used as a Text-to-SQL reference
   baseline.

The five evaluation datasets are EDU (LUBM), TRN (GTFS), NRG (Norwegian
Petroleum Directorate), BSBM (Berlin SPARQL Benchmark) and LCA, plus a
heterogeneous combined configuration.

## Repository layout

```
.
|-- src/                       # Source for the agentic SPARQL pipeline
|   |-- agents/                # Orchestrator + retrieval/generation agents
|   |-- baseline/              # Shared LLM/metric helpers used by the runner
|   |-- concurrency/           # Slot manager for parallel container access
|   |-- endpoints/             # OnTop SPARQL endpoint client
|   |-- evaluation/            # Tiered ESSENTIAL/PREFERRED metrics + runner
|   |-- models/                # Pydantic data models
|   |-- tools/                 # SPARQL/mapping tools called by agents
|   |-- tracing/               # Trace collection and rendering
|   |-- utils/                 # Query validators and OnTop hints
|   `-- validation/            # Schema graph + semantic validator
|-- scripts/                   # Experiment runner, dataset converters, dashboards
|-- mappings/                  # OnTop OBDA mappings (Turtle) per dataset
|-- data/queries/              # NL query corpus and tiered ground truth
|-- data/datasets/<DS>/        # Per-class semantic models read by the retrieval
|   `-- class_semantic_models/ #   agents (one .ttl per class, see below)
|-- config/mysql/              # MySQL configuration used by the OnTop containers
|-- docker-compose.yml         # Base services (MySQL + one OnTop per dataset)
|-- docker-compose.slots.yml   # Slot overlay (parallel OnTop containers)
|-- Dockerfile.ontop           # Image for the OnTop endpoints
|-- tools/chess/               # Vendored CHESS Text-to-SQL pipeline (Apache 2.0)
`-- pyproject.toml
```

## Setup

Requires Python 3.11+ and Docker (for the OnTop SPARQL endpoints).

```bash
pip install -e .
cp .env.example .env   # then fill in API keys for the providers you use
```

The `.env.example` file lists the supported LLM providers. You only need
keys for the providers you intend to evaluate.

### SPARQL endpoints

The agentic pipeline queries OnTop endpoints that expose relational
datasets as RDF via OBDA mappings. The deployment used in the paper
launches one OnTop container per dataset and replicates each container
across `N` *slots* so multiple agent runs can hit the database in
parallel without interfering with each other.

```bash
# 1. Initialise per-slot mapping copies (defaults to 5 slots)
python scripts/initialize_slot_mappings.py --slots 5 --clean

# 2. Bring up MySQL + the OnTop slot containers
docker compose -f docker-compose.yml -f docker-compose.slots.yml up -d

# 3. (Optional) Pre-build the embedding index used by the semantic agent
python scripts/preindex_embeddings.py
```

Endpoint URLs default to `http://localhost:8080..8084/sparql`, one per
dataset (EDU, TRN, NRG, BSBM, LCA); the slot variants listen on
neighbouring ports defined in [docker-compose.slots.yml](docker-compose.slots.yml).

### Datasets

The evaluation uses publicly available datasets:

| Code | Source |
|------|--------|
| EDU  | LUBM (Lehigh University Benchmark) |
| TRN  | GTFS public transport feeds |
| NRG  | Norwegian Petroleum Directorate "FactPages" / energy domain |
| BSBM | Berlin SPARQL Benchmark |
| LCA  | Life-cycle assessment data |

Each dataset must be downloaded from its upstream distribution channel.
The conversion scripts under [scripts/convert_*.py](scripts/) and
[scripts/csv_to_sql.py](scripts/csv_to_sql.py) turn the raw
distributions into the relational form expected by the OnTop mappings
in [mappings/](mappings/).

### Class semantic models

The retrieval agents do not search the OBDA mapping directly. Instead
they search a per-class **semantic model**: one short Turtle file per
RDF class that lists all datatype and object properties attached to
that class together with their range. The files live under
`data/datasets/<DATASET>/class_semantic_models/<ClassName>.ttl`. A
minimal example:

```turtle
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix eduo: <http://example.org/ontology/education#> .

eduo:DoctoralCandidate eduo:emailAddress xsd:string ;
    eduo:label xsd:string ;
    eduo:belongsTo eduo:Division ;
    eduo:enrolledIn eduo:AdvancedModule ;
    eduo:mentor eduo:JuniorScholar .
```

Pre-generated semantic models for the five evaluation datasets ship
with this repository so the agents can be run out of the box. They
were generated from the OBDA mappings using
[scripts/rml_schema_extractor.py](scripts/rml_schema_extractor.py).

#### Regenerating the semantic models

The extractor performs a "minimal materialisation": it parses the
R2RML mapping, builds a small SQLite stub populated with placeholder
rows, runs Morph-KGC to materialise the triples, then writes one
Turtle file per RDF class found in the output.

```bash
python scripts/rml_schema_extractor.py \
    mappings/edu-small/mapping.ttl \
    mappings/edu-small/ontology.owl \
    <path-to-EDU-SQL-dump-or-CSV-folder> \
    data/datasets/EDU/class_semantic_models
```

Run analogously for `trn-small`, `nrg-small`, `bsbm` and `lca`. The
data-source argument accepts either a `.sql` dump (gzip-compressed
dumps are also supported) or a folder of CSV files, depending on how
the upstream distribution is published.

## Running the experiments

### Agentic pipelines

```bash
python scripts/run_full_experiment.py \
    --approach agentic_grep \
    --llm-model deepseek-chat \
    --query-set BASE
```

See `python scripts/run_full_experiment.py --help` for the full set of
flags (model, approach, query subset, output directory). Trace files
land in `results/experiments/<run-id>/`. Re-evaluation against the
tiered ground truth is done with
[scripts/reevaluate_tiered.py](scripts/reevaluate_tiered.py); a small
HTML dashboard is provided by
[scripts/tiered_dashboard.py](scripts/tiered_dashboard.py).

### CHESS (Text2SQL)

CHESS is vendored under [tools/chess/](tools/chess/) together with its
original Apache 2.0 licence. From the project root:

```bash
cd tools/chess
PYTHONUTF8=1 python -u src/main.py \
    --data_mode dev \
    --data_path data/dev/dev.json \
    --config run/configs/CHESS_deepseek.yaml \
    --num_workers 1 \
    --pick_final_sql True
```

Sample CHESS run outputs (with local paths stripped) are kept under
[tools/chess/results/](tools/chess/results/) for reference.

## Evaluation

The evaluation uses a tiered ESSENTIAL/PREFERRED ground-truth design
together with schema, result and path-coherence metrics. Evaluation
entry points live in [scripts/reevaluate_*.py](scripts/) and the
implementation is in [src/evaluation/](src/evaluation/).

## Citation

Citation information will be added once the paper is published.

## Licence

The code in this repository is released under the MIT licence (see
[LICENSE](LICENSE)). Vendored third-party components retain their
original licences:

* [tools/chess/](tools/chess/) is from
  <https://github.com/ShayanTalaei/CHESS> and remains under the Apache
  2.0 licence (see [tools/chess/LICENSE](tools/chess/LICENSE)).
