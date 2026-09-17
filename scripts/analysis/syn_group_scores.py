"""Per SYN question: data F1, schema F1 and a decoy-aware retrieval F1 for the four cells
BASE/SYN x lexical/semantic, as the mean over the three backbones of the per-question means.

The retrieval F1 used here is literal-tolerant (string and number literals in the retrieved triples
are neutralised), takes the best score over all admissible ground-truth readings and counts every
retrieved schema triple in the precision, so that decoy triples lower the score.

Input of fig_syn_subtypes.py. Run from the repository root:
  python scripts/analysis/syn_group_scores.py
Writes results/analysis/syn_group_scores.json.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)

from rdflib import Graph, URIRef  # noqa: E402
from rdflib.namespace import RDF, RDFS  # noqa: E402

from canonical import MODELS, load_traces  # noqa: E402
from data.queries.experimental_corpus import ALL_EXPERIMENTAL_QUERIES as Q  # noqa: E402
from src.evaluation.retrieval_ground_truth import build_schema_gt  # noqa: E402
from src.evaluation.retrieval_metrics import _f1, calculate_retrieval_metrics  # noqa: E402
from src.validation.schema_graph import get_combined_schema  # noqa: E402

OUT = ROOT / "results" / "analysis" / "syn_group_scores.json"
BY = {q.query_id: q for q in Q if q.query_set in ("BASE", "SYN")}
_LIT = re.compile(r'"(?:[^"\\]|\\.)*"(?:@[a-z\-]+|\^\^\S+)?|(?<=\s)-?\d+(?:\.\d+)?(?=\s*[;,.])')


def literal_tolerant(blob: str) -> str:
    if "xsd:" not in blob and "XMLSchema" not in blob:
        blob = "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n" + blob
    return _LIT.sub("xsd:string", blob)


def n_schema_triples(blobs: list[str]) -> int:
    g = Graph()
    for b in blobs:
        try:
            g.parse(data=b, format="turtle")
        except Exception:
            pass
    return len({(str(s), str(p), str(o)) for s, p, o in g
                if isinstance(s, URIRef) and isinstance(p, URIRef)
                and str(p) not in (str(RDF.type), str(RDFS.subClassOf))})


_GT: dict[str, list] = {}


def retrieval_f1(blobs: list[str], qid: str) -> float:
    if not blobs:
        return 0.0
    q = BY[qid]
    ds = q.datasets or [q.dataset]
    if qid not in _GT:
        _GT[qid] = [g for g in (build_schema_gt(sp, ds, qid) for sp in q.sparql_queries) if g.triples]
    schema = get_combined_schema(ds)
    n_all = n_schema_triples(blobs)
    best = 0.0
    for g in _GT[qid]:
        r = calculate_retrieval_metrics(blobs, schema, g)
        prec = min(r.schema_triples_matched / n_all, 1.0) if n_all else 0.0
        best = max(best, _f1(prec, r.schema_triple_recall))
    return best


runs: dict[tuple, list[dict]] = {}
for folder, label in MODELS:
    for r in load_traces(folder, with_raw=True):
        qset = r.get("query_set")
        appr = r.get("approach")
        if qset not in ("BASE", "SYN") or appr not in ("agentic_grep", "agentic_semantic"):
            continue
        raw = r.get("_raw") or {}
        blobs = [literal_tolerant(x) for x in (raw.get("all_retrieved_triples") or []) if isinstance(x, str) and x.strip()]
        am = r.get("adaptive_metrics") or {}
        n = r["query_id"][len(qset):]
        key = (label, n, qset, appr[len("agentic_"):])
        runs.setdefault(key, []).append(dict(f1=am.get("best_f1") or 0, sf1=am.get("schema_f1") or 0,
                                             v3=retrieval_f1(blobs, r["query_id"])))
    print(f"{label}: scored", flush=True)

IDS = sorted({k[1] for k in runs if k[2] == "SYN"})


def cell(n: str, cond: str, appr: str, metric: str) -> float | None:
    per = []
    for _folder, label in MODELS:
        v = [x[metric] for x in runs.get((label, n, cond, appr), [])]
        if v:
            per.append(statistics.mean(v))
    return statistics.mean(per) if per else None


rows = []
for n in IDS:
    d = dict(n=n, ds=BY[f"SYN{n}"].dataset)
    for metric in ("f1", "sf1", "v3"):
        for cond, appr, tag in (("BASE", "grep", "bl"), ("BASE", "semantic", "bs"),
                                ("SYN", "grep", "sl"), ("SYN", "semantic", "ss")):
            d[f"{tag}_{metric}"] = cell(n, cond, appr, metric)
    rows.append(d)
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)

fmt = lambda v: f"{v:.2f}" if v is not None else "  nan"  # noqa: E731
print(f"{'q':<4}{'ds':<6}{'data F1 BASEsem SYNsem':>24}{'schema F1':>18}{'retrieval F1':>18}")
for r in rows:
    print(f"{r['n']:<4}{r['ds']:<6}{fmt(r['bs_f1']):>12}{fmt(r['ss_f1']):>8}{fmt(r['bs_sf1']):>10}{fmt(r['ss_sf1']):>8}"
          f"{fmt(r['bs_v3']):>10}{fmt(r['ss_v3']):>8}")
print("wrote", OUT)
