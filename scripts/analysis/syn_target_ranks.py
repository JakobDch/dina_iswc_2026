"""Outcome-independent grouping criteria for the SYN questions (Section 6.2, Figure 5).

Two features per SYN question, computed from the corpus, the schema index of the semantic linking
tools and the tool's ranking alone (no run results are used):

  A. target rank under the SYN phrase. The corpus notes list every substitution as
     "<BASE phrase>-><SYN phrase>". Whenever the BASE phrase names a ground-truth schema element
     (it does by construction: BASE echoes the schema), the SYN phrase is embedded with the tool's
     model (text-embedding-3-large) and the element's rank among the same-type elements of the
     dataset is read off the tool's own index. The tool's default top_k is 5; it scans 3*top_k
     candidates overall and keeps top_k of the requested type, so an element is reachable at the
     default when its type-rank is <= 5 and its overall rank <= 15. Per question the worst (largest)
     rank over its substituted REQUIRED elements is reported.
  B. entity mention similarity. For every string literal in the ground truth (TROLL, STATOIL, ...),
     the best character-level similarity (difflib ratio, lower-cased, spaces removed) between the
     literal and any 1-4 word window of the SYN question. 1.0 = the question uses the literal itself.

Outcome columns (from results/analysis/syn_group_scores.json, if present) are printed only for the
cross-check at the end; they play no role in the features.

Run from the repository root:  python scripts/analysis/syn_target_ranks.py
Phrase embeddings are cached in results/analysis/syn_term_embeddings.json; an OPENAI_API_KEY (.env)
is needed only for phrases that are not in the cache. Writes results/analysis/syn_target_ranks.json.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

import faiss  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from data.queries.experimental_corpus import ALL_EXPERIMENTAL_QUERIES as Q  # noqa: E402
from src.evaluation.retrieval_ground_truth import extract_retrieval_ground_truth  # noqa: E402

load_dotenv(ROOT / ".env")
ANALYSIS = ROOT / "results" / "analysis"
OUT = ANALYSIS / "syn_target_ranks.json"
CACHE = ANALYSIS / "syn_term_embeddings.json"
TOPK = 5
EMB_MODEL = "text-embedding-3-large"
_cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}


def embed(terms):
    todo = sorted({t for t in terms if t not in _cache})
    if not todo:
        return
    from openai import OpenAI
    client = OpenAI()
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        for t, d in zip(chunk, client.embeddings.create(model=EMB_MODEL, input=chunk).data):
            _cache[t] = d.embedding
    json.dump(_cache, open(CACHE, "w", encoding="utf-8"))


# ---- the tool's own index ----
el = json.load(open(ROOT / "data/cache/embeddings/_global/schema_elements.json", encoding="utf-8"))
gidx = faiss.read_index(str(ROOT / "data/cache/embeddings/_global/schema_index.faiss"))
GX = gidx.reconstruct_n(0, gidx.ntotal)
INDEX = {}
for ds in ("NRG", "TRN", "BSBM", "LCA", "EDU"):
    ids = [i for i, e in enumerate(el) if e.get("dataset") == ds]
    INDEX[ds] = ([el[i] for i in ids], GX[ids])


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def forms(s):
    """the normalised phrase plus its plausible singular forms"""
    out = {s}
    if s.endswith("ies") and len(s) > 5:
        out.add(s[:-3] + "y")
    if s.endswith("es") and len(s) > 4:
        out.add(s[:-2])
    if s.endswith("s") and len(s) > 3:
        out.add(s[:-1])
    return out


def stem(s):
    for suf in ("ing", "ed", "es", "s"):
        if s.endswith(suf) and len(s) > len(suf) + 3:
            return s[:-len(suf)]
    return s


def match_score(base_phrase, name):
    """how well does the BASE phrase name this schema element? 2 = exact (modulo plural),
    1 = the element name is contained in a longer BASE phrase ('total drill depth greater than'
    -> totalDrillDepth) or the phrase stem is contained in the name ('operated' -> activeDepositOperator);
    ties on 1 are broken by the longer name. 0 = no."""
    nb, nn = norm(base_phrase), norm(name)
    if forms(nb) & forms(nn):
        return 2, len(nn)
    if len(nn) >= 6 and nn in nb:
        return 1, len(nn)
    if len(stem(nb)) >= 5 and stem(nb) in nn:
        return 1, len(nn)
    return 0, 0


def cos(a, b):
    x = np.array(_cache[a]); y = np.array(_cache[b])
    return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))


def shares_token_prefix(literal, mention, k=3):
    lt = [t.lower() for t in re.findall(r"[A-Za-z]+", literal)]
    mt = [t.lower() for t in re.findall(r"[A-Za-z]+", mention)]
    return any(a[:k] == b[:k] and len(a) >= k and len(b) >= k for a in lt for b in mt)


def local(u):
    return u.split("#")[-1].split("/")[-1]


def ranks(phrase, ds, etype, name, domains):
    """(type rank, overall rank) of the element with this local name under the phrase.
    For properties, prefer the entry whose domain is a GT class; otherwise the best-ranked entry."""
    els, X = INDEX[ds]
    q = np.array(_cache[phrase], dtype=np.float32); q /= np.linalg.norm(q)
    s = X @ q
    order = np.argsort(-s)
    typed = 0; fallback = None
    for overall, i in enumerate(order, 1):
        e = els[i]
        if e["type"] != etype:
            continue
        typed += 1
        if e["name"] != name:
            continue
        if etype == "class" or local(e.get("domain_class") or "") in domains:
            return typed, overall
        if fallback is None:
            fallback = (typed, overall)
    return fallback if fallback else (None, None)


def entity_sim(literal, question):
    lit = norm(literal)
    words = re.findall(r"[A-Za-z0-9']+", question)
    best = 0.0; arg = ""
    for n in range(1, 5):
        for i in range(len(words) - n + 1):
            w = " ".join(words[i:i + n])
            r = difflib.SequenceMatcher(None, lit, norm(w)).ratio()
            if r > best:
                best, arg = r, w
    return best, arg


def gt_for(q):
    try:
        return extract_retrieval_ground_truth(q.sparql_queries[0], q.query_id)
    except TypeError:  # SYN19: the extractor cannot sort its InstanceReferences; neutralise the literals
        sp = re.sub(r'"\d\d:\d\d:\d\d"', '"t"', q.sparql_queries[0])
        return extract_retrieval_ground_truth(sp, q.query_id)


prep = {}
phrases = set()
for q in Q:
    if q.query_set != "SYN":
        continue
    gt = gt_for(q)
    elems = {}
    for u, nec in gt.classes.items():
        elems[("class", local(u))] = nec.value
    for u, nec in gt.properties.items():
        elems[("property", local(u))] = nec.value
    domains = {local(u) for u in gt.classes}
    pairs = re.findall(r"([A-Za-z][A-Za-z0-9 '\-]*?)\s*->\s*([A-Za-z0-9][A-Za-z0-9 '\-]*)", q.notes or "")
    matched = []
    for base_p, syn_p in pairs:
        base_p, syn_p = base_p.strip(), syn_p.strip()
        best = max(elems.items(), key=lambda kv: match_score(base_p, kv[0][1]))
        if match_score(base_p, best[0][1])[0] > 0:
            (etype, name), nec = best
            matched.append((base_p, syn_p, etype, name, nec))
            phrases.update([base_p, syn_p])
    inst = [ir for ir in getattr(gt, "instances", []) if ir.value_type == "string" and ir.associated_class]
    prep[q.query_id] = dict(q=q, ds=q.dataset, matched=matched, inst=inst, domains=domains)
embed(phrases)

scores_file = ANALYSIS / "syn_group_scores.json"
OUTC = {r["n"]: r for r in json.load(open(scores_file, encoding="utf-8"))} if scores_file.exists() else {}
rows = []
print(f"top_k default = {TOPK}; type-rank/overall rank of the target under the SYN phrase [BASE phrase in brackets]\n")
for qid, p in prep.items():
    n = qid[3:]; q = p["q"]
    print(f"== {qid} [{p['ds']}] {q.query}")
    worst = 0; worst_el = ""; unreachable = []; detail = []; min_sim = 1.0
    for base_p, syn_p, etype, name, nec in p["matched"]:
        tr, orr = ranks(syn_p, p["ds"], etype, name, p["domains"])
        btr, borr = ranks(base_p, p["ds"], etype, name, p["domains"])
        reach = tr is not None and tr <= TOPK and orr <= 3 * TOPK
        sim = cos(syn_p, base_p)
        print(f"   {syn_p:<28} -> {etype[:4]} {name:<24} {nec[:3]}  sim {sim:.2f}  rank {tr}/{orr}   "
              f"[BASE '{base_p}': {btr}/{borr}]  {'' if reach else 'UNREACHABLE@5'}")
        detail.append(dict(syn=syn_p, base=base_p, etype=etype, name=name, nec=nec, sim=round(sim, 3),
                           rank=tr, overall=orr, base_rank=btr))
        min_sim = min(min_sim, sim)
        # worst rank over REQUIRED elements only; ACCEPTABLE elements are still listed in `detail`
        if tr is not None and tr > worst and nec == "required":
            worst, worst_el = tr, name
        if not reach:
            unreachable.append(name)
    ents = []
    for ir in p["inst"]:
        sim, arg = entity_sim(ir.value, q.query)
        shared = shares_token_prefix(ir.value, arg)
        ents.append(dict(literal=ir.value, mention=arg, sim=round(sim, 2), shared_prefix=shared))
        print(f"   entity '{ir.value}' ~ '{arg}'  sim {sim:.2f}  {'variant of the literal' if shared else 'DIFFERENT NAME'}")
    row = dict(n=n, ds=p["ds"], min_sim=round(min_sim, 3), worst_rank=worst, worst_el=worst_el,
               unreachable=unreachable, n_pairs=len(p["matched"]), detail=detail, entity=ents)
    if n in OUTC:
        row.update({k: OUTC[n][k] for k in ("ss_f1", "bs_f1", "sl_f1", "bl_f1")})
    rows.append(row)
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)

print(f"\n{'q':<4}{'ds':<5}{'pairs':>5}{'minSim':>7}{'worst':>6}  {'worst element':<22}{'entity':<15} | SYNsem BASEsem SYNlex")
for r in sorted(rows, key=lambda r: (-r["worst_rank"], r["n"])):
    ent = "-" if not r["entity"] else ("variant" if all(e["shared_prefix"] for e in r["entity"]) else "DIFFERENT NAME")
    f = lambda k: f"{r[k]:.2f}" if k in r and r[k] is not None else "  - "  # noqa: E731
    print(f"{r['n']:<4}{r['ds']:<5}{r['n_pairs']:>5}{r['min_sim']:>7.2f}{r['worst_rank']:>6}  {r['worst_el']:<22}{ent:<15}"
          f" | {f('ss_f1')}   {f('bs_f1')}   {f('sl_f1')}")
print("wrote", OUT)
