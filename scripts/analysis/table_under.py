"""Recognition on the UNDER condition (Table 3).

Default configuration:     results/experiments/<model>_under            (5 questions x 6 runs, both retrieval configurations)
Instructed configuration:  results/experiments/<model>_under_instructed (same questions and runs, prompt rule added)
False-alarm control:       results/experiments/<model>_base_instructed  (17 BASE questions x 3 runs, semantic retrieval)

Recognition = the run stopped with stop_kind "underspecified" (no final query). Missing-schema stops are
counted separately. A false alarm is a BASE control run that stopped as underspecified.

Run from the repository root:  python scripts/analysis/table_under.py
Writes results/analysis/under_recognition.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import MODELS, ROOT, load_run  # noqa: E402

UNDER = ["UNDER01", "UNDER03", "UNDER04", "UNDER06", "UNDER08"]
OUT = ROOT / "results" / "analysis" / "under_recognition.json"


def rows(run: str) -> list[dict]:
    out = []
    for t in load_run(run):
        stop = t.get("stop_kind", "")
        if t.get("is_unanswerable") and not stop:
            stop = "missing_schema"  # traces without a stop kind ended because the agent found no schema
        out.append(dict(q=t["query_metadata"]["query_id"], appr=t["approach"].replace("agentic_", ""),
                        run=t.get("run_number"), unanswerable=bool(t.get("is_unanswerable")), stop=stop,
                        reason=(t.get("unanswerable_reason") or "")[:160], empty=not t.get("final_sparql"),
                        f1=(t.get("result_metrics") or {}).get("f1_score")))
    return out


res = {}
print(f"{'backbone':<14}{'retrieval':<10}{'default rec.':>13}{'instr. rec.':>12}{'instr. miss-schema':>19}{'BASE false alarms':>18}")
for folder, label in MODELS:
    inst = rows(f"{folder}_under_instructed")
    base = rows(f"{folder}_base_instructed")
    dflt = [r for r in rows(f"{folder}_under") if r["q"] in UNDER]
    for appr in ("grep", "semantic"):
        d = [r for r in dflt if r["appr"] == appr]
        i = [r for r in inst if r["appr"] == appr and r["q"] in UNDER]
        b = [r for r in base if r["appr"] == appr]
        d_rec = sum(r["stop"] == "underspecified" for r in d)
        d_ms = sum(r["stop"] == "missing_schema" for r in d)
        i_rec = sum(r["stop"] == "underspecified" for r in i)
        i_ms = sum(r["stop"] == "missing_schema" for r in i)
        fa = sum(r["stop"] == "underspecified" for r in b)
        answered = [r["f1"] or 0 for r in b if not r["unanswerable"]]
        res[f"{label}|{'lexical' if appr == 'grep' else appr}"] = dict(
            default_recognized=d_rec, default_n=len(d), default_missing_schema=d_ms,
            instructed_recognized=i_rec, instructed_n=len(i), instructed_missing_schema=i_ms,
            base_false_alarms=fa, base_n=len(b),
            base_f1_answered=statistics.mean(answered) if answered else None,
            per_question={q: dict(recognized=sum(r["stop"] == "underspecified" for r in i if r["q"] == q),
                                  n=sum(r["q"] == q for r in i),
                                  reasons=[r["reason"] for r in i if r["q"] == q and r["stop"] == "underspecified"][:3])
                          for q in UNDER})
        print(f"{label:<14}{('lexical' if appr == 'grep' else appr):<10}{f'{d_rec}/{len(d)} (ms {d_ms})':>13}"
              f"{f'{i_rec}/{len(i)}':>12}{i_ms:>19}{f'{fa}/{len(b)}':>18}")

print("\nPer question (instructed, recognized/n):")
print(f"{'backbone':<14}{'retrieval':<10}" + "".join(f"{q:>10}" for q in UNDER))
for key, r_ in res.items():
    label, appr = key.split("|")
    print(f"{label:<14}{appr:<10}" + "".join(f"{r_['per_question'][q]['recognized']}/{r_['per_question'][q]['n']}".rjust(10) for q in UNDER))

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print("\nwrote", OUT)
