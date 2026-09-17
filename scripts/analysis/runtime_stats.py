"""Run-time and tool statistics of the canonical runs (Section 6.2: tool execution time per run).

Reads the raw traces of the four condition runs of every backbone and prints latency, tool-call
and token tables. Run from the repository root:  python scripts/analysis/runtime_stats.py
Writes results/analysis/runtime_stats.json.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "results" / "experiments"

RUNS = {
    label: [EXPERIMENTS / f"{folder}_{c}" / "traces" for c in ("base", "syn", "typo", "under")]
    for folder, label in (("deepseek", "DeepSeek-V3.2"), ("gpt5", "GPT-5.4"), ("qwen", "Qwen3.5-27B"))
}


def load_traces(dir_path: Path) -> list[dict[str, Any]]:
    traces = []
    for fn in sorted(f for f in os.listdir(dir_path) if f.endswith(".json")):
        with open(dir_path / fn, encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summary_stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "n": 0}
    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p95": pct(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
    }


def trace_metrics(t: dict[str, Any]) -> dict[str, Any]:
    timing = t.get("timing") or {}
    tokens = t.get("token_usage") or {}
    agent_details = t.get("agent_timing_details") or []

    retrieval_agent_time = 0.0
    retrieval_agent_tools = 0
    gen_agent_time = 0.0
    gen_agent_tools = 0
    for a in agent_details:
        name = a.get("agent_name", "")
        if "retrieval" in name:
            retrieval_agent_time = a.get("total_duration_ms", 0)
            retrieval_agent_tools = a.get("total_tool_calls", 0)
        elif "sparql" in name or "generation" in name:
            gen_agent_time = a.get("total_duration_ms", 0)
            gen_agent_tools = a.get("total_tool_calls", 0)

    return {
        "approach": t.get("approach", ""),
        "query_set": (t.get("query_metadata") or {}).get("query_set", ""),
        "success": t.get("success", False),
        "total_time_ms": timing.get("total_time_ms", 0),
        "retrieval_time_ms": timing.get("retrieval_time_ms", 0),
        "generation_time_ms": timing.get("generation_time_ms", 0),
        "llm_inference_time_ms": timing.get("llm_inference_time_ms", 0),
        "tool_execution_time_ms": timing.get("tool_execution_time_ms", 0),
        "sparql_execution_time_ms": timing.get("sparql_execution_time_ms", 0),
        "total_tool_calls": t.get("total_tool_calls", 0),
        "unique_tools_used": t.get("unique_tools_used", 0),
        "total_llm_calls": t.get("total_llm_calls", 0),
        "retrieval_iterations": t.get("retrieval_iterations", 0),
        "generation_iterations": t.get("generation_iterations", 0),
        "total_sparql_executions": t.get("total_sparql_executions", 0),
        "prompt_tokens": tokens.get("prompt_tokens", 0),
        "completion_tokens": tokens.get("completion_tokens", 0),
        "total_tokens": tokens.get("total_tokens", 0),
        "retrieval_tokens": tokens.get("retrieval_tokens", 0),
        "generation_tokens": tokens.get("generation_tokens", 0),
        "retrieval_agent_time": retrieval_agent_time,
        "retrieval_agent_tools": retrieval_agent_tools,
        "gen_agent_time": gen_agent_time,
        "gen_agent_tools": gen_agent_tools,
        "tool_invocations": t.get("tool_invocations", []),
    }


def sparql_error_rate(t: dict[str, Any]) -> tuple[int, int]:
    """Return (errors, total_attempts) for SPARQL generation attempts."""
    errs, total = 0, 0
    for gp in t.get("generation_phases", []) or []:
        total += gp.get("queries_generated", 0)
        errs += gp.get("queries_with_syntax_errors", 0)
    return errs, total


def tool_success_rate(t: dict[str, Any]) -> tuple[int, int]:
    ok, total = 0, 0
    for inv in t.get("tool_invocations", []) or []:
        total += 1
        if inv.get("success"):
            ok += 1
    return ok, total


def aggregate(traces: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [trace_metrics(t) for t in traces]

    by_approach = defaultdict(list)
    for m in metrics:
        by_approach[m["approach"]].append(m)

    out: dict[str, Any] = {"overall": {}, "by_approach": {}, "by_set_approach": {}}

    def summarize(ms: list[dict[str, Any]]) -> dict[str, Any]:
        if not ms:
            return {}
        return {
            "n_traces": len(ms),
            "success_rate": sum(1 for m in ms if m["success"]) / len(ms),
            "total_time_s": summary_stats([m["total_time_ms"] / 1000 for m in ms]),
            "retrieval_time_s": summary_stats([m["retrieval_time_ms"] / 1000 for m in ms]),
            "generation_time_s": summary_stats([m["generation_time_ms"] / 1000 for m in ms]),
            "llm_inference_time_s": summary_stats([m["llm_inference_time_ms"] / 1000 for m in ms]),
            "tool_execution_time_ms": summary_stats([m["tool_execution_time_ms"] for m in ms]),
            "tool_calls": summary_stats([m["total_tool_calls"] for m in ms]),
            "unique_tools": summary_stats([m["unique_tools_used"] for m in ms]),
            "llm_calls": summary_stats([m["total_llm_calls"] for m in ms]),
            "retrieval_iterations": summary_stats([m["retrieval_iterations"] for m in ms]),
            "generation_iterations": summary_stats([m["generation_iterations"] for m in ms]),
            "sparql_executions": summary_stats([m["total_sparql_executions"] for m in ms]),
            "retrieval_agent_time_s": summary_stats([m["retrieval_agent_time"] / 1000 for m in ms]),
            "gen_agent_time_s": summary_stats([m["gen_agent_time"] / 1000 for m in ms]),
            "retrieval_agent_tools": summary_stats([m["retrieval_agent_tools"] for m in ms]),
            "gen_agent_tools": summary_stats([m["gen_agent_tools"] for m in ms]),
            "prompt_tokens": summary_stats([m["prompt_tokens"] for m in ms]),
            "completion_tokens": summary_stats([m["completion_tokens"] for m in ms]),
            "total_tokens": summary_stats([m["total_tokens"] for m in ms]),
            "retrieval_tokens": summary_stats([m["retrieval_tokens"] for m in ms]),
            "generation_tokens": summary_stats([m["generation_tokens"] for m in ms]),
        }

    out["overall"] = summarize(metrics)
    for appr, ms in by_approach.items():
        out["by_approach"][appr] = summarize(ms)

    by_set_appr = defaultdict(list)
    for m in metrics:
        by_set_appr[(m["query_set"], m["approach"])].append(m)
    for (qs, appr), ms in by_set_appr.items():
        out["by_set_approach"][f"{qs}|{appr}"] = summarize(ms)

    # SPARQL syntax errors + tool success (from raw traces)
    se_by_appr = defaultdict(lambda: [0, 0])
    ts_by_appr = defaultdict(lambda: [0, 0])
    for t in traces:
        appr = t.get("approach", "")
        errs, total = sparql_error_rate(t)
        se_by_appr[appr][0] += errs
        se_by_appr[appr][1] += total
        se_by_appr["__overall__"][0] += errs
        se_by_appr["__overall__"][1] += total
        ok, tot = tool_success_rate(t)
        ts_by_appr[appr][0] += ok
        ts_by_appr[appr][1] += tot
        ts_by_appr["__overall__"][0] += ok
        ts_by_appr["__overall__"][1] += tot
    out["sparql_syntax_error_rate"] = {
        k: (v[0] / v[1] if v[1] else 0.0, v[1]) for k, v in se_by_appr.items()
    }
    out["tool_success_rate"] = {
        k: (v[0] / v[1] if v[1] else 0.0, v[1]) for k, v in ts_by_appr.items()
    }

    # Tool frequency
    tool_counter: Counter[str] = Counter()
    tool_duration: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        for inv in t.get("tool_invocations", []) or []:
            name = inv.get("tool_name", "")
            tool_counter[name] += 1
            tool_duration[name].append(inv.get("duration_ms", 0))
    out["top_tools"] = tool_counter.most_common(20)
    out["tool_avg_latency_ms"] = {
        name: statistics.mean(durs) for name, durs in tool_duration.items() if durs
    }

    return out


def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def main() -> None:
    results: dict[str, Any] = {}
    for model, trace_dir in RUNS.items():
        print(f"Loading {model} from {trace_dir} ...")
        traces = [t for d in trace_dir for t in load_traces(d)]
        results[model] = aggregate(traces)
        print(f"  {len(traces)} traces processed")

    out_path = ROOT / "results" / "analysis" / "runtime_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # ---- Print comparison tables (markdown) ----
    def row(model: str, stats: dict[str, Any]) -> list[str]:
        t = stats["total_time_s"]
        tc = stats["tool_calls"]
        lc = stats["llm_calls"]
        ri = stats["retrieval_iterations"]
        return [
            model,
            str(stats["n_traces"]),
            fmt(stats["success_rate"] * 100, 1) + "%",
            fmt(t["mean"], 1),
            fmt(t["median"], 1),
            fmt(t["p95"], 1),
            fmt(tc["mean"], 2),
            fmt(lc["mean"], 2),
            fmt(ri["mean"], 2),
        ]

    print("\n### Latency / Efficiency (Overall)")
    print("| Model | Traces | Success | Mean Time (s) | Median (s) | p95 (s) | Tool Calls | LLM Calls | Ret Iter |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in results:
        print("| " + " | ".join(row(model, results[model]["overall"])) + " |")

    print("\n### Latency by Approach")
    print("| Model | Approach | Mean Time (s) | Median (s) | p95 (s) | Tool Calls | LLM Calls |")
    print("|---|---|---|---|---|---|---|")
    for model in results:
        for appr in ("agentic_grep", "agentic_semantic"):
            s = results[model]["by_approach"].get(appr, {})
            if not s:
                continue
            t = s["total_time_s"]
            print(
                f"| {model} | {appr} | {fmt(t['mean'], 1)} | {fmt(t['median'], 1)} | "
                f"{fmt(t['p95'], 1)} | {fmt(s['tool_calls']['mean'], 2)} | "
                f"{fmt(s['llm_calls']['mean'], 2)} |"
            )

    print("\n### Per-Agent Latency (seconds, mean)")
    print("| Model | Approach | Retrieval Agent | Gen Agent | Ret Tools | Gen Tools |")
    print("|---|---|---|---|---|---|")
    for model in results:
        for appr in ("agentic_grep", "agentic_semantic"):
            s = results[model]["by_approach"].get(appr, {})
            if not s:
                continue
            print(
                f"| {model} | {appr} | "
                f"{fmt(s['retrieval_agent_time_s']['mean'], 1)} | "
                f"{fmt(s['gen_agent_time_s']['mean'], 1)} | "
                f"{fmt(s['retrieval_agent_tools']['mean'], 2)} | "
                f"{fmt(s['gen_agent_tools']['mean'], 2)} |"
            )

    print("\n### Token Usage (where the provider reports it)")
    print("| Model | Approach | Prompt Tok (mean) | Completion Tok (mean) | Total Tok (mean) | Ret Tok | Gen Tok |")
    print("|---|---|---|---|---|---|---|")
    for model in results:
        for appr in ("agentic_grep", "agentic_semantic"):
            s = results[model]["by_approach"].get(appr, {})
            if not s:
                continue
            print(
                f"| {model} | {appr} | "
                f"{fmt(s['prompt_tokens']['mean'], 0)} | "
                f"{fmt(s['completion_tokens']['mean'], 0)} | "
                f"{fmt(s['total_tokens']['mean'], 0)} | "
                f"{fmt(s['retrieval_tokens']['mean'], 0)} | "
                f"{fmt(s['generation_tokens']['mean'], 0)} |"
            )

    print("\n### SPARQL Syntax Error Rate & Tool Success")
    for model in results:
        r = results[model]
        print(f"\n**{model}**")
        for k, (rate, tot) in r["sparql_syntax_error_rate"].items():
            if k == "__overall__":
                print(f"  SPARQL syntax-error rate (overall): {fmt(rate * 100, 1)}% of {tot} queries generated")
            else:
                print(f"  SPARQL syntax-error rate [{k}]: {fmt(rate * 100, 1)}% of {tot}")
        for k, (rate, tot) in r["tool_success_rate"].items():
            if k == "__overall__":
                print(f"  Tool success rate (overall): {fmt(rate * 100, 2)}% of {tot} invocations")
            else:
                print(f"  Tool success rate [{k}]: {fmt(rate * 100, 2)}% of {tot}")

    print("\n### Top tools used (invocations)")
    for model in results:
        print(f"\n**{model}**")
        for name, n in results[model]["top_tools"][:8]:
            lat = results[model]["tool_avg_latency_ms"].get(name, 0)
            print(f"  {name}: {n} calls, avg {fmt(lat, 1)}ms")

    print("\n### By Set × Approach — total time mean (s)")
    print("| Model | Set | Approach | Mean (s) | Tool Calls | LLM Calls |")
    print("|---|---|---|---|---|---|")
    for model in results:
        for key in sorted(results[model]["by_set_approach"].keys()):
            s = results[model]["by_set_approach"][key]
            qs, appr = key.split("|")
            print(
                f"| {model} | {qs} | {appr} | "
                f"{fmt(s['total_time_s']['mean'], 1)} | "
                f"{fmt(s['tool_calls']['mean'], 2)} | "
                f"{fmt(s['llm_calls']['mean'], 2)} |"
            )


if __name__ == "__main__":
    main()
