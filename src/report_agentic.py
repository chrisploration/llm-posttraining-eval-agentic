from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


_TASK_DIRS: dict[str, str] = {
    "rag_qa": "results/rag",
    "agent_tool_capability": "results/agentic",
    "agent_framework_tool_use": "results/langgraph_agent",
    "agent_supervisor_orchestration": "results/supervisor_agent",
    "agent_memory_followup": "results/agent_memory"
}

_SPECIALIST_PREFIX_RE = re.compile(r"^\s*(calc_agent|weather_agent|coder_agent)\s*:")


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_meta(task_dir: str) -> dict[str, Any]:
    return _load_json(os.path.join(task_dir, "meta.json")) or {}


def _format_header(any_task_dir: str) -> str:
    meta = _run_meta(any_task_dir)
    gpu = meta.get("gpu", {})
    lines = [
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"**Hardware:** {gpu.get('name', 'unknown GPU')}, {meta.get('platform', 'unknown platform')}",
        f"**Python:** {meta.get('python', 'unknown')}  **Torch:** {meta.get('torch', 'unknown')}  **CUDA:** {meta.get('cuda', 'unknown')}",
        f"**Base model:** {meta.get('model_id', 'unknown')}",
        f"**Git SHA:** {meta.get('git_sha', 'unknown')}"
    ]
    return "\n".join(lines)


def _metrics_table() -> str:
    rows = ["| Task | Accuracy | n | Other metrics |", "|---|---|---|---|"]
    for task, task_dir in _TASK_DIRS.items():
        metrics = (_load_json(os.path.join(task_dir, "metrics.json")) or {}).get(task, {})
        acc = metrics.get("accuracy", {})
        acc_str = f"{acc.get('mean'):.1%}" if acc.get("mean") is not None else "n/a"
        n = acc.get("n", "n/a")
        extras = [f"{k}: {v.get('mean'):.1%}" if isinstance(v, dict) and "mean" in v else f"{k}: {v}"
                  for k, v in metrics.items() if k != "accuracy"]
        rows.append(f"| {task} | {acc_str} | {n} | {'; '.join(extras) if extras else '-'} |")
    return "\n".join(rows)


def _delegation_rates(task_dir: str) -> dict[str, dict[str, int]]:
    """Best-effort delegation-rate proxy, computed only from what's persisted
    to disk (the samples preview + all recorded failures for this run), not
    every raw item — run_eval.py doesn't save full per-item output by design."""
    rows = _load_jsonl(os.path.join(task_dir, "samples.jsonl")) + _load_jsonl(os.path.join(task_dir, "failures.jsonl"))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "delegated": 0})
    for row in rows:
        kind = row.get("kind")
        if kind is None:
            continue
        output = str(row.get("output") or row.get("prediction") or "")
        counts[kind]["total"] += 1
        if _SPECIALIST_PREFIX_RE.match(output.strip()):
            counts[kind]["delegated"] += 1
    return dict(counts)


def _delegation_table() -> str:
    rates = _delegation_rates(_TASK_DIRS["agent_supervisor_orchestration"])
    if not rates:
        return "_No supervisor-task samples/failures found to analyze._"
    rows = ["| Specialist (by item kind) | Delegated / Sampled | Rate |", "|---|---|---|"]
    for kind, c in sorted(rates.items()):
        rate = c["delegated"] / c["total"] if c["total"] else 0.0
        rows.append(f"| {kind} | {c['delegated']} / {c['total']} | {rate:.0%} |")
    rows.append("")
    rows.append("_Delegation is measured by whether the output is explicitly prefixed with a "
                "specialist name (e.g. `calc_agent:`), which only appears when a handoff genuinely "
                "occurred. Covers only the samples preview + failures persisted to disk for this "
                "run, not literally every item evaluated._")
    return "\n".join(rows)


def _shared_failures_note() -> str:
    hand_rolled = {r["id"] for r in _load_jsonl(os.path.join(_TASK_DIRS["agent_tool_capability"], "failures.jsonl"))}
    langgraph = {r["id"] for r in _load_jsonl(os.path.join(_TASK_DIRS["agent_framework_tool_use"], "failures.jsonl"))}
    shared = hand_rolled & langgraph
    if not shared:
        return "_No overlapping failed items between the hand-rolled and LangGraph agents in this run._"
    return (f"Items failed identically by both the hand-rolled and LangGraph agent: "
            f"{', '.join(sorted(shared))}. When neither agent invokes its tool, both fall back "
            f"to the same underlying model behavior, including its mistakes.")


_KNOWN_LIMITATIONS = """\
## Known limitations (project design, not derived from this run)

- Sandboxed code execution uses OS-level resource limits (CPU/memory/process
  caps), not container isolation — Docker-in-Docker isn't reliably available
  on RunPod, so this is a tradeoff, not an oversight.
- Langfuse tracing and the LiteLLM judge are wired into the pipeline but only
  activate if their respective external dependencies (a configured Langfuse
  account, a running Ollama instance) are present.
- All new-axis eval counts are 50 (or 3, for memory), appropriate for a
  portfolio-scale demonstration, not a statistically robust benchmark.
"""


def build_report() -> str:
    header = _format_header(_TASK_DIRS["rag_qa"])
    parts = [
        "# Results — Agentic Extension (auto-generated)",
        "",
        header,
        "",
        "## Metrics — new axes",
        "",
        _metrics_table(),
        "",
        "## Multi-agent supervisor — delegation rate by specialist",
        "",
        _delegation_table(),
        "",
        "## Hand-rolled vs. LangGraph agent",
        "",
        _shared_failures_note(),
        "",
        _KNOWN_LIMITATIONS
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a markdown report from agentic-extension eval results.")
    ap.add_argument("--output", default="results/report_agentic.md", help="Path to write the report to.")
    args = ap.parse_args()

    report = build_report()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()