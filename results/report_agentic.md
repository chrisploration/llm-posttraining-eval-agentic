# Results — Agentic Extension (auto-generated)

**Date:** 2026-08-25
**Hardware:** NVIDIA GeForce RTX 4090, Linux-6.8.0-63-generic-x86_64-with-glibc2.35
**Python:** 3.11.10  **Torch:** 2.4.1+cu124  **CUDA:** 12.4
**Base model:** mistralai/Mistral-7B-Instruct-v0.3
**Git SHA:** b3abb8f76a895c8820d01f24bee09a2425d63a0f

## Metrics — new axes

| Task | Accuracy | n | Other metrics |
|---|---|---|---|
| rag_qa | 100.0% | 50 | accuracy_without_context: 96.0%; groundedness_delta: 0.040000000000000036 |
| agent_tool_capability | 96.0% | 50 | tool_call_rate: 4.0% |
| agent_framework_tool_use | 96.0% | 50 | - |
| agent_supervisor_orchestration | 60.0% | 50 | - |
| agent_memory_followup | 100.0% | 3 | - |

## Multi-agent supervisor — delegation rate by specialist

| Specialist (by item kind) | Delegated / Sampled | Rate |
|---|---|---|
| calculator | 2 / 2 | 100% |
| code | 0 / 2 | 0% |
| weather | 0 / 1 | 0% |

_Delegation is measured by whether the output is explicitly prefixed with a specialist name (e.g. `calc_agent:`), which only appears when a handoff genuinely occurred. Covers only the samples preview + failures persisted to disk for this run, not literally every item evaluated._

## Hand-rolled vs. LangGraph agent

Items failed identically by both the hand-rolled and LangGraph agent: cap_add_27_23. When neither agent invokes its tool, both fall back to the same underlying model behavior, including its mistakes.

## Known limitations (project design, not derived from this run)

- Sandboxed code execution uses OS-level resource limits (CPU/memory/process
  caps), not container isolation — Docker-in-Docker isn't reliably available
  on RunPod, so this is a tradeoff, not an oversight.
- Langfuse tracing and the LiteLLM judge are wired into the pipeline but only
  activate if their respective external dependencies (a configured Langfuse
  account, a running Ollama instance) are present.
- All new-axis eval counts are 50 (or 3, for memory), appropriate for a
  portfolio-scale demonstration, not a statistically robust benchmark.