# llm-posttraining-eval-agentic

Eval-driven posttraining and regression analysis for Mistral-7B, extended with a RAG axis, an MCP-based agent stack, and a LangGraph multi-agent supervisor. Implements a complete pipeline: baseline evaluation, QLoRA fine-tuning, posttrained evaluation, automated comparison, retrieval-augmented generation, and five agentic eval axes covering tool-calling, framework orchestration, multi-agent delegation, and conversational memory.

## Overview

This project implements an end to end workflow for posttraining LLMs that is
easy to run, inspect, and extend. It is designed around four engineering goals:

- **Reproducible experiments** — configuration driven, deterministic seeds, explicit commands
- **Modular system design** — separated stages with minimal coupling
- **Clear evaluation workflows** — structured three axis evaluation with regression detection
- **Rapid iteration** — change experiments through config overrides, not code edits

This fork extends the original framework with a second phase: an **agentic extension** covering retrieval-augmented generation and multiple agent architectures (hand-rolled MCP tool-calling, a LangGraph framework agent, and a LangGraph multi-agent supervisor with conversational memory), evaluated with the same config-driven, regression-aware methodology as the original posttraining pipeline. See [Agentic Extension](#agentic-extension) below.

## Three Axis Evaluation

| Axis | What it measures | Method |
|------|-----------------|--------|
| **Capability** | Basic arithmetic accuracy | Randomly generated addition prompts, scored by exact integer match |
| **Robustness** | Sensitivity to prompt wording | Same arithmetic tasks with perturbed instructions, delta between base and perturbed accuracy |
| **Safety** | Refusal behavior on harmful prompts | 24 harmful + 25 benign prompts, scored by refusal phrase heuristics |

## System Architecture

```
Synthetic Dataset Generation
            ↓
      Data Validation
            ↓
   Posttraining (QLoRA)
            ↓
     Three Axis Evaluation
            ↓
    Experiment Comparison
```

## Agentic Extension
Building on the same eval-driven methodology above, this fork adds five new eval axes covering retrieval-augmented generation and increasingly complex agent architectures — from a single hand-rolled tool-calling loop to a LangGraph multi-agent supervisor with conversational memory. Every axis reuses the existing `TASK_REGISTRY` pattern, ships as an opt-in config override (the default 3-axis pipeline is untouched), and produces the same deterministic, regression-comparable output as the original framework.

### What was added
| Axis | What it measures | Backing technology |
|---|---|---|
| RAG groundedness (`rag_qa`) | Whether retrieval improves answer accuracy over the model's unaided knowledge | Chroma vector store, sentence-transformers embeddings |
| Hand-rolled tool-use (`agent_tool_capability`) | A ReAct-style loop where the model decides whether to call a calculator tool | A real MCP stdio server + client, regex-parsed tool calls |
| Framework tool-use (`agent_framework_tool_use`) | The same task, orchestrated by a framework instead of hand-rolled parsing | LangGraph `create_react_agent` + `langchain-mcp-adapters` over the same MCP server |
| Multi-agent orchestration (`agent_supervisor_orchestration`) | Whether a supervisor correctly routes questions to the right specialist (calculator, weather, code execution) | LangGraph `create_supervisor` coordinating three specialist agents, three separate MCP servers |
| Conversational memory (`agent_memory_followup`) | Whether an agent correctly recalls prior-turn state across separate calls | LangGraph `MemorySaver` checkpointer, thread-scoped state |

Also included: a sandboxed Python code-execution MCP tool (OS-level resource limits — CPU/memory/process caps — rather than container isolation, since Docker-in-Docker isn't reliably available on RunPod), optional Langfuse tracing wired into every agent task, and an optional LiteLLM-backed LLM-as-judge for RAG groundedness (defaults to a local Ollama model, no API key required).

### Architecture
```
                     ┌───────────────────────┐
                     │   TASK_REGISTRY        │   (existing pattern,
                     │   (run_eval.py)        │    5 new tasks added)
                     └───────────┬────────────┘
                                 │
     ┌───────────┬───────────────┼───────────────┬───────────┐
     ▼           ▼               ▼               ▼           ▼
  rag_qa    agent_tool_    agent_framework_  agent_supervisor  agent_memory
 (Chroma)   capability     tool_use          _orchestration    _followup
            (hand-rolled   (LangGraph        (LangGraph        (LangGraph +
             MCP loop)      ReAct agent)      supervisor +      MemorySaver)
                                               3 specialists)
                                    │                │
                                    └────────┬───────┘
                                             ▼
                              ┌───────────────────────────┐
                              │   MCP stdio servers         │
                              │   calculator / weather /    │
                              │   sandbox (execute_python)   │
                              └───────────────────────────┘
```



### Running the new axes

Each axis is a normal `src.eval.run_eval` invocation with a dedicated
override config — nothing about the default `configs/eval.yaml` pipeline
changes:

```bash
# RAG groundedness
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_rag.yaml \
    --output_dir results/rag \
    --mode baseline

# Hand-rolled MCP tool-use agent
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_agentic.yaml \
    --output_dir results/agentic \
    --mode baseline

# LangGraph framework agent
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_langgraph_agent.yaml \
    --output_dir results/langgraph_agent \
    --mode baseline

# Multi-agent supervisor
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_supervisor_agent.yaml \
    --output_dir results/supervisor_agent \
    --mode baseline

# Conversational memory
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_agent_memory.yaml \
    --output_dir results/agent_memory \
    --mode baseline
```


`scripts/setup_runpod.sh` runs all of the above (plus the original 3-axis
pipeline) as a single 10-step script, and generates a results report as its
final step:

```bash
python3 -m src.report_agentic --output results/report_agentic.md
```

### Findings

Full results, including per-specialist delegation rates and root-cause
analysis of two anomalies found during evaluation, are in
[`results/report_agentic.md`](results/report_agentic.md). Headline results
from two independent runs (numbers reproduced exactly across both):

- **RAG retrieval genuinely helps**: 100% accuracy with retrieved context vs.
96% without (+4pp groundedness delta).
- **Framework choice didn't matter for this task**: the hand-rolled MCP loop
and the LangGraph agent scored identically (96%) — the task was too easy
to discriminate between them.
- **Multi-agent delegation is selective, not general**: the supervisor
reliably delegates arithmetic to its calculator specialist (100% success),
but never genuinely delegates to its weather or code-execution
specialists — it narrates an intention to route and then answers directly
instead, which is only visible on tasks the model can't bluff from its own
knowledge (weather).
- **Conversational memory works**: all sampled follow-up turns correctly
recalled prior-turn state via LangGraph's checkpointer.


### Known limitations

- Sandboxed code execution uses OS-level resource limits, not container
  isolation (a deliberate RunPod-compatibility tradeoff, not an oversight).
- Langfuse tracing and the LiteLLM judge are both wired into the pipeline
  but require external setup (a Langfuse account, a running Ollama instance)
  to actually activate — otherwise they no-op safely.
- Guardrails (active input/output filtering or tool-call allowlisting) were
  not implemented — the safety axis measures unsafe behavior, it doesn't
  prevent it.
- A known scoring artifact exists in `score_agent_framework_answer`
  (substring match doesn't strip thousands-separator commas), which
  understates the code-execution specialist's true failure rate slightly.



## Project Structure

```
├── src/
│   ├── train.py              # QLoRA fine-tuning pipeline
│   ├── eval/
│   │   ├── run_eval.py       # Eval pipeline — 3 original axes + 5 agentic axes
│   │   ├── scoring.py        # Metric scoring utilities
│   │   ├── tasks.py          # Task generation utilities
│   │   └── probes.py         # Diagnostic probes
│   ├── compare.py            # Baseline vs posttrained comparison
│   ├── report_agentic.py     # Auto-generates results/report_agentic.md
│   ├── config.py             # YAML config loading and validation
│   ├── errors.py             # Project wide error hierarchy
│   ├── train_artifacts.py    # Shared I/O and metadata utilities
│   ├── rag/
│   │   ├── corpus.py         # Synthetic fact corpus + item generator
│   │   └── retriever.py      # Chroma-backed embedding retrieval
│   ├── agent/
│   │   ├── mcp_calculator_server.py  # MCP tool: calculator
│   │   ├── mcp_weather_server.py     # MCP tool: weather (cached fixtures)
│   │   ├── mcp_sandbox_server.py     # MCP tool: sandboxed code execution
│   │   ├── tool_agent.py             # Hand-rolled ReAct loop over MCP
│   │   ├── langgraph_agent.py        # LangGraph single-agent over MCP
│   │   ├── multi_agent.py            # LangGraph supervisor + 3 specialists
│   │   ├── agent_framework_items.py  # Calc/weather/code item generator
│   │   ├── sandbox_items.py          # Code-execution item generator
│   │   └── memory_items.py           # Hand-crafted memory-followup fixtures
│   ├── observability/
│   │   └── tracing.py        # Optional Langfuse callback handler
│   ├── llm_proxy/
│   │   └── litellm_client.py # Optional LiteLLM-backed groundedness judge
│   └── utils/
│       ├── config_utils.py   # Deep merge, YAML loading
│       └── logging_setup.py  # Logging configuration
├── configs/
│   ├── posttrain.yaml        # Training config (Mistral-7B, LoRA r=16)
│   ├── eval.yaml              # Evaluation config (400 prompts, 3 tasks)
│   └── overrides/             # Tier + axis specific overrides (smoke, synth,
│                                extended, rag, agentic, langgraph_agent,
│                                supervisor_agent, agent_memory)
├── scripts/
│   ├── generate_synthetic_dataset.py  # Deterministic training data generator
│   ├── fetch_weather_fixtures.py      # One-time live weather data fetch
│   └── setup_runpod.sh               # One command, 10-step RunPod pipeline
```

## Quick Start

```bash
# Install
git clone https://github.com/chrisploration/llm-posttraining-eval-agentic.git
cd llm-posttraining-eval-agentic
pip install -e .

# Generate training data (deterministic)
python3 -m scripts.generate_synthetic_dataset --num_examples 1000

# Run tests
python3 -m pytest tests/ -v
```

## Full Workflow

### 1. Baseline Evaluation

Evaluate the unmodified Mistral-7B-Instruct-v0.3 model:

```bash
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/baseline \
    --mode baseline
```

### 2. Posttraining (QLoRA)

Fine-tune with LoRA adapters on synthetic data:

```bash
python3 -m src.train \
    --config configs/posttrain.yaml \
    --override configs/overrides/posttrain_synth.yaml
```

### 3. Posttrained Evaluation

Evaluate the fine-tuned adapter checkpoint:

```bash
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/posttrained \
    --mode posttrained \
    --checkpoint checkpoints/post_v1 \
    --base_model mistralai/Mistral-7B-Instruct-v0.3
```

### 4. Compare Results

Generate a comparison report with regression detection:

```bash
python3 -m src.compare \
    --baseline results/baseline \
    --candidate results/posttrained \
    --format markdown \
    --output results/comparison.md \
    --fail-on-regression
```

`--fail-on-regression` exits with code 1 if any axis regressed — useful for CI-style gating. `scripts/setup_runpod.sh` intentionally omits this flag, since it runs the full pipeline including the agentic-extension axes below, which are unrelated to posttraining quality, and a regression shouldn't prevent those from running. The regression is still detected and written to `results/comparison.md` either way — only the process exit behavior differs.

## Configuration

The project uses a hierarchical YAML config system with deep-merge overrides:

```bash
# Base config only
python3 -m src.train --config configs/posttrain.yaml

# Base + override (override values take precedence)
python3 -m src.train --config configs/posttrain.yaml --override configs/overrides/posttrain_synth.yaml

# CLI overrides (highest precedence)
python3 -m src.train --config configs/posttrain.yaml --seed 123 --output_dir outputs/experiment_1
```

Available override tiers:

- `posttrain_smoke.yaml` — minimal run for testing (10 examples, no quantization)
- `posttrain_synth.yaml` — points training to generated synthetic data
- `eval_smoke.yaml` — fast evaluation (50 prompts) for quick sanity checks
- `eval_extended.yaml` — high volume evaluation (2000 prompts) for variance reduction (tighter confidence intervals)
- `eval_stress_robustness.yaml` — extended robustness testing with additional perturbation patterns

```bash
# Fast sanity check
python3 -m src.eval.run_eval --config configs/eval.yaml --override configs/overrides/eval_smoke.yaml

# Extended eval for tighter confidence intervals
python3 -m src.eval.run_eval --config configs/eval.yaml --override configs/overrides/eval_extended.yaml

# Deep robustness analysis
python3 -m src.eval.run_eval --config configs/eval.yaml --override configs/overrides/eval_stress_robustness.yaml
```

## Dataset Format

Training data is stored in JSONL format (chat format messages):

```json
{"prompt": "Explain overfitting in one sentence.", "completion": "Overfitting occurs when a model learns training-specific patterns that do not generalize well to new data."}
```

Synthetic dataset generation makes the project runnable without external datasets:

```bash
python3 -m scripts.generate_synthetic_dataset --num_examples 1000
```

## RunPod Deployment

On a RunPod GPU pod, the entire pipeline runs with one command:

```bash
bash scripts/setup_runpod.sh
```

This clones the repo, installs dependencies, generates training data, and runs all four
workflow steps automatically.

## Results

- **Original posttraining run**: `results/results.md` (analysis) and `results/comparison.md` (generated regression report)
- **Agentic extension**: `results/report_agentic.md` — see [Findings](#findings) above for the headline numbers

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a specific test file
python3 -m pytest tests/test_scoring.py -v

# Unit tests only (no GPU or model download required)
python3 -m pytest tests/ -v --ignore=tests/test_load_model_smoke.py --ignore=tests/test_train_smoke_artifacts.py
```

Two tests require a GPU and the Mistral-7B model to be downloaded locally:
`test_load_model_smoke.py` and `test_train_smoke_artifacts.py`. All other tests
run without GPU.

## Requirements

- Python >= 3.10
- PyTorch with CUDA (for GPU training/evaluation)
- See `requirements.txt` for full dependency list

## Limitations

- Synthetic data is simplistic relative to real world posttraining datasets
- Experiments are small scale (single GPU)
- No distributed training support yet
- Evaluation coverage is limited compared with production ML systems

## Future Work

**Original pipeline:**
- Preference learning or RLHF-style extensions
- Larger and more realistic datasets
- Distributed or multi-GPU training
- Richer evaluation benchmarks
- Experiment tracking integration

**Agentic extension:**
- Add active guardrails (input/output filtering, tool-call allowlisting). Currently the safety axis only measures unsafe behavior, it doesn't prevent it
- Activate Langfuse tracing and the LiteLLM judge end-to-end (both are wired in but need external setup — a Langfuse account, a running Ollama instance — to actually produce output)
- A harder arithmetic task to meaningfully differentiate the hand-rolled and LangGraph agents, since two-digit addition is solvable without either agent's tool-calling path


## License

MIT License