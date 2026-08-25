#!/bin/bash
set -e

cd /workspace

# Setup
if [ ! -d "llm-posttraining-eval-agentic" ]; then
    git clone https://github.com/chrisploration/llm-posttraining-eval-agentic llm-posttraining-eval-agentic
fi

cd llm-posttraining-eval-agentic
git pull origin main
pip install --upgrade setuptools pip
pip install pytest
pip install -e .

# Verify environment
# Run GPU-heavy training smoke test first in its own process (needs full VRAM).
python3 -m pytest tests/test_train_smoke_artifacts.py -v
# Run remaining tests in a separate process to avoid CUDA OOM conflicts.
python3 -m pytest tests/ --ignore=tests/test_train_smoke_artifacts.py -v

# Fetch weather fixtures for the agent_framework_tool_use task (needs network, one-time)
if [ ! -f "src/agent/weather_fixtures.json" ]; then
    python3 -m scripts.fetch_weather_fixtures
fi

# Generate training data
python3 -m scripts.generate_synthetic_dataset --num_examples 1000 --force
echo "Generated training data: data/synthetic/train.jsonl (1000 examples)"

echo "=== Setup complete. Running pipeline... ==="

# Clear any stale output directories from a previous run — guard_output_dir_empty
# refuses to write into a non-empty directory, and a fresh clone still carries
# whatever results/ content is committed to git history.
rm -rf results/baseline results/posttrained results/rag results/agentic \
       results/langgraph_agent results/supervisor_agent results/agent_memory


# Step 1: Baseline evaluation
echo ""
echo "=== Step 1/10: Baseline evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/baseline \
    --mode baseline


# Step 2: Posttraining
echo ""
echo "=== Step 2/10: Posttraining ==="
python3 -m src.train \
    --config configs/posttrain.yaml \
    --override configs/overrides/posttrain_synth.yaml


# Step 3: Posttrained evaluation
echo ""
echo "=== Step 3/10: Posttrained evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/posttrained \
    --mode posttrained \
    --checkpoint checkpoints/post_v1 \
    --base_model mistralai/Mistral-7B-Instruct-v0.3


# Step 4: Compare baseline vs post-trained results
# omits --fail-on-regression: this script runs the full
# pipeline including axes unrelated to posttraining (RAG, agents), so a
# detected regression is recorded but doesn't abort the remaining steps.
# Use --fail-on-regression directly via src.compare for CI-style gating.
echo ""
echo "=== Step 4/10: Compare baseline vs post-trained results ==="
python3 -m src.compare \
    --baseline results/baseline \
    --candidate results/posttrained \
    --format markdown \
    --output results/comparison.md \
    #--fail-on-regression


# Step 5: RAG groundedness evaluation
echo ""
echo "=== Step 5/10: RAG groundedness evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_rag.yaml \
    --override configs/overrides/eval_smoke.yaml \
    --output_dir results/rag \
    --mode baseline


# Step 6: Hand-rolled MCP tool-use agent evaluation
echo ""
echo "=== Step 6/10: Hand-rolled MCP tool-use agent evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_agentic.yaml \
    --override configs/overrides/eval_smoke.yaml \
    --output_dir results/agentic \
    --mode baseline


# Step 7: LangGraph framework agent evaluation
echo ""
echo "=== Step 7/10: LangGraph framework agent evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_langgraph_agent.yaml \
    --override configs/overrides/eval_smoke.yaml \
    --output_dir results/langgraph_agent \
    --mode baseline


# Step 8: Multi-agent supervisor evaluation (calculator/weather/coder specialists)
echo ""
echo "=== Step 8/10: Multi-agent supervisor evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_supervisor_agent.yaml \
    --override configs/overrides/eval_smoke.yaml \
    --output_dir results/supervisor_agent \
    --mode baseline


# Step 9: Agent memory-followup evaluation
echo ""
echo "=== Step 9/10: Agent memory-followup evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --override configs/overrides/eval_agent_memory.yaml \
    --output_dir results/agent_memory \
    --mode baseline


# Step 10: Generate the agentic-extension results report
echo ""
echo "=== Step 10/10: Generate results report ==="
python3 -m src.report_agentic --output results/report_agentic.md


echo ""
echo "=== Pipeline complete ==="
echo "Comparison report: results/comparison.md"
echo "RAG results: results/rag/metrics.json"
echo "Agentic (hand-rolled MCP) results: results/agentic/metrics.json"
echo "Agentic (LangGraph) results: results/langgraph_agent/metrics.json"
echo "Agentic (multi-agent supervisor) results: results/supervisor_agent/metrics.json"
echo "Agentic (memory followup) results: results/agent_memory/metrics.json"
echo ""
echo "Not run automatically (optional, need extra setup):"
echo " - Langfuse tracing: export LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST first, then re-run any agent task"
echo " - LiteLLM judge on rag_qa: needs 'ollama serve' + 'ollama pull mistral' running, then LITELLM_JUDGE=1 before the RAG eval step"
echo "Agentic extension report: results/report_agentic.md"