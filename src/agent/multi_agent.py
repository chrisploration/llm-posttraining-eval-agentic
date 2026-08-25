from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.agent.langgraph_agent import build_chat_model

_MCP_SERVERS: dict[str, dict[str, Any]] = {
    "calculator": {
        "command": sys.executable,
        "args": ["-m", "src.agent.mcp_calculator_server"],
        "transport": "stdio"
    },
    "weather": {
        "command": sys.executable,
        "args": ["-m", "src.agent.mcp_weather_server"],
        "transport": "stdio"
    },
    "sandbox": {
        "command": sys.executable,
        "args": ["-m", "src.agent.mcp_sandbox_server"],
        "transport": "stdio"
    }
}

# Module-level singleton so conversation state actually persists across
# separate run_supervisor(...) calls that share a thread_id.
_checkpointer = MemorySaver()


async def _build_supervisor(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, gen_params: Mapping[str, Any]) -> Any:
    chat_model = build_chat_model(model, tokenizer, gen_params)

    client = MultiServerMCPClient(_MCP_SERVERS)
    calc_tools = await client.get_tools(server_name="calculator")
    weather_tools = await client.get_tools(server_name="weather")
    sandbox_tools = await client.get_tools(server_name="sandbox")

    calc_agent = create_react_agent(chat_model, calc_tools, name="calc_agent")
    weather_agent = create_react_agent(chat_model, weather_tools, name="weather_agent")
    coder_agent = create_react_agent(chat_model, sandbox_tools, name="coder_agent")

    supervisor = create_supervisor(
        [calc_agent, weather_agent, coder_agent],
        model=chat_model,
        prompt=(
            "You route each question to exactly one specialist: "
            "calc_agent for arithmetic, weather_agent for weather questions, "
            "coder_agent for anything that needs running code."
        )
    ).compile(checkpointer=_checkpointer)

    return supervisor


async def _run_supervisor_async(question: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, gen_params: Mapping[str, Any], *, thread_id: str, callbacks: list[Any] | None) -> str:
    supervisor = await _build_supervisor(model, tokenizer, gen_params)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks

    result = await supervisor.ainvoke({"messages": [("user", question)]}, config=config)
    final_message = result["messages"][-1]
    return str(final_message.content)


def run_supervisor(question: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, gen_params: Mapping[str, Any], thread_id: str = "default", callbacks: list[Any] | None = None) -> str:
    """Synchronous entry point: run one question through the supervisor multi-agent graph."""
    return asyncio.run(_run_supervisor_async(question, model, tokenizer, gen_params, thread_id=thread_id, callbacks=callbacks))