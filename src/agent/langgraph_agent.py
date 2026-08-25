from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Any

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import pipeline as hf_pipeline

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
    }
}


def build_chat_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, gen_params: Mapping[str, Any]) -> ChatHuggingFace:
    """Wrap the already-loaded eval model/tokenizer as a LangChain chat model — no second model load."""
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=gen_params["max_new_tokens"],
        do_sample=gen_params["do_sample"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        num_beams=gen_params["num_beams"],
        pad_token_id=gen_params["pad_token_id"],
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm, tokenizer=tokenizer)


async def _run_agent_async(question: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, gen_params: Mapping[str, Any]) -> str:
    chat_model = build_chat_model(model, tokenizer, gen_params)

    client = MultiServerMCPClient(_MCP_SERVERS)
    tools = await client.get_tools()

    agent = create_react_agent(chat_model, tools)
    result = await agent.ainvoke({"messages": [("user", question)]})

    final_message = result["messages"][-1]
    return str(final_message.content)


def run_agent(question: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, gen_params: Mapping[str, Any]) -> str:
    """Synchronous entry point: run one question through the LangGraph MCP agent."""
    return asyncio.run(_run_agent_async(question, model, tokenizer, gen_params))