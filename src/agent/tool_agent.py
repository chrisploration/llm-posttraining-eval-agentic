from __future__ import annotations

import asyncio
import re
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:\s*calculator\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*[\"']?([+\-*/])[\"']?\s*\)",
    re.IGNORECASE
)

_SERVER_PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "src.agent.mcp_calculator_server"])

_INT_RE = re.compile(r"(-?\d+)")


def format_agent_prompt(question: str) -> str:
    return (
        "You can use a calculator tool for arithmetic. "
        "If you need it, respond with EXACTLY one line in this format: "
        "TOOL_CALL: calculator(<a>, <b>, <op>) where <op> is one of + - * /. "
        "Otherwise, answer directly with just the number.\n\n"
        f"Question: {question}"
    )


def parse_tool_call(generation: str) -> tuple[float, float, str] | None:
    """Extract (a, b, op) from a TOOL_CALL: line, or None if not present."""
    match = _TOOL_CALL_RE.search(generation)
    if not match:
        return None
    a, b, op = match.groups()
    return float(a), float(b), op


async def _call_calculator_async(a: float, b: float, op: str) -> str:
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("calculator", {"a": a, "b": b, "op": op})
            return result.content[0].text


def call_calculator(a: float, b: float, op: str) -> str:
    """Synchronous wrapper around the MCP calculator tool call."""
    return asyncio.run(_call_calculator_async(a, b, op))


def resolve_agent_answer(generation: str) -> tuple[str, bool]:
    """Resolve a model generation to a final numeric answer string.

    Returns (answer, used_tool). Dispatches to the MCP calculator when a
    valid TOOL_CALL is present; otherwise falls back to extracting the last
    integer in the generation, same as the raw-capability scorer.
    """
    call = parse_tool_call(generation)
    if call is not None:
        a, b, op = call
        try:
            return call_calculator(a, b, op), True
        except Exception:
            pass  # fall through to direct-parse fallback

    matches = _INT_RE.findall(generation)
    return (matches[-1] if matches else ""), False