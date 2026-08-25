from __future__ import annotations

import resource
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sandbox-server")

_TIMEOUT_SECONDS = 15
_MAX_MEMORY_BYTES = 256 * 1024 * 1024 # 256MB
_MAX_CPU_SECONDS = 10

def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_MEMORY_BYTES, _MAX_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (_MAX_CPU_SECONDS, _MAX_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

@mcp.tool()
def execute_python(code: str) -> str:
    """Run Python code in an isolated, network-disabled Docker container and return stdout."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            preexec_fn=_limit_resources
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"execute_python timed out after {_TIMEOUT_SECONDS}s") from e

    if result.returncode != 0:
        raise RuntimeError(f"execute_python failed (exit {result.returncode}): {result.stderr.strip()}")

    return result.stdout.strip()


if __name__ == "__main__":
    mcp.run(transport="stdio")