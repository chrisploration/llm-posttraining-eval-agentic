from __future__ import annotations

import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sandbox-server")

_TIMEOUT_SECONDS = 15
_DOCKER_IMAGE = "python:3.12-slim"


@mcp.tool()
def execute_python(code: str) -> str:
    """Run Python code in an isolated, network-disabled Docker container and return stdout."""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-i",
                "--network", "none",
                "--memory", "128m",
                "--cpus", "0.5",
                "--pids-limit", "64",
                "--read-only",
                _DOCKER_IMAGE,
                "python", "-"
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"execute_python timed out after {_TIMEOUT_SECONDS}s") from e
    except FileNotFoundError as e:
        raise RuntimeError("docker is not installed or not on PATH") from e

    if result.returncode != 0:
        raise RuntimeError(f"execute_python failed (exit {result.returncode}): {result.stderr.strip()}")

    return result.stdout.strip()


if __name__ == "__main__":
    mcp.run(transport="stdio")