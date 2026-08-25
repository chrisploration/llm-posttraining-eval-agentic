from __future__ import annotations

import math
import random
from typing import Any


def _fib(k: int) -> int:
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


def make_code_execution_items(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate n simple, independently-verifiable compute tasks."""
    kinds = ["sum_range", "factorial", "fibonacci"]
    items: list[dict[str, Any]] = []

    for i in range(n):
        kind = rng.choice(kinds)

        if kind == "sum_range":
            k = rng.randint(5, 50)
            prompt = f"Write and run Python code to compute the sum of the integers from 1 to {k} (inclusive). Report only the final number."
            answer = str(sum(range(1, k + 1)))
        elif kind == "factorial":
            k = rng.randint(3, 10)
            prompt = f"Write and run Python code to compute {k} factorial. Report only the final number."
            answer = str(math.factorial(k))
        else:
            k = rng.randint(5, 20)
            prompt = f"Write and run Python code to compute the {k}th Fibonacci number (0-indexed, fib(0)=0, fib(1)=1). Report only the final number."
            answer = str(_fib(k))

        items.append({
            "id": f"code_{i}_{kind}",
            "prompt": prompt,
            "answer": answer,
            "kind": kind
        })

    return items