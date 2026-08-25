from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

_FIXTURES_PATH = Path(__file__).parent / "weather_fixtures.json"


def _load_fixtures() -> dict[str, dict[str, float]]:
    with open(_FIXTURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_agent_framework_items(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate a mix of arithmetic, weather, and code-execution questions with known expected answers."""
    fixtures = _load_fixtures()
    cities = sorted(fixtures.keys())

    items: list[dict[str, Any]] = []
    for i in range(n):
        roll = rng.random()
        if roll < 0.34:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            items.append({
                "id": f"agentfw_calc_{i}_{a}_{b}",
                "prompt": f"What is {a} + {b}?",
                "answer": str(a + b),
                "kind": "calculator"
            })
        elif roll < 0.67:
            city = rng.choice(cities)
            temp = fixtures[city]["temperature_c"]
            items.append({
                "id": f"agentfw_weather_{i}_{city.replace(' ', '_')}",
                "prompt": f"What is the current temperature in {city.title()}? Answer with the number in Celsius.",
                "answer": str(temp),
                "kind": "weather"
            })
        else:
            k = rng.randint(3, 8)
            items.append({
                "id": f"agentfw_code_{i}_{k}",
                "prompt": f"Write and run Python code to compute {k} factorial. Report only the final number.",
                "answer": str(math.factorial(k)),
                "kind": "code"
            })
    return items