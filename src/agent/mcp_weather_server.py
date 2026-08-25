from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

_FIXTURES_PATH = Path(__file__).parent / "weather_fixtures.json"


def _load_fixtures() -> dict[str, dict[str, float]]:
    with open(_FIXTURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _fetch_live(lat: float, lon: float) -> float:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
        timeout=10,
    )
    resp.raise_for_status()
    return float(resp.json()["current"]["temperature_2m"])


@mcp.tool()
def get_weather(city: str) -> str:
    """Return the current temperature in Celsius for a known fixed city."""
    fixtures = _load_fixtures()
    key = city.strip().lower()
    if key not in fixtures:
        raise ValueError(f"Unknown city: {city}. Known cities: {sorted(fixtures)}")

    entry = fixtures[key]
    temp_c = _fetch_live(entry["lat"], entry["lon"]) if os.environ.get("WEATHER_LIVE") == "1" else entry["temperature_c"]
    return str(temp_c)


if __name__ == "__main__":
    mcp.run(transport="stdio")