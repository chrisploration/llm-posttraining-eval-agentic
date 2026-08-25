import argparse
import json
import os

import requests

DEFAULT_OUT_PATH = os.path.join("src", "agent", "weather_fixtures.json")

CITIES: dict[str, tuple[float, float]] = {
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
    "cairo": (30.0444, 31.2357),
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "sydney": (-33.8688, 151.2093),
    "berlin": (52.5200, 13.4050),
    "moscow": (55.7558, 37.6173),
    "sao paulo": (-23.5505, -46.6333),
    "mumbai": (19.0760, 72.8777)
}


def fetch_all(cities: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    fixtures: dict[str, dict[str, float]] = {}
    for city, (lat, lon) in cities.items():
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
            timeout=10
        )
        resp.raise_for_status()
        temp_c = float(resp.json()["current"]["temperature_2m"])
        fixtures[city] = {"lat": lat, "lon": lon, "temperature_c": temp_c}
        print(f"{city}: {temp_c} C")
    return fixtures


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch live weather fixtures from Open-Meteo (one-time, needs network).")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help=f"Output JSON path. Default: {DEFAULT_OUT_PATH}")
    ap.add_argument("--force", action="store_true", help="Overwrite output file if it exists.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"File already exists, skipping: {args.out}. Use --force to overwrite.")
        return

    fixtures = fetch_all(CITIES)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Wrote {len(fixtures)} city fixtures to {args.out}")


if __name__ == "__main__":
    main()