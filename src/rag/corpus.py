from __future__ import annotations

import random
from typing import Any

CORPUS: list[dict[str, str]] = [
    {"id": "doc_capital_france", "text": "The capital of France is Paris."},
    {"id": "doc_capital_japan", "text": "The capital of Japan is Tokyo."},
    {"id": "doc_capital_egypt", "text": "The capital of Egypt is Cairo."},
    {"id": "doc_capital_brazil", "text": "The capital of Brazil is Brasilia."},
    {"id": "doc_capital_australia", "text": "The capital of Australia is Canberra."},
    {"id": "doc_river_nile", "text": "The Nile is the longest river in Africa."},
    {"id": "doc_river_amazon", "text": "The Amazon is the largest river by discharge volume in the world."},
    {"id": "doc_planet_largest", "text": "Jupiter is the largest planet in the Solar System."},
    {"id": "doc_planet_closest", "text": "Mercury is the closest planet to the Sun."},
    {"id": "doc_ocean_largest", "text": "The Pacific Ocean is the largest and deepest ocean on Earth."},
    {"id": "doc_mountain_tallest", "text": "Mount Everest is the tallest mountain above sea level."},
    {"id": "doc_animal_fastest", "text": "The cheetah is the fastest land animal."},
    {"id": "doc_animal_largest", "text": "The blue whale is the largest animal on Earth."},
    {"id": "doc_element_lightest", "text": "Hydrogen is the lightest chemical element."},
    {"id": "doc_metal_conductive", "text": "Silver is the most electrically conductive metal."},
    {"id": "doc_language_speakers", "text": "Mandarin Chinese has the most native speakers of any language."},
    {"id": "doc_desert_largest", "text": "The Antarctic Desert is the largest desert in the world."},
    {"id": "doc_lake_deepest", "text": "Lake Baikal is the deepest lake in the world."},
    {"id": "doc_country_largest", "text": "Russia is the largest country in the world by area."},
    {"id": "doc_country_populous", "text": "India is the most populous country in the world."},
    {"id": "doc_bone_longest", "text": "The femur is the longest bone in the human body."},
    {"id": "doc_organ_largest", "text": "The skin is the largest organ in the human body."},
    {"id": "doc_metal_densest", "text": "Osmium is the densest naturally occurring metal."},
    {"id": "doc_star_nearest", "text": "Proxima Centauri is the nearest star to the Sun other than the Sun itself."},
    {"id": "doc_gas_atmosphere", "text": "Nitrogen makes up about 78 percent of Earth's atmosphere."},
    {"id": "doc_continent_smallest", "text": "Australia is the smallest continent by land area."},
    {"id": "doc_continent_largest", "text": "Asia is the largest continent by land area."},
    {"id": "doc_ocean_smallest", "text": "The Arctic Ocean is the smallest of the world's five oceans."},
    {"id": "doc_volcano_active", "text": "Mauna Loa is one of the largest active volcanoes on Earth."},
    {"id": "doc_currency_japan", "text": "The official currency of Japan is the yen."}
]

# One templated question per doc, phrased so the expected answer is a short
# span lifted directly from the fact (keeps scoring a simple substring check).
_QUESTION_BY_DOC_ID: dict[str, tuple[str, str]] = {
    "doc_capital_france": ("What is the capital of France?", "Paris"),
    "doc_capital_japan": ("What is the capital of Japan?", "Tokyo"),
    "doc_capital_egypt": ("What is the capital of Egypt?", "Cairo"),
    "doc_capital_brazil": ("What is the capital of Brazil?", "Brasilia"),
    "doc_capital_australia": ("What is the capital of Australia?", "Canberra"),
    "doc_river_nile": ("Which river is the longest in Africa?", "Nile"),
    "doc_river_amazon": ("Which river has the largest discharge volume in the world?", "Amazon"),
    "doc_planet_largest": ("Which is the largest planet in the Solar System?", "Jupiter"),
    "doc_planet_closest": ("Which planet is closest to the Sun?", "Mercury"),
    "doc_ocean_largest": ("Which is the largest ocean on Earth?", "Pacific"),
    "doc_mountain_tallest": ("Which is the tallest mountain above sea level?", "Everest"),
    "doc_animal_fastest": ("Which is the fastest land animal?", "Cheetah"),
    "doc_animal_largest": ("Which is the largest animal on Earth?", "Blue whale"),
    "doc_element_lightest": ("Which is the lightest chemical element?", "Hydrogen"),
    "doc_metal_conductive": ("Which metal is the most electrically conductive?", "Silver"),
    "doc_language_speakers": ("Which language has the most native speakers?", "Mandarin"),
    "doc_desert_largest": ("Which is the largest desert in the world?", "Antarctic"),
    "doc_lake_deepest": ("Which is the deepest lake in the world?", "Baikal"),
    "doc_country_largest": ("Which is the largest country by area?", "Russia"),
    "doc_country_populous": ("Which is the most populous country in the world?", "India"),
    "doc_bone_longest": ("Which is the longest bone in the human body?", "Femur"),
    "doc_organ_largest": ("Which is the largest organ in the human body?", "Skin"),
    "doc_metal_densest": ("Which is the densest naturally occurring metal?", "Osmium"),
    "doc_star_nearest": ("Which star is nearest to the Sun, other than the Sun?", "Proxima Centauri"),
    "doc_gas_atmosphere": ("Which gas makes up about 78 percent of Earth's atmosphere?", "Nitrogen"),
    "doc_continent_smallest": ("Which is the smallest continent by land area?", "Australia"),
    "doc_continent_largest": ("Which is the largest continent by land area?", "Asia"),
    "doc_ocean_smallest": ("Which is the smallest of the world's five oceans?", "Arctic"),
    "doc_volcano_active": ("Name one of the largest active volcanoes on Earth.", "Mauna Loa"),
    "doc_currency_japan": ("What is the official currency of Japan?", "yen")
}


def make_rag_items(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate n grounded QA items, each tied to one corpus doc id."""
    doc_ids = [d["id"] for d in CORPUS]
    items: list[dict[str, Any]] = []
    for i in range(n):
        doc_id = rng.choice(doc_ids)
        question, answer = _QUESTION_BY_DOC_ID[doc_id]
        items.append({
            "id": f"rag_{i}_{doc_id}",
            "prompt": question,
            "answer": answer,
            "gold_doc_id": doc_id
        })
    return items