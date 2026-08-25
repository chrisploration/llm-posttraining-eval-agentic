from __future__ import annotations

from typing import Any

MEMORY_FIXTURES: list[dict[str, Any]] = [
    {
        "id": "mem_calc_followup_1",
        "thread_id": "mem_thread_1",
        "turns": [
            {"prompt": "What is 12 + 30?", "answer": "42"},
            {"prompt": "Now multiply that result by 2.", "answer": "84"}
        ]
    },
    {
        "id": "mem_calc_followup_2",
        "thread_id": "mem_thread_2",
        "turns": [
            {"prompt": "What is 100 - 35?", "answer": "65"},
            {"prompt": "Subtract 15 from that result.", "answer": "50"}
        ]
    },
    {
        "id": "mem_calc_followup_3",
        "thread_id": "mem_thread_3",
        "turns": [
            {"prompt": "What is 7 times 6?", "answer": "42"},
            {"prompt": "Add 8 to that result.", "answer": "50"}
        ]
    }
]