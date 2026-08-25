import unittest
from random import Random

from src.agent.sandbox_items import make_code_execution_items


class TestMakeCodeExecutionItems(unittest.TestCase):
    def test_generates_requested_count(self) -> None:
        items = make_code_execution_items(15, Random(1))
        self.assertEqual(len(items), 15)

    def test_deterministic_for_same_seed(self) -> None:
        items_a = make_code_execution_items(20, Random(7))
        items_b = make_code_execution_items(20, Random(7))
        self.assertEqual(items_a, items_b)

    def test_answers_are_valid_non_negative_integers(self) -> None:
        items = make_code_execution_items(50, Random(9))
        for it in items:
            self.assertTrue(it["answer"].isdigit(), f"bad answer for {it['id']}: {it['answer']}")

    def test_kinds_are_known(self) -> None:
        items = make_code_execution_items(30, Random(3))
        kinds = {it["kind"] for it in items}
        self.assertTrue(kinds.issubset({"sum_range", "factorial", "fibonacci"}))


if __name__ == "__main__":
    unittest.main()