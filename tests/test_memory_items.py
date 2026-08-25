import unittest

from src.agent.memory_items import MEMORY_FIXTURES


class TestMemoryFixtures(unittest.TestCase):
    def test_each_fixture_has_two_turns(self) -> None:
        for fixture in MEMORY_FIXTURES:
            self.assertEqual(len(fixture["turns"]), 2)

    def test_thread_ids_are_unique(self) -> None:
        thread_ids = [f["thread_id"] for f in MEMORY_FIXTURES]
        self.assertEqual(len(thread_ids), len(set(thread_ids)))

    def test_fixture_ids_are_unique(self) -> None:
        ids = [f["id"] for f in MEMORY_FIXTURES]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()