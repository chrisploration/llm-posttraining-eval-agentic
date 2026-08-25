import unittest

from src.llm_proxy.litellm_client import judge_groundedness


class TestJudgeGroundedness(unittest.TestCase):
    def test_returns_none_when_provider_unreachable(self) -> None:
        result = judge_groundedness(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "Paris",
            model="ollama/definitely-not-a-real-model"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()