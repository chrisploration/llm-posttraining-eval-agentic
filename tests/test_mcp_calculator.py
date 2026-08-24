import unittest

from src.agent.tool_agent import call_calculator


class TestMcpCalculator(unittest.TestCase):
    def test_addition(self) -> None:
        self.assertEqual(call_calculator(2, 3, "+"), "5")

    def test_subtraction(self) -> None:
        self.assertEqual(call_calculator(10, 4, "-"), "6")

    def test_multiplication(self) -> None:
        self.assertEqual(call_calculator(6, 7, "*"), "42")

    def test_division(self) -> None:
        self.assertEqual(call_calculator(9, 2, "/"), "4.5")


if __name__ == "__main__":
    unittest.main()