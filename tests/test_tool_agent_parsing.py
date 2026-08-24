import unittest
from unittest.mock import patch

from src.agent.tool_agent import parse_tool_call, resolve_agent_answer


class TestParseToolCall(unittest.TestCase):
    def test_valid_tool_call(self) -> None:
        result = parse_tool_call("TOOL_CALL: calculator(47, 68, +)")
        self.assertEqual(result, (47.0, 68.0, "+"))

    def test_no_tool_call_returns_none(self) -> None:
        self.assertIsNone(parse_tool_call("The answer is 115"))


class TestResolveAgentAnswer(unittest.TestCase):
    @patch("src.agent.tool_agent.call_calculator", return_value="115")
    def test_uses_tool_when_call_present(self, mock_call) -> None:
        answer, used_tool = resolve_agent_answer("TOOL_CALL: calculator(47, 68, +)")
        self.assertEqual(answer, "115")
        self.assertTrue(used_tool)
        mock_call.assert_called_once_with(47.0, 68.0, "+")

    def test_falls_back_to_direct_parse(self) -> None:
        answer, used_tool = resolve_agent_answer("The answer is 115")
        self.assertEqual(answer, "115")
        self.assertFalse(used_tool)


if __name__ == "__main__":
    unittest.main()