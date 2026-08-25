import unittest


class TestMcpSandboxServer(unittest.TestCase):
    def test_simple_computation(self) -> None:
        from src.agent.mcp_sandbox_server import execute_python

        self.assertEqual(execute_python("print(2 + 2)"), "4")


    def test_syntax_error_raises(self) -> None:
        from src.agent.mcp_sandbox_server import execute_python

        with self.assertRaises(RuntimeError):
            execute_python("this is not valid python(((")


    def test_memory_limit_enforced(self) -> None:
        from src.agent.mcp_sandbox_server import execute_python

        # Attempt to allocate ~1GB, well over the 256MB cap — should fail, not hang.
        with self.assertRaises(RuntimeError):
            execute_python("x = bytearray(1_000_000_000)")


if __name__ == "__main__":
    unittest.main()