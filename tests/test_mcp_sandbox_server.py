import shutil
import unittest


@unittest.skipUnless(shutil.which("docker"), "docker not installed/available; skipping sandbox test")
class TestMcpSandboxServer(unittest.TestCase):
    def test_simple_computation(self) -> None:
        from src.agent.mcp_sandbox_server import execute_python

        self.assertEqual(execute_python("print(2 + 2)"), "4")

    def test_syntax_error_raises(self) -> None:
        from src.agent.mcp_sandbox_server import execute_python

        with self.assertRaises(RuntimeError):
            execute_python("this is not valid python(((")


if __name__ == "__main__":
    unittest.main()