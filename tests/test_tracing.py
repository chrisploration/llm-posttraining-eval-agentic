import os
import unittest



class TestGetLangfuseHandler(unittest.TestCase):
    def test_returns_none_when_env_vars_unset(self) -> None:
        from src.observability.tracing import get_langfuse_handler

        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        self.assertIsNone(get_langfuse_handler())


if __name__ == "__main__":
    unittest.main()