from __future__ import annotations

import os
from typing import Any


def get_langfuse_handler() -> Any | None:
    """Return a configured Langfuse CallbackHandler, or None if not configured."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return None

    from langfuse.langchain import CallbackHandler
    return CallbackHandler()