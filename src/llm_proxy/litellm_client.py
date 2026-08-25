from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are a strict grading assistant. Given a question, a set of \
source facts, and an answer, respond with exactly one word: YES if the \
answer is fully supported by the facts, or NO if it is not supported or \
contradicts the facts.

Facts:
{context}

Question: {question}

Answer: {answer}

Respond with exactly one word: YES or NO."""


def judge_groundedness(question: str, context: str, answer: str, *, model: str = "ollama/mistral") -> bool | None:
    """Ask a (possibly different, possibly hosted) model whether `answer` is grounded in `context`."""
    try:
        import litellm

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(context=context, question=question, answer=answer)}],
            temperature=0.0,
            max_tokens=5
        )
        verdict = response["choices"][0]["message"]["content"].strip().upper()
        return verdict.startswith("YES")
    except Exception:
        logger.warning("judge_groundedness: LiteLLM call failed (model=%s); skipping.", model, exc_info=True)
        return None