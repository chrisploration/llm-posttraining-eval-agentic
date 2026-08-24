import unittest

import numpy as np

from src.rag.retriever import Retriever


def _stub_embed_fn(texts):
    """Deterministic fake embedder: one-hot on first-word identity, no downloads."""
    vocab = ["paris", "tokyo", "cairo", "unrelated"]
    vecs = []
    for t in texts:
        v = np.zeros(len(vocab))
        low = t.lower()
        for i, word in enumerate(vocab):
            if word in low:
                v[i] = 1.0
        if v.sum() == 0:
            v[-1] = 1.0
        vecs.append(v)
    return np.array(vecs)


class TestRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [
            {"id": "doc_paris", "text": "Paris is a city."},
            {"id": "doc_tokyo", "text": "Tokyo is a city."},
            {"id": "doc_cairo", "text": "Cairo is a city."},
        ]
        self.retriever = Retriever(self.corpus, embed_fn=_stub_embed_fn)

    def test_retrieves_matching_doc_first(self) -> None:
        results = self.retriever.retrieve("Where is Paris?", top_k=1)
        self.assertEqual(results[0]["id"], "doc_paris")

    def test_top_k_respected(self) -> None:
        results = self.retriever.retrieve("Where is Tokyo?", top_k=2)
        self.assertEqual(len(results), 2)

    def test_results_include_score(self) -> None:
        results = self.retriever.retrieve("Where is Cairo?", top_k=1)
        self.assertIn("score", results[0])


if __name__ == "__main__":
    unittest.main()