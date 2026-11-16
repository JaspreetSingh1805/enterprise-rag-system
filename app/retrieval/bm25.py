import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from app.models.domain import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25SearchEngine:
    """
    In-memory BM25 keyword search engine scoped per knowledge base.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0

        candidates: list[tuple[Chunk, float]] = []
        for chunk, score in zip(self.chunks, raw_scores):
            if score <= 0:
                continue

            # Apply filters
            if filters:
                match = True
                for k, v in filters.items():
                    if k == "document_id" and chunk.document_id != v:
                        match = False
                        break
                    elif k in chunk.metadata and chunk.metadata[k] != v:
                        match = False
                        break
                if not match:
                    continue

            # Normalize to [0.0, 1.0]
            norm_score = min(1.0, float(score / max_score))
            candidates.append((chunk, norm_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
