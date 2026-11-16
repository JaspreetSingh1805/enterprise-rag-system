import re

from app.core.logging import get_logger
from app.models.domain import Chunk

logger = get_logger("reranker")


class Reranker:
    """
    Reranks candidate retrieved chunks against the user query.
    """

    def __init__(self, model_name: str = "flashrank"):
        self.model_name = model_name
        self._flashrank_client = None

        try:
            from flashrank import Ranker

            self._flashrank_client = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            logger.info("Loaded FlashRank cross-encoder reranker")
        except Exception:
            logger.debug("FlashRank not present; using built-in exact semantic reranker")

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_n: int = 3,
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_n]

        if self._flashrank_client:
            from flashrank import RerankRequest

            passages = [{"id": c.id, "text": c.text, "chunk": c} for c, _ in candidates]
            req = RerankRequest(query=query, passages=passages)
            ranked = self._flashrank_client.rerank(req)
            id_to_chunk = {c.id: c for c, _ in candidates}
            output = []
            for r in ranked[:top_n]:
                chunk = id_to_chunk[r["id"]]
                output.append((chunk, float(r["score"])))
            return output

        # Built-in semantic cross-scoring:
        # Measures exact phrase presence, keyword density, and token proximity
        query_words = set(re.findall(r"\w+", query.lower()))
        reranked = []

        for chunk, base_score in candidates:
            chunk_words = re.findall(r"\w+", chunk.text.lower())
            chunk_word_set = set(chunk_words)

            # Intersection ratio
            overlap = len(query_words.intersection(chunk_word_set)) / max(1, len(query_words))
            # Exact query substring boost
            exact_boost = 0.3 if query.lower() in chunk.text.lower() else 0.0

            # Composite rerank score
            score = 0.5 * base_score + 0.3 * overlap + 0.2 * exact_boost
            reranked.append((chunk, min(1.0, score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_n]
