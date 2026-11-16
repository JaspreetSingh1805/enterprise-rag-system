import hashlib
import math
import re

from app.adapters.embeddings.base import BaseEmbeddingAdapter


class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Deterministic Mock Embedding Adapter.
    Uses feature hashing and L2 normalization to generate realistic dense embeddings.
    Texts sharing vocabulary will have proportional cosine similarities.
    """

    def __init__(self, model_name: str = "mock-minilm-l6", dimension: int = 384):
        self._model_name = model_name
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            # Deterministic hash to an index and sign
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_vector(text)
