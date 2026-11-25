import hashlib
import math
import re

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.core.logging import get_logger

logger = get_logger("local_embedding_adapter")


class LocalEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Local embedding adapter producing 384-dimensional dense vectors.
    Supports FastEmbed / Sentence-Transformers if installed,
    with an intelligent fallback hashing/subword vectorizer that ensures
    100% zero-dependency local execution across all platforms.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self._model_name = model_name
        self._dimension = dimension
        self._fastembed_model = None
        self._sentence_transformer = None

        # Map common names to FastEmbed registry identifiers
        fe_name = self._model_name
        if "minilm" in self._model_name.lower():
            fe_name = "sentence-transformers/all-MiniLM-L6-v2"
        elif "bge-small" in self._model_name.lower():
            fe_name = "BAAI/bge-small-en-v1.5"

        # Attempt to load FastEmbed or SentenceTransformers if present
        try:
            from fastembed import TextEmbedding

            self._fastembed_model = TextEmbedding(model_name=fe_name)
            logger.info(f"Loaded FastEmbed local model: {fe_name} (dim={self._dimension})")
        except Exception as e:
            logger.debug(
                f"FastEmbed load error: {e}; falling back to SentenceTransformers or built-in"
            )
            try:
                from sentence_transformers import SentenceTransformer

                self._sentence_transformer = SentenceTransformer(self._model_name)
                logger.info(f"Loaded SentenceTransformer local model: {self._model_name}")
            except Exception:
                logger.info(
                    f"Using built-in zero-dependency local vectorizer for {self._model_name} (dim={self._dimension})"
                )

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _fallback_vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        # Weight words and character n-grams
        for i, token in enumerate(tokens):
            weight = 1.0 / (1.0 + 0.05 * min(i, 20))
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vec[idx] += sign * weight

            # Bigrams
            if len(token) >= 4:
                for j in range(len(token) - 2):
                    sub = token[j : j + 3]
                    sh = int(hashlib.md5(sub.encode("utf-8")).hexdigest()[:6], 16)
                    sidx = sh % self._dimension
                    vec[sidx] += 0.3 * (1.0 if sh % 2 == 0 else -1.0)

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._fastembed_model is not None:
            embeddings = list(self._fastembed_model.embed(texts))
            return [e.tolist() for e in embeddings]

        if self._sentence_transformer is not None:
            embeddings = self._sentence_transformer.encode(texts, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]

        return [self._fallback_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_documents([text])
        return res[0]
