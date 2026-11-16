from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.core.exceptions import RAGException
from app.core.logging import get_logger

logger = get_logger("openai_embedding_adapter")


class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    OpenAI Embeddings Adapter (e.g. text-embedding-3-small, 1536 dim).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        max_retries: int = 3,
    ):
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimension = dimension
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RAGException("OPENAI_API_KEY is not configured", status_code=500)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "input": batch,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            return [item["embedding"] for item in data["data"]]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            # Process in batches of 32
            results = []
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                embeddings = await self._embed_batch(batch)
                results.extend(embeddings)
            return results
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            raise RAGException(f"OpenAI embedding generation failed: {str(e)}", status_code=502)

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_documents([text])
        return res[0]
