from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.adapters.llm.base import BaseLLMAdapter, LLMResponse
from app.core.exceptions import RAGException
from app.core.logging import get_logger

logger = get_logger("openai_llm_adapter")


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI and OpenAI-compatible (Groq, vLLM, etc.) LLM Adapter.
    Includes bounded retries with exponential backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        if not self.api_key:
            raise RAGException(
                "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your environment.",
                status_code=500,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            reraise=True,
        )
        async def _call_api() -> dict:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                res.raise_for_status()
                return res.json()

        try:
            data = await _call_api()
            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                text=choice,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                model_name=self.model_name,
            )
        except Exception as e:
            logger.error(f"OpenAI LLM call failed after retries: {str(e)}")
            raise RAGException(f"OpenAI completion failed: {str(e)}", status_code=502)
