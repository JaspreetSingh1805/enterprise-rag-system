import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.adapters.llm.base import BaseLLMAdapter, LLMResponse
from app.core.exceptions import RAGException
from app.core.logging import get_logger

logger = get_logger("ollama_llm_adapter")


class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama Local LLM Adapter for completely free, local execution.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:latest",
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "ollama"

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            reraise=True,
        )
        async def _call_ollama() -> dict:
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                res.raise_for_status()
                return res.json()

        try:
            data = await _call_ollama()
            content = data.get("message", {}).get("content", "")
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            return LLMResponse(
                text=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model_name=self.model_name,
            )
        except Exception as e:
            logger.error(f"Ollama call failed: {str(e)}")
            raise RAGException(f"Ollama completion failed: {str(e)}", status_code=502)
