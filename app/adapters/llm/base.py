from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str


class BaseLLMAdapter(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'openai', 'ollama', 'mock')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying model (e.g. 'gpt-4o-mini', 'llama3.2', 'mock-gpt-4o')."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate completion from LLM with bounded retries.
        """
        pass
