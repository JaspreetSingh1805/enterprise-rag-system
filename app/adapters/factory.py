import threading
from typing import Optional

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.embeddings.local_adapter import LocalEmbeddingAdapter
from app.adapters.embeddings.mock_adapter import MockEmbeddingAdapter
from app.adapters.embeddings.openai_adapter import OpenAIEmbeddingAdapter
from app.adapters.llm.base import BaseLLMAdapter
from app.adapters.llm.mock_adapter import MockLLMAdapter
from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.core.config import settings
from app.core.logging import get_logger
from app.models.domain import (
    EmbeddingProviderType,
    KnowledgeBase,
    LLMProviderType,
)
from app.models.pipeline_config import EmbeddingConfig, LLMConfig

logger = get_logger("adapter_factory")


class AdapterRegistry:
    """
    Central factory and lifecycle registry for LLM and Embedding adapters.
    Ensures zero-downtime, runtime configuration switching without server restarts.
    Thread-safe instance caching prevents redundant model reinstantiations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_cache: dict[str, BaseLLMAdapter] = {}
        self._embedding_cache: dict[str, BaseEmbeddingAdapter] = {}
        self._llm_override: Optional[BaseLLMAdapter] = None
        self._embedding_override: Optional[BaseEmbeddingAdapter] = None

    def set_llm_override(self, adapter: Optional[BaseLLMAdapter]) -> None:
        """Testing hook to force a specific LLM adapter."""
        with self._lock:
            self._llm_override = adapter

    def set_embedding_override(self, adapter: Optional[BaseEmbeddingAdapter]) -> None:
        """Testing hook to force a specific Embedding adapter."""
        with self._lock:
            self._embedding_override = adapter

    def get_llm_adapter(self, config: Optional[LLMConfig] = None) -> BaseLLMAdapter:
        """
        Dynamically resolves or creates an LLM adapter based on the active LLMConfig.
        Switches between OpenAI, Ollama, and Mock providers in real time.
        """
        with self._lock:
            if self._llm_override is not None:
                return self._llm_override

            provider = config.provider if config else LLMProviderType(settings.default_llm_provider)
            model_name = config.model_name if config else settings.openai_model
            temp = config.temperature if config else 0.1
            max_tokens = config.max_tokens if config else 1000

            cache_key = f"{provider}:{model_name}:{temp}:{max_tokens}"
            if cache_key in self._llm_cache:
                return self._llm_cache[cache_key]

            logger.info(f"Creating LLM adapter: provider='{provider}', model='{model_name}'")

            adapter: BaseLLMAdapter
            if provider == LLMProviderType.OPENAI:
                adapter = OpenAIAdapter(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    model_name=model_name,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            elif provider == LLMProviderType.OLLAMA:
                adapter = OllamaAdapter(
                    base_url=settings.ollama_base_url,
                    model_name=model_name,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            else:
                adapter = MockLLMAdapter(model_name=model_name)

            self._llm_cache[cache_key] = adapter
            return adapter

    def get_embedding_adapter(
        self, config: Optional[EmbeddingConfig] = None
    ) -> BaseEmbeddingAdapter:
        """
        Dynamically resolves or creates an Embedding adapter based on the active EmbeddingConfig.
        Switches between FastEmbed (Local), OpenAI, and Mock providers in real time.
        """
        with self._lock:
            if self._embedding_override is not None:
                return self._embedding_override

            provider = (
                config.provider
                if config
                else EmbeddingProviderType(settings.default_embedding_provider)
            )
            model_name = config.model_name if config else settings.local_embedding_model
            dimension = config.dimension if config else 384

            cache_key = f"{provider}:{model_name}:{dimension}"
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

            logger.info(
                f"Creating Embedding adapter: provider='{provider}', model='{model_name}', dim={dimension}"
            )

            adapter: BaseEmbeddingAdapter
            if provider == EmbeddingProviderType.LOCAL:
                adapter = LocalEmbeddingAdapter(
                    model_name=model_name,
                    dimension=dimension,
                )
            elif provider == EmbeddingProviderType.OPENAI:
                adapter = OpenAIEmbeddingAdapter(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    model_name=model_name,
                    dimension=dimension,
                )
            else:
                adapter = MockEmbeddingAdapter(
                    model_name=model_name,
                    dimension=dimension,
                )

            self._embedding_cache[cache_key] = adapter
            return adapter

    def get_embedding_adapter_for_kb(self, kb: KnowledgeBase) -> BaseEmbeddingAdapter:
        """
        Resolves the exact embedding adapter required by a Knowledge Base's schema
        (guaranteeing dimensional compatibility during retrieval and safe re-indexing).
        """
        with self._lock:
            if self._embedding_override is not None:
                return self._embedding_override

        model_name = kb.embedding_model.lower()
        dim = kb.embedding_dimension

        if "openai" in model_name or "text-embedding" in model_name:
            cfg = EmbeddingConfig(
                provider=EmbeddingProviderType.OPENAI,
                model_name=kb.embedding_model,
                dimension=dim,
            )
        elif "mock" in model_name:
            cfg = EmbeddingConfig(
                provider=EmbeddingProviderType.MOCK,
                model_name=kb.embedding_model,
                dimension=dim,
            )
        else:
            cfg = EmbeddingConfig(
                provider=EmbeddingProviderType.LOCAL,
                model_name=kb.embedding_model,
                dimension=dim,
            )

        return self.get_embedding_adapter(cfg)
