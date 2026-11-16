from abc import ABC, abstractmethod


class BaseEmbeddingAdapter(ABC):
    """
    Abstract base class for all embedding providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier ('local', 'openai', 'mock')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector embedding dimensionality."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document chunks."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single search query."""
        pass
