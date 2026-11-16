from abc import ABC, abstractmethod
from typing import Any, Optional

from app.models.domain import Chunk


class BaseVectorStoreAdapter(ABC):
    """
    Abstract interface for Vector Store operations.
    Enforces multi-tenant isolation per Knowledge Base.
    """

    @property
    @abstractmethod
    def store_type(self) -> str:
        """Store identifier ('qdrant', 'memory')."""
        pass

    @abstractmethod
    async def create_collection(self, kb_id: str, dimension: int) -> bool:
        """Create a collection/index for the knowledge base."""
        pass

    @abstractmethod
    async def upsert_chunks(self, kb_id: str, chunks: list[Chunk]) -> int:
        """Insert or replace vector chunks into the knowledge base collection."""
        pass

    @abstractmethod
    async def search(
        self,
        kb_id: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        """Search the collection and return top_k chunks paired with their similarity score (0.0 to 1.0)."""
        pass

    @abstractmethod
    async def delete_by_document(self, kb_id: str, document_id: str) -> int:
        """Delete all chunk vectors belonging to a document."""
        pass

    @abstractmethod
    async def delete_collection(self, kb_id: str) -> bool:
        """Delete the entire collection for a knowledge base."""
        pass

    @abstractmethod
    async def count(self, kb_id: str) -> int:
        """Return the number of vectors stored for this knowledge base."""
        pass

    @abstractmethod
    async def get_collection_dimension(self, kb_id: str) -> Optional[int]:
        """Return the vector dimensionality of the collection, or None if it does not exist."""
        pass
