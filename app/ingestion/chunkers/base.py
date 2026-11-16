from abc import ABC, abstractmethod
from typing import Any, Optional

from app.ingestion.parsers.base import ParsedDocument
from app.models.domain import Chunk


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Strategy identifier ('recursive', 'structure')."""
        pass

    @abstractmethod
    def chunk(
        self,
        parsed_doc: ParsedDocument,
        kb_id: str,
        doc_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        custom_fields: Optional[dict[str, Any]] = None,
    ) -> list[Chunk]:
        """Split parsed document into indexed chunks with metadata."""
        pass
