from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.chunkers.recursive_chunker import RecursiveCharacterChunker
from app.ingestion.chunkers.structure_chunker import StructureAwareChunker
from app.models.domain import ChunkingStrategyType


def get_chunker(strategy: ChunkingStrategyType | str) -> BaseChunker:
    strat_str = (
        strategy.value if isinstance(strategy, ChunkingStrategyType) else str(strategy).lower()
    )
    if strat_str == "structure":
        return StructureAwareChunker()
    return RecursiveCharacterChunker()


__all__ = [
    "BaseChunker",
    "RecursiveCharacterChunker",
    "StructureAwareChunker",
    "get_chunker",
]
