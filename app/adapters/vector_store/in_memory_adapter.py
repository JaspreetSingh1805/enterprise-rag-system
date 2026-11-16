import math
from typing import Any, Optional

from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.models.domain import Chunk


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a, b in zip(v1, v2)))  # or sum(a * a for a in v1)
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    sim = dot / (norm1 * norm2)
    # Clamp to [0.0, 1.0] for cosine on unit/non-negative vectors or [-1.0, 1.0]
    return max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))


class InMemoryVectorStoreAdapter(BaseVectorStoreAdapter):
    """
    Lightweight In-Memory Vector Store for testing and isolated dev environments.
    """

    def __init__(self):
        # Structure: {kb_id: {"dimension": int, "chunks": {chunk_id: Chunk}}}
        self._collections: dict[str, dict[str, Any]] = {}

    @property
    def store_type(self) -> str:
        return "memory"

    async def create_collection(self, kb_id: str, dimension: int) -> bool:
        if kb_id not in self._collections:
            self._collections[kb_id] = {"dimension": dimension, "chunks": {}}
        else:
            self._collections[kb_id]["dimension"] = dimension
        return True

    async def upsert_chunks(self, kb_id: str, chunks: list[Chunk]) -> int:
        if kb_id not in self._collections:
            dim = len(chunks[0].embedding) if (chunks and chunks[0].embedding) else 384
            await self.create_collection(kb_id, dim)

        count = 0
        for chunk in chunks:
            self._collections[kb_id]["chunks"][chunk.id] = chunk
            count += 1
        return count

    async def search(
        self,
        kb_id: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        if kb_id not in self._collections:
            return []

        chunks_dict: dict[str, Chunk] = self._collections[kb_id]["chunks"]
        scored: list[tuple[Chunk, float]] = []

        for chunk in chunks_dict.values():
            if not chunk.embedding:
                continue

            # Apply metadata filters if provided
            if filters:
                match = True
                for k, v in filters.items():
                    if k == "document_id" and chunk.document_id != v:
                        match = False
                        break
                    elif k in chunk.metadata and chunk.metadata[k] != v:
                        match = False
                        break
                if not match:
                    continue

            score = cosine_similarity(query_vector, chunk.embedding)
            scored.append((chunk, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete_by_document(self, kb_id: str, document_id: str) -> int:
        if kb_id not in self._collections:
            return 0
        chunks_dict = self._collections[kb_id]["chunks"]
        to_delete = [cid for cid, c in chunks_dict.items() if c.document_id == document_id]
        for cid in to_delete:
            del chunks_dict[cid]
        return len(to_delete)

    async def delete_collection(self, kb_id: str) -> bool:
        if kb_id in self._collections:
            del self._collections[kb_id]
            return True
        return False

    async def count(self, kb_id: str) -> int:
        if kb_id not in self._collections:
            return 0
        return len(self._collections[kb_id]["chunks"])

    async def get_collection_dimension(self, kb_id: str) -> Optional[int]:
        if kb_id not in self._collections:
            return None
        return self._collections[kb_id]["dimension"]
