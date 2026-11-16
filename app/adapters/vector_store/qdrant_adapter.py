import uuid
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.core.logging import get_logger
from app.models.domain import Chunk

logger = get_logger("qdrant_vector_store")


class QdrantVectorStoreAdapter(BaseVectorStoreAdapter):
    """
    Qdrant Vector Store Adapter.
    Supports in-memory, local embedded disk, or networked Docker container.
    """

    def __init__(
        self,
        host: Optional[str] = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        in_memory: bool = False,
    ):
        if in_memory:
            logger.info("Initializing In-Memory Qdrant client")
            self.client = QdrantClient(":memory:")
        elif path:
            logger.info(f"Initializing Embedded Qdrant client at {path}")
            self.client = QdrantClient(path=path)
        else:
            logger.info(f"Connecting to Qdrant server at {host}:{port}")
            self.client = QdrantClient(host=host, port=port, api_key=api_key, timeout=10.0)

    @property
    def store_type(self) -> str:
        return "qdrant"

    def _collection_name(self, kb_id: str) -> str:
        # Qdrant collection name safe slug
        return f"kb_{kb_id}".replace("-", "_")

    async def create_collection(self, kb_id: str, dimension: int) -> bool:
        col_name = self._collection_name(kb_id)
        collections = self.client.get_collections().collections
        exists = any(c.name == col_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=col_name,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection {col_name} with dimension {dimension}")
        return True

    async def upsert_chunks(self, kb_id: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        col_name = self._collection_name(kb_id)
        dim = len(chunks[0].embedding) if chunks[0].embedding else 384
        await self.create_collection(kb_id, dim)

        points = []
        for chunk in chunks:
            if not chunk.embedding:
                continue

            # Convert string ID to valid Qdrant point ID (UUID or int)
            try:
                point_id = str(uuid.UUID(chunk.id))
            except Exception:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))

            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "knowledge_base_id": chunk.knowledge_base_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata,
            }

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=col_name, points=points)
        return len(points)

    async def search(
        self,
        kb_id: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        col_name = self._collection_name(kb_id)
        collections = self.client.get_collections().collections
        if not any(c.name == col_name for c in collections):
            return []

        qfilter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                if k == "document_id":
                    conditions.append(
                        qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=v))
                    )
                else:
                    conditions.append(
                        qmodels.FieldCondition(
                            key=f"metadata.{k}", match=qmodels.MatchValue(value=v)
                        )
                    )
            if conditions:
                qfilter = qmodels.Filter(must=conditions)

        if hasattr(self.client, "query_points"):
            results_obj = self.client.query_points(
                collection_name=col_name,
                query=query_vector,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
            points = results_obj.points
        else:
            points = self.client.search(
                collection_name=col_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )

        output: list[tuple[Chunk, float]] = []
        for r in points:
            payload = r.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id", str(r.id)),
                document_id=payload.get("document_id", ""),
                knowledge_base_id=payload.get("knowledge_base_id", kb_id),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", 0),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
                metadata=payload.get("metadata", {}),
            )
            # Normalize cosine score
            score = max(0.0, min(1.0, float(r.score)))
            output.append((chunk, score))

        return output

    async def delete_by_document(self, kb_id: str, document_id: str) -> int:
        col_name = self._collection_name(kb_id)
        collections = self.client.get_collections().collections
        if not any(c.name == col_name for c in collections):
            return 0

        # Count chunks for this doc first
        count_res = self.client.count(
            collection_name=col_name,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id", match=qmodels.MatchValue(value=document_id)
                    )
                ]
            ),
        )
        count = count_res.count

        if count > 0:
            self.client.delete(
                collection_name=col_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
        return count

    async def delete_collection(self, kb_id: str) -> bool:
        col_name = self._collection_name(kb_id)
        collections = self.client.get_collections().collections
        if any(c.name == col_name for c in collections):
            self.client.delete_collection(collection_name=col_name)
            return True
        return False

    async def count(self, kb_id: str) -> int:
        col_name = self._collection_name(kb_id)
        collections = self.client.get_collections().collections
        if not any(c.name == col_name for c in collections):
            return 0
        return self.client.count(collection_name=col_name).count

    async def get_collection_dimension(self, kb_id: str) -> Optional[int]:
        col_name = self._collection_name(kb_id)
        collections = self.client.get_collections().collections
        if not any(c.name == col_name for c in collections):
            return None
        info = self.client.get_collection(collection_name=col_name)
        params = info.config.params.vectors
        if isinstance(params, qmodels.VectorParams):
            return params.size
        return None
