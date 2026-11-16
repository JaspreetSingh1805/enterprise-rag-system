from typing import Optional

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.core.exceptions import ConfigurationError, KnowledgeBaseNotFoundError
from app.core.logging import get_logger
from app.models.pipeline_config import PipelineConfig
from app.storage.metadata_store import MetadataStore

logger = get_logger("config_manager")


class DynamicConfigManager:
    """
    Manages dynamic pipeline configuration versioning, real-time activation without restarts,
    and safe re-indexing on embedding model modifications.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        vector_adapter: BaseVectorStoreAdapter,
    ):
        self.store = metadata_store
        self.vector_adapter = vector_adapter

    def get_active_config(self) -> PipelineConfig:
        return self.store.get_active_config()

    def list_configs(self) -> list[PipelineConfig]:
        return self.store.list_configs()

    def get_config(self, version: str) -> Optional[PipelineConfig]:
        return self.store.get_config(version)

    def save_new_config(self, config: PipelineConfig) -> PipelineConfig:
        existing = self.store.get_config(config.version)
        if existing:
            raise ConfigurationError(f"Config version '{config.version}' already exists")
        return self.store.save_config(config)

    def activate_config(self, version: str) -> PipelineConfig:
        logger.info(f"Dynamically activating pipeline configuration version '{version}'")
        return self.store.set_active_config(version)

    async def safe_reindex_knowledge_base(
        self,
        kb_id: str,
        new_embedding_adapter: BaseEmbeddingAdapter,
    ) -> int:
        """
        Safely re-index an existing Knowledge Base when the embedding model or dimension changes.
        Re-embeds all existing chunks and updates vector collection with zero data loss.
        """
        kb = self.store.get_kb(kb_id)
        if not kb:
            raise KnowledgeBaseNotFoundError(kb_id)

        logger.info(
            f"Initiating safe re-index for KB '{kb_id}': "
            f"from {kb.embedding_model} ({kb.embedding_dimension}d) to {new_embedding_adapter.model_name} ({new_embedding_adapter.dimension}d)"
        )

        chunks = self.store.get_all_chunks_for_kb(kb_id)
        if not chunks:
            logger.info(f"No chunks found for KB '{kb_id}' to re-index")
            self.store.update_kb_embedding(
                kb_id, new_embedding_adapter.model_name, new_embedding_adapter.dimension
            )
            return 0

        # 1. Delete existing vector collection
        await self.vector_adapter.delete_collection(kb_id)

        # 2. Re-create collection with new dimension
        await self.vector_adapter.create_collection(kb_id, new_embedding_adapter.dimension)

        # 3. Generate new embeddings for all chunks
        texts = [c.text for c in chunks]
        new_embeddings = await new_embedding_adapter.embed_documents(texts)
        for c, emb in zip(chunks, new_embeddings):
            c.embedding = emb

        # 4. Upsert into new vector store
        reindexed_count = await self.vector_adapter.upsert_chunks(kb_id, chunks)

        # 5. Update KB embedding metadata
        self.store.update_kb_embedding(
            kb_id, new_embedding_adapter.model_name, new_embedding_adapter.dimension
        )

        logger.info(f"Safe re-index completed for KB '{kb_id}': {reindexed_count} chunks updated")
        return reindexed_count
