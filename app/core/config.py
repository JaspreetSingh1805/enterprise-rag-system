from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central Application Settings.
    Loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Server & Environment
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    secret_key: str = "dev-insecure-secret-key-change-in-production"

    # Persistent Storage Paths
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/rag_metadata.db"

    # Vector Store
    vector_store_type: Literal["qdrant", "memory"] = "memory"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_path: Optional[str] = None

    # LLM Providers
    default_llm_provider: Literal["openai", "ollama", "mock"] = "mock"
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"

    # Embedding Providers
    default_embedding_provider: Literal["local", "openai", "mock"] = "mock"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Default Pipeline Parameters
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    default_chunking_strategy: Literal["recursive", "structure"] = "recursive"
    default_search_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    default_similarity_threshold: float = 0.35
    default_top_k: int = 5
    default_rerank_enabled: bool = False
    default_query_rewrite_enabled: bool = False


settings = Settings()
