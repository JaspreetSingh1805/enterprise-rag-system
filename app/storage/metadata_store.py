import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from app.core.exceptions import (
    ConfigurationError,
    KnowledgeBaseAlreadyExistsError,
    KnowledgeBaseNotFoundError,
)
from app.core.logging import get_logger
from app.models.domain import (
    Chunk,
    Document,
    DocumentType,
    Job,
    JobStatus,
    KnowledgeBase,
)
from app.models.pipeline_config import PipelineConfig

logger = get_logger("metadata_store")


class MetadataStore:
    """
    Thread-safe SQLite persistent metadata store for Knowledge Bases, Documents,
    Raw Chunks, Dynamic Pipeline Configurations, and Background Jobs.
    """

    def __init__(self, db_path: str = "./data/rag_metadata.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    document_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    page_count INTEGER DEFAULT 1,
                    section_count INTEGER DEFAULT 1,
                    custom_fields_json TEXT DEFAULT '{}',
                    uploaded_at TEXT NOT NULL,
                    status TEXT DEFAULT 'indexed',
                    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    section_title TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS pipeline_configs (
                    version TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_percent INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    config_version TEXT NOT NULL,
                    details_json TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_docs_kb ON documents(knowledge_base_id);
                CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(knowledge_base_id, content_hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(knowledge_base_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_kb ON jobs(knowledge_base_id);
                """
            )

    # --------------------------------------------------------------------------
    # Knowledge Base Methods
    # --------------------------------------------------------------------------

    def create_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT id FROM knowledge_bases WHERE id = ?", (kb.id,))
            if cur.fetchone():
                raise KnowledgeBaseAlreadyExistsError(kb.id)

            conn.execute(
                """
                INSERT INTO knowledge_bases
                (id, name, description, embedding_model, embedding_dimension, document_count, chunk_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kb.id,
                    kb.name,
                    kb.description,
                    kb.embedding_model,
                    kb.embedding_dimension,
                    kb.document_count,
                    kb.chunk_count,
                    kb.created_at.isoformat(),
                    kb.updated_at.isoformat(),
                ),
            )
        return kb

    def get_kb(self, kb_id: str) -> Optional[KnowledgeBase]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,))
            row = cur.fetchone()
            if not row:
                return None
            return KnowledgeBase(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                embedding_model=row["embedding_model"],
                embedding_dimension=row["embedding_dimension"],
                document_count=row["document_count"],
                chunk_count=row["chunk_count"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_kbs(self) -> list[KnowledgeBase]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC")
            rows = cur.fetchall()
            return [
                KnowledgeBase(
                    id=r["id"],
                    name=r["name"],
                    description=r["description"],
                    embedding_model=r["embedding_model"],
                    embedding_dimension=r["embedding_dimension"],
                    document_count=r["document_count"],
                    chunk_count=r["chunk_count"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
                for r in rows
            ]

    def update_kb(self, kb_id: str, name: Optional[str] = None, description: Optional[str] = None) -> KnowledgeBase:
        kb = self.get_kb(kb_id)
        if not kb:
            raise KnowledgeBaseNotFoundError(kb_id)

        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            if name is not None:
                conn.execute("UPDATE knowledge_bases SET name = ?, updated_at = ? WHERE id = ?", (name, now, kb_id))
            if description is not None:
                conn.execute("UPDATE knowledge_bases SET description = ?, updated_at = ? WHERE id = ?", (description, now, kb_id))

        return self.get_kb(kb_id)  # type: ignore

    def update_kb_embedding(self, kb_id: str, embedding_model: str, embedding_dimension: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "UPDATE knowledge_bases SET embedding_model = ?, embedding_dimension = ?, updated_at = ? WHERE id = ?",
                (embedding_model, embedding_dimension, now, kb_id),
            )

    def delete_kb(self, kb_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE knowledge_base_id = ?", (kb_id,))
            conn.execute("DELETE FROM documents WHERE knowledge_base_id = ?", (kb_id,))
            conn.execute("DELETE FROM jobs WHERE knowledge_base_id = ?", (kb_id,))
            cur = conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
            return cur.rowcount > 0

    def refresh_kb_counts(self, kb_id: str) -> None:
        with self._lock, self._get_connection() as conn:
            doc_cnt = conn.execute("SELECT COUNT(*) FROM documents WHERE knowledge_base_id = ?", (kb_id,)).fetchone()[0]
            chunk_cnt = conn.execute("SELECT COUNT(*) FROM chunks WHERE knowledge_base_id = ?", (kb_id,)).fetchone()[0]
            conn.execute(
                "UPDATE knowledge_bases SET document_count = ?, chunk_count = ? WHERE id = ?",
                (doc_cnt, chunk_cnt, kb_id),
            )

    # --------------------------------------------------------------------------
    # Document Methods
    # --------------------------------------------------------------------------

    def create_document(self, doc: Document) -> Document:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents
                (id, knowledge_base_id, filename, content_type, content_hash, file_size_bytes,
                 page_count, section_count, custom_fields_json, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.knowledge_base_id,
                    doc.filename,
                    doc.content_type.value,
                    doc.content_hash,
                    doc.file_size_bytes,
                    doc.page_count,
                    doc.section_count,
                    json.dumps(doc.custom_fields),
                    doc.uploaded_at.isoformat(),
                    doc.status,
                ),
            )
        self.refresh_kb_counts(doc.knowledge_base_id)
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                knowledge_base_id=row["knowledge_base_id"],
                filename=row["filename"],
                content_type=DocumentType(row["content_type"]),
                content_hash=row["content_hash"],
                file_size_bytes=row["file_size_bytes"],
                page_count=row["page_count"],
                section_count=row["section_count"],
                custom_fields=json.loads(row["custom_fields_json"]),
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                status=row["status"],
            )

    def get_document_by_hash(self, kb_id: str, content_hash: str) -> Optional[Document]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM documents WHERE knowledge_base_id = ? AND content_hash = ?",
                (kb_id, content_hash),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                knowledge_base_id=row["knowledge_base_id"],
                filename=row["filename"],
                content_type=DocumentType(row["content_type"]),
                content_hash=row["content_hash"],
                file_size_bytes=row["file_size_bytes"],
                page_count=row["page_count"],
                section_count=row["section_count"],
                custom_fields=json.loads(row["custom_fields_json"]),
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                status=row["status"],
            )

    def list_documents(self, kb_id: str) -> list[Document]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM documents WHERE knowledge_base_id = ? ORDER BY uploaded_at DESC",
                (kb_id,),
            )
            rows = cur.fetchall()
            return [
                Document(
                    id=r["id"],
                    knowledge_base_id=r["knowledge_base_id"],
                    filename=r["filename"],
                    content_type=DocumentType(r["content_type"]),
                    content_hash=r["content_hash"],
                    file_size_bytes=r["file_size_bytes"],
                    page_count=r["page_count"],
                    section_count=r["section_count"],
                    custom_fields=json.loads(r["custom_fields_json"]),
                    uploaded_at=datetime.fromisoformat(r["uploaded_at"]),
                    status=r["status"],
                )
                for r in rows
            ]

    def delete_document(self, doc_id: str) -> bool:
        doc = self.get_document(doc_id)
        if not doc:
            return False
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cur.rowcount > 0
        if deleted:
            self.refresh_kb_counts(doc.knowledge_base_id)
        return deleted

    # --------------------------------------------------------------------------
    # Chunks Methods
    # --------------------------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        with self._lock, self._get_connection() as conn:
            for c in chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (id, document_id, knowledge_base_id, chunk_index, text, page_number, section_title, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        c.id,
                        c.document_id,
                        c.knowledge_base_id,
                        c.chunk_index,
                        c.text,
                        c.page_number,
                        c.section_title,
                        json.dumps(c.metadata),
                    ),
                )
        if chunks:
            self.refresh_kb_counts(chunks[0].knowledge_base_id)
        return len(chunks)

    def get_chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC", (doc_id,))
            rows = cur.fetchall()
            return [
                Chunk(
                    id=r["id"],
                    document_id=r["document_id"],
                    knowledge_base_id=r["knowledge_base_id"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    page_number=r["page_number"],
                    section_title=r["section_title"],
                    metadata=json.loads(r["metadata_json"]),
                )
                for r in rows
            ]

    def get_all_chunks_for_kb(self, kb_id: str) -> list[Chunk]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE knowledge_base_id = ? ORDER BY chunk_index ASC", (kb_id,))
            rows = cur.fetchall()
            return [
                Chunk(
                    id=r["id"],
                    document_id=r["document_id"],
                    knowledge_base_id=r["knowledge_base_id"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    page_number=r["page_number"],
                    section_title=r["section_title"],
                    metadata=json.loads(r["metadata_json"]),
                )
                for r in rows
            ]

    # --------------------------------------------------------------------------
    # Pipeline Configs Methods
    # --------------------------------------------------------------------------

    def save_config(self, config: PipelineConfig) -> PipelineConfig:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_configs
                (version, description, is_active, created_at, updated_at, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    config.version,
                    config.description,
                    1 if config.is_active else 0,
                    config.created_at.isoformat(),
                    config.updated_at.isoformat(),
                    config.model_dump_json(),
                ),
            )
        return config

    def get_config(self, version: str) -> Optional[PipelineConfig]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT config_json FROM pipeline_configs WHERE version = ?", (version,))
            row = cur.fetchone()
            if not row:
                return None
            return PipelineConfig.model_validate_json(row["config_json"])

    def list_configs(self) -> list[PipelineConfig]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT config_json FROM pipeline_configs ORDER BY created_at DESC")
            rows = cur.fetchall()
            return [PipelineConfig.model_validate_json(r["config_json"]) for r in rows]

    def set_active_config(self, version: str) -> PipelineConfig:
        cfg = self.get_config(version)
        if not cfg:
            raise ConfigurationError(f"Config version '{version}' not found")

        with self._lock, self._get_connection() as conn:
            # Set all to inactive
            conn.execute("UPDATE pipeline_configs SET is_active = 0")
            # Set target to active
            conn.execute("UPDATE pipeline_configs SET is_active = 1 WHERE version = ?", (version,))

        cfg.is_active = True
        self.save_config(cfg)
        return cfg

    def get_active_config(self) -> PipelineConfig:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT config_json FROM pipeline_configs WHERE is_active = 1")
            row = cur.fetchone()
            if row:
                return PipelineConfig.model_validate_json(row["config_json"])

            # Fallback to latest
            cur2 = conn.execute("SELECT config_json FROM pipeline_configs ORDER BY created_at DESC LIMIT 1")
            row2 = cur2.fetchone()
            if row2:
                cfg = PipelineConfig.model_validate_json(row2["config_json"])
                cfg.is_active = True
                return cfg

        # If completely empty, generate default v1
        v1 = PipelineConfig(
            version="v1",
            description="Default initial enterprise RAG pipeline configuration",
            is_active=True,
        )
        self.save_config(v1)
        return v1

    # --------------------------------------------------------------------------
    # Background Job Methods
    # --------------------------------------------------------------------------

    def create_job(self, job: Job) -> Job:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                (id, knowledge_base_id, job_type, status, progress_percent,
                 created_at, started_at, completed_at, error_message, config_version, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.knowledge_base_id,
                    job.job_type,
                    job.status.value,
                    job.progress_percent,
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.error_message,
                    job.config_version,
                    json.dumps(job.details),
                ),
            )
        return job

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_percent: Optional[int] = None,
        error_message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> Optional[Job]:
        job = self.get_job(job_id)
        if not job:
            return None

        now = datetime.now(timezone.utc)
        if status:
            job.status = status
            if status == JobStatus.PROCESSING and not job.started_at:
                job.started_at = now
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = now

        if progress_percent is not None:
            job.progress_percent = progress_percent
        if error_message is not None:
            job.error_message = error_message
        if details:
            job.details.update(details)

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    progress_percent = ?,
                    started_at = ?,
                    completed_at = ?,
                    error_message = ?,
                    details_json = ?
                WHERE id = ?
                """,
                (
                    job.status.value,
                    job.progress_percent,
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.error_message,
                    json.dumps(job.details),
                    job_id,
                ),
            )
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Job(
                id=row["id"],
                knowledge_base_id=row["knowledge_base_id"],
                job_type=row["job_type"],
                status=JobStatus(row["status"]),
                progress_percent=row["progress_percent"],
                created_at=datetime.fromisoformat(row["created_at"]),
                started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                error_message=row["error_message"],
                config_version=row["config_version"],
                details=json.loads(row["details_json"]),
            )

    def list_jobs(self, kb_id: Optional[str] = None) -> list[Job]:
        with self._lock, self._get_connection() as conn:
            if kb_id:
                cur = conn.execute("SELECT * FROM jobs WHERE knowledge_base_id = ? ORDER BY created_at DESC", (kb_id,))
            else:
                cur = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = cur.fetchall()
            return [
                Job(
                    id=r["id"],
                    knowledge_base_id=r["knowledge_base_id"],
                    job_type=r["job_type"],
                    status=JobStatus(r["status"]),
                    progress_percent=r["progress_percent"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    started_at=datetime.fromisoformat(r["started_at"]) if r["started_at"] else None,
                    completed_at=datetime.fromisoformat(r["completed_at"]) if r["completed_at"] else None,
                    error_message=r["error_message"],
                    config_version=r["config_version"],
                    details=json.loads(r["details_json"]),
                )
                for r in rows
            ]
