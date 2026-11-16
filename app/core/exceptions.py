"""
Domain exceptions for the RAG platform.
"""


class RAGException(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class KnowledgeBaseNotFoundError(RAGException):
    """Knowledge base does not exist."""

    def __init__(self, kb_id: str):
        super().__init__(
            f"Knowledge base '{kb_id}' not found", status_code=404, details={"kb_id": kb_id}
        )


class KnowledgeBaseAlreadyExistsError(RAGException):
    """Knowledge base already exists."""

    def __init__(self, kb_id: str):
        super().__init__(
            f"Knowledge base '{kb_id}' already exists", status_code=409, details={"kb_id": kb_id}
        )


class DocumentNotFoundError(RAGException):
    """Document does not exist."""

    def __init__(self, doc_id: str):
        super().__init__(
            f"Document '{doc_id}' not found", status_code=404, details={"doc_id": doc_id}
        )


class EmptyFileError(RAGException):
    """Uploaded file is empty (0 bytes or no extractable text)."""

    def __init__(self, filename: str):
        super().__init__(
            f"File '{filename}' is empty or contains no extractable text",
            status_code=400,
            details={"filename": filename},
        )


class MalformedFileError(RAGException):
    """Uploaded file has an invalid format or corrupted header."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"File '{filename}' is malformed: {reason}",
            status_code=400,
            details={"filename": filename, "reason": reason},
        )


class DuplicateFileError(RAGException):
    """File with identical content hash already exists in this knowledge base."""

    def __init__(self, filename: str, content_hash: str):
        super().__init__(
            f"File '{filename}' is a duplicate (SHA-256: {content_hash[:12]}...)",
            status_code=409,
            details={"filename": filename, "content_hash": content_hash},
        )


class ConfigurationError(RAGException):
    """Invalid pipeline configuration or version conflict."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, details=details)


class JobNotFoundError(RAGException):
    """Background ingestion job not found."""

    def __init__(self, job_id: str):
        super().__init__(f"Job '{job_id}' not found", status_code=404, details={"job_id": job_id})


class EmbeddingModelMismatchError(RAGException):
    """Embedding model dimension or name mismatch detected for knowledge base."""

    def __init__(self, expected_model: str, expected_dim: int, actual_model: str, actual_dim: int):
        super().__init__(
            f"Embedding model mismatch: collection expects {expected_model} (dim {expected_dim}), "
            f"but current config uses {actual_model} (dim {actual_dim}). Safe re-index required.",
            status_code=400,
            details={
                "expected_model": expected_model,
                "expected_dim": expected_dim,
                "actual_model": actual_model,
                "actual_dim": actual_dim,
                "requires_reindex": True,
            },
        )
