import hashlib

from app.core.exceptions import EmptyFileError, RAGException


def calculate_content_hash(file_bytes: bytes) -> str:
    """Calculate SHA-256 checksum of raw file bytes."""
    if not file_bytes:
        return ""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_file_bytes(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 50,
) -> str:
    """
    Validate basic file constraints:
    - Non-empty
    - Size within limit
    Returns calculated SHA-256 content hash.
    """
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise EmptyFileError(filename)

    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise RAGException(
            f"File '{filename}' ({len(file_bytes)} bytes) exceeds maximum allowed size of {max_size_mb} MB",
            status_code=413,
        )

    return calculate_content_hash(file_bytes)
