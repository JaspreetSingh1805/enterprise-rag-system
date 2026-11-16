import io

import docx
import pytest

from app.core.exceptions import (
    DuplicateFileError,
    EmptyFileError,
    MalformedFileError,
    RAGException,
)
from app.ingestion.deduplication import validate_file_bytes
from app.ingestion.parsers import (
    DOCXParser,
    MarkdownParser,
    PDFParser,
    TextParser,
)
from app.ingestion.pipeline import IngestionPipeline
from app.models.domain import KnowledgeBase
from app.models.pipeline_config import IngestionConfig, PipelineConfig


def test_markdown_parser():
    parser = MarkdownParser()
    md_text = "# Section 1\nContent of section 1.\n\n## Sub Section 1.1\nDetailed content."
    parsed = parser.parse(md_text.encode("utf-8"), "test.md")
    assert parsed.filename == "test.md"
    assert len(parsed.sections) == 2
    assert parsed.sections[0].title == "Section 1"
    assert parsed.sections[1].title == "Sub Section 1.1"


def test_text_parser():
    parser = TextParser()
    txt = "Paragraph one content.\n\nParagraph two content."
    parsed = parser.parse(txt.encode("utf-8"), "notes.txt")
    assert parsed.filename == "notes.txt"
    assert len(parsed.sections) == 2


def test_docx_parser():
    doc = docx.Document()
    doc.add_heading("Company Policy", level=1)
    doc.add_paragraph("All employees must follow guidelines.")
    buf = io.BytesIO()
    doc.save(buf)
    file_bytes = buf.getvalue()

    parser = DOCXParser()
    parsed = parser.parse(file_bytes, "policy.docx")
    assert parsed.filename == "policy.docx"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].title == "Company Policy"


def test_empty_file_rejected():
    with pytest.raises(EmptyFileError):
        validate_file_bytes(b"", "empty.txt")

    with pytest.raises(EmptyFileError):
        validate_file_bytes(b"   \n\t  ", "blank.txt")


def test_malformed_pdf_rejected():
    parser = PDFParser()
    with pytest.raises(MalformedFileError) as exc_info:
        parser.parse(b"NOT_A_PDF_STREAM_CONTENT", "corrupt.pdf")
    assert "PDF specification" in str(exc_info.value)


def test_malformed_docx_rejected():
    parser = DOCXParser()
    with pytest.raises(MalformedFileError) as exc_info:
        parser.parse(b"NOT_A_ZIP_CONTAINER", "corrupt.docx")
    assert "PK magic missing" in str(exc_info.value)


def test_oversized_file_rejected():
    large_bytes = b"x" * (1024 * 1024 + 10)  # ~1.01 MB
    with pytest.raises(RAGException) as exc_info:
        validate_file_bytes(large_bytes, "large.txt", max_size_mb=1)
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_duplicate_file_handling(metadata_store, mock_embedding, in_memory_vector_store):
    kb = KnowledgeBase(
        id="kb_dup_test",
        name="Dup Test KB",
        embedding_model="mock",
        embedding_dimension=384,
    )
    metadata_store.create_kb(kb)
    pipeline = IngestionPipeline(metadata_store, mock_embedding, in_memory_vector_store)

    file_content = b"Exact duplicate content for testing hash matching."

    # 1. Ingest first time
    doc1 = await pipeline.ingest_file("kb_dup_test", "doc1.txt", file_content)
    assert doc1.id is not None

    # 2. Ingest duplicate with mode 'reject'
    cfg_reject = PipelineConfig(
        version="v1.0",
        ingestion=IngestionConfig(deduplication_mode="reject"),
    )
    with pytest.raises(DuplicateFileError):
        await pipeline.ingest_file("kb_dup_test", "doc2.txt", file_content, config=cfg_reject)

    # 3. Ingest duplicate with mode 'skip'
    cfg_skip = PipelineConfig(
        version="v1.1",
        ingestion=IngestionConfig(deduplication_mode="skip"),
    )
    doc_skipped = await pipeline.ingest_file(
        "kb_dup_test", "doc2.txt", file_content, config=cfg_skip
    )
    assert doc_skipped.id == doc1.id

    # 4. Ingest duplicate with mode 'update'
    cfg_update = PipelineConfig(
        version="v1.2",
        ingestion=IngestionConfig(deduplication_mode="update"),
    )
    doc_updated = await pipeline.ingest_file(
        "kb_dup_test", "doc1.txt", file_content, config=cfg_update
    )
    assert doc_updated.id != doc1.id  # New ID replaced the old one
