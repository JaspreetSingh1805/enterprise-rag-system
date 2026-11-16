import pytest
from pydantic import ValidationError

from app.ingestion.chunkers import (
    RecursiveCharacterChunker,
    StructureAwareChunker,
    get_chunker,
)
from app.ingestion.parsers.base import PageContent, ParsedDocument, SectionContent
from app.models.domain import ChunkingStrategyType
from app.models.pipeline_config import ChunkingConfig


def test_recursive_chunker_basic():
    chunker = RecursiveCharacterChunker()
    text = "Sentence one. " * 50  # ~700 chars
    parsed_doc = ParsedDocument(
        filename="test.txt",
        full_text=text,
        pages=[PageContent(page_number=1, text=text)],
        sections=[],
    )

    chunks = chunker.chunk(parsed_doc, kb_id="kb1", doc_id="d1", chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 220  # Near target size
        assert c.page_number == 1
        assert c.metadata["strategy"] == "recursive"


def test_structure_aware_chunker():
    chunker = StructureAwareChunker()
    sections = [
        SectionContent(title="Introduction", text="This is the introduction.", level=1),
        SectionContent(title="Sick Leave", text="Sick leave rules and requirements.", level=2),
    ]
    parsed_doc = ParsedDocument(
        filename="policy.md",
        full_text="intro and sick leave",
        sections=sections,
    )

    chunks = chunker.chunk(parsed_doc, kb_id="kb1", doc_id="d2", chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].section_title == "Introduction"
    assert "[Introduction]" in chunks[0].text
    assert chunks[1].section_title == "Sick Leave"
    assert "[Sick Leave]" in chunks[1].text


def test_invalid_chunk_overlap_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
    assert "chunk_overlap" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        ChunkingConfig(chunk_size=100, chunk_overlap=150)
    assert "chunk_overlap" in str(exc_info.value)


def test_get_chunker_factory():
    c1 = get_chunker("recursive")
    assert isinstance(c1, RecursiveCharacterChunker)
    c2 = get_chunker("structure")
    assert isinstance(c2, StructureAwareChunker)
    c3 = get_chunker(ChunkingStrategyType.STRUCTURE)
    assert isinstance(c3, StructureAwareChunker)
