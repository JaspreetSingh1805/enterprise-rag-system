from pathlib import Path

from app.core.exceptions import RAGException
from app.ingestion.parsers.base import BaseParser, ParsedDocument
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.text_parser import TextParser

_PARSER_MAP = {
    ".pdf": PDFParser,
    ".docx": DOCXParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".txt": TextParser,
}


def get_parser_for_file(filename: str) -> BaseParser:
    ext = Path(filename).suffix.lower()
    parser_cls = _PARSER_MAP.get(ext)
    if not parser_cls:
        raise RAGException(
            f"Unsupported file format '{ext}'. Supported formats: {list(_PARSER_MAP.keys())}",
            status_code=400,
        )
    return parser_cls()


__all__ = [
    "BaseParser",
    "ParsedDocument",
    "PDFParser",
    "DOCXParser",
    "MarkdownParser",
    "TextParser",
    "get_parser_for_file",
]
