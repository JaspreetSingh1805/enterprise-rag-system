import io

from pypdf import PdfReader

from app.core.exceptions import EmptyFileError, MalformedFileError
from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument


class PDFParser(BaseParser):
    """
    Parser for PDF documents extracting text with exact page numbers.
    """

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise EmptyFileError(filename)

        if not file_bytes.startswith(b"%PDF"):
            raise MalformedFileError(
                filename, "File header does not match PDF specification (%PDF- missing)"
            )

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise MalformedFileError(filename, f"Failed to initialize PDF reader: {str(e)}")

        if len(reader.pages) == 0:
            raise EmptyFileError(filename)

        pages: list[PageContent] = []
        full_text_parts = []

        try:
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                clean_page_text = page_text.strip()
                pages.append(PageContent(page_number=idx + 1, text=clean_page_text))
                if clean_page_text:
                    full_text_parts.append(clean_page_text)
        except Exception as e:
            raise MalformedFileError(filename, f"Error extracting PDF text: {str(e)}")

        full_text = "\n\n".join(full_text_parts).strip()
        if not full_text:
            raise EmptyFileError(filename)

        return ParsedDocument(
            filename=filename,
            full_text=full_text,
            pages=pages,
            sections=[],
            metadata={"page_count": len(pages), "format": "pdf"},
        )
