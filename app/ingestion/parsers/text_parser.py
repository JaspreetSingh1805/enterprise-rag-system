from app.core.exceptions import EmptyFileError
from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument, SectionContent


class TextParser(BaseParser):
    """
    Parser for plain text (.txt) documents.
    """

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise EmptyFileError(filename)

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        text = None
        for enc in encodings:
            try:
                text = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            text = file_bytes.decode("utf-8", errors="replace")

        clean_text = text.strip()
        if not clean_text:
            raise EmptyFileError(filename)

        # Break into paragraphs as virtual sections
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        sections: list[SectionContent] = []
        for idx, p in enumerate(paragraphs):
            sections.append(
                SectionContent(
                    title=f"Paragraph {idx + 1}",
                    text=p,
                    level=1,
                )
            )

        return ParsedDocument(
            filename=filename,
            full_text=clean_text,
            pages=[PageContent(page_number=1, text=clean_text)],
            sections=sections,
            metadata={"paragraph_count": len(paragraphs), "format": "txt"},
        )
