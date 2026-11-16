import io

import docx

from app.core.exceptions import EmptyFileError, MalformedFileError
from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument, SectionContent


class DOCXParser(BaseParser):
    """
    Parser for Microsoft Word (.docx) files with heading/section extraction.
    """

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise EmptyFileError(filename)

        # DOCX is a zip container starting with PK\x03\x04
        if not file_bytes.startswith(b"PK\x03\x04"):
            raise MalformedFileError(
                filename, "File header does not match valid DOCX archive (PK magic missing)"
            )

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise MalformedFileError(filename, f"Corrupted DOCX document: {str(e)}")

        sections: list[SectionContent] = []
        current_title = "Overview"
        current_paragraphs: list[str] = []
        current_level = 1

        all_text_parts: list[str] = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            style_name = p.style.name.lower() if p.style and p.style.name else ""
            if "heading" in style_name:
                # Save previous section if it has text
                if current_paragraphs:
                    sections.append(
                        SectionContent(
                            title=current_title,
                            text="\n".join(current_paragraphs),
                            level=current_level,
                        )
                    )
                    current_paragraphs = []
                current_title = text
                # Extract level if available e.g. "Heading 1"
                try:
                    current_level = int(style_name.replace("heading", "").strip())
                except ValueError:
                    current_level = 1
            else:
                current_paragraphs.append(text)
                all_text_parts.append(text)

        # Flush final section
        if current_paragraphs:
            sections.append(
                SectionContent(
                    title=current_title,
                    text="\n".join(current_paragraphs),
                    level=current_level,
                )
            )

        full_text = "\n\n".join(all_text_parts).strip()
        if not full_text:
            raise EmptyFileError(filename)

        return ParsedDocument(
            filename=filename,
            full_text=full_text,
            pages=[PageContent(page_number=1, text=full_text)],
            sections=sections,
            metadata={"section_count": len(sections), "format": "docx"},
        )
