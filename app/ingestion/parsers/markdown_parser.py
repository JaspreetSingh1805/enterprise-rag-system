import re

from app.core.exceptions import EmptyFileError
from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument, SectionContent


class MarkdownParser(BaseParser):
    """
    Parser for Markdown (.md) documents with header hierarchy extraction.
    """

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise EmptyFileError(filename)

        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception:
                text = file_bytes.decode("utf-8", errors="replace")

        clean_text = text.strip()
        if not clean_text:
            raise EmptyFileError(filename)

        sections: list[SectionContent] = []
        lines = clean_text.splitlines()

        current_title = "Introduction"
        current_level = 1
        current_lines: list[str] = []

        header_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_regex.match(line.strip())
            if match:
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append(
                            SectionContent(
                                title=current_title,
                                text=sec_text,
                                level=current_level,
                            )
                        )
                    current_lines = []
                current_level = len(match.group(1))
                current_title = match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                sections.append(
                    SectionContent(
                        title=current_title,
                        text=sec_text,
                        level=current_level,
                    )
                )

        return ParsedDocument(
            filename=filename,
            full_text=clean_text,
            pages=[PageContent(page_number=1, text=clean_text)],
            sections=sections,
            metadata={"section_count": len(sections), "format": "markdown"},
        )
