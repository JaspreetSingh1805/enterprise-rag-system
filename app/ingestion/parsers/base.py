from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    page_number: int
    text: str


class SectionContent(BaseModel):
    title: str
    text: str
    level: int = 1


class ParsedDocument(BaseModel):
    filename: str
    full_text: str
    pages: list[PageContent] = Field(default_factory=list)
    sections: list[SectionContent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseParser(ABC):
    """
    Base parser interface for file types.
    """

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse raw file bytes into structured ParsedDocument."""
        pass
