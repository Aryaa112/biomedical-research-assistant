from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PageContent:
    page_number: int
    text: str
    extraction_method: str
    quality_score: float
    needs_ocr: bool = False


@dataclass
class Document:
    document_id: str
    filename: str
    title: str
    authors: list[str]
    year: Optional[int]
    source_url: Optional[str]
    pages: list[PageContent]
    extraction_method: str
    status: str

    def to_dict(self):
        return asdict(self)