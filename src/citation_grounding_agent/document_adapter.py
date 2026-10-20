

from pathlib import Path
from typing import Any , Self


from langchain_core.documents import Document
from pydantic import BaseModel , ConfigDict , Field , model_validator

from citation_grounding_agent.models import EvidenceChunk


class ChunkExport(BaseModel):
    """Validate the JSON structure written by the Sprint 1 CLI."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    dataset_id: str
    chunk_count: int = Field(ge=0)
    chunks: list[EvidenceChunk]

    @model_validator(mode="after")
    def check_chunks(self) -> Self:
        """Reject incomplete exports and duplicate chunk identifiers."""
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count does not match the number of chunks")
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk_id found")
        return self


def load_evidence_chunks(json_path: Path) -> list[EvidenceChunk]:

    exported = ChunkExport.model_validate_json(
        json_path.read_text(encoding="utf-8")
    )

    return exported.chunks




def to_langchain_document(chunk: EvidenceChunk) -> Document:
    """Preserve searchable text and original citation metadata."""
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.search_text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "faq_id": chunk.faq_id,
            "document_id": chunk.document_id,
            "category": chunk.category,
            "section": chunk.section,
            "question": chunk.question,
            "original_answer": chunk.answer,
            "source_path": chunk.source_path,
            "source_page": chunk.source_page,
            "source_version": chunk.source_version,
            "effective_date": chunk.effective_date.isoformat(),
        },
    )



def load_langchain_documents(json_path: Path) -> list[Document]:

   chunks = load_evidence_chunks(json_path)

   return [to_langchain_document(chunk) for chunk in chunks]



