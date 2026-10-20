"""Pydantic data contracts for the Cedarbridge FAQ knowledge base.

Sprint 1 uses these models to validate extracted PDF metadata, FAQ records,
and (later) searchable evidence chunks. No retrieval or LLM code belongs here.
"""

from datetime import date
from pathlib import PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_nonblank(value: str) -> str:
    """Reject empty or whitespace-only text without changing source content."""
    if not value.strip():
        raise ValueError("must not be empty or whitespace-only")
    return value


def _require_relative_path(value: str) -> str:
    """Require a safe path relative to the dataset JSON file's directory."""
    _require_nonblank(value)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("must be a relative POSIX path without '..' or backslashes")
    return value


class FAQBaseModel(BaseModel):
    """Shared behavior for all Sprint 1 data models."""

    model_config = ConfigDict(extra="forbid")


class CompanyInfo(FAQBaseModel):
    """Company metadata stored at the dataset level."""

    name: str = Field(min_length=1)
    fictional: bool
    description: str = Field(min_length=1)

    @field_validator("name", "description")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _require_nonblank(value)


class SourceDocument(FAQBaseModel):
    """Metadata for one source PDF document."""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    effective_date: date
    page_count: int = Field(ge=1, strict=True)
    faq_count: int = Field(ge=0, strict=True)

    @field_validator("document_id", "title", "category", "source_version")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _require_nonblank(value)

    @field_validator("source_path")
    @classmethod
    def check_source_path(cls, value: str) -> str:
        return _require_relative_path(value)


class FAQRecord(FAQBaseModel):
    """One Q&A pair extracted from the source PDF."""

    faq_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    section: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_page: int = Field(ge=1, strict=True)
    source_version: str = Field(min_length=1)
    effective_date: date

    @field_validator(
        "faq_id", "document_id", "category", "section",
        "question", "answer", "source_version"
    )
    @classmethod
    def check_text(cls, value: str) -> str:
        # Important: never strip or rewrite the original question/answer.
        return _require_nonblank(value)

    @field_validator("source_path")
    @classmethod
    def check_source_path(cls, value: str) -> str:
        return _require_relative_path(value)


class FAQDataset(FAQBaseModel):
    """Validated source documents and FAQ records from sample_faqs.json."""

    schema_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    company: CompanyInfo
    dataset_notes: list[str] = Field(default_factory=list)
    documents: list[SourceDocument] = Field(min_length=1)
    faqs: list[FAQRecord] = Field(min_length=1)

    @field_validator("schema_version", "dataset_id")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _require_nonblank(value)

    @model_validator(mode="after")
    def check_references(self) -> Self:
        """Validate references and consistency across the complete dataset."""
        documents_by_id = {doc.document_id: doc for doc in self.documents}
        if len(documents_by_id) != len(self.documents):
            raise ValueError("duplicate document_id found")

        faq_ids = [faq.faq_id for faq in self.faqs]
        if len(faq_ids) != len(set(faq_ids)):
            raise ValueError("duplicate faq_id found")

        actual_counts = {doc.document_id: 0 for doc in self.documents}

        for faq in self.faqs:
            document = documents_by_id.get(faq.document_id)
            if document is None:
                raise ValueError(
                    f"FAQ {faq.faq_id} references unknown document {faq.document_id}"
                )

            if faq.source_page > document.page_count:
                raise ValueError(
                    f"FAQ {faq.faq_id} refers to page {faq.source_page}, "
                    f"but document has only {document.page_count} pages"
                )

            if (
                faq.source_path != document.source_path
                or faq.source_version != document.source_version
                or faq.effective_date != document.effective_date
                or faq.category != document.category
            ):
                raise ValueError(
                    f"FAQ {faq.faq_id} source metadata does not match "
                    f"document {document.document_id}"
                )

            actual_counts[document.document_id] += 1

        for document in self.documents:
            if actual_counts[document.document_id] != document.faq_count:
                raise ValueError(
                    f"Document {document.document_id} declares "
                    f"{document.faq_count} FAQs but contains "
                    f"{actual_counts[document.document_id]}"
                )

        return self


class EvidenceChunk(FAQBaseModel):
    """One traceable, searchable FAQ chunk produced in the next sprint step."""

    chunk_id: str = Field(min_length=1)
    faq_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    section: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    search_text: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_page: int = Field(ge=1, strict=True)
    source_version: str = Field(min_length=1)
    effective_date: date

    @field_validator(
        "chunk_id", "faq_id", "document_id", "category", "section",
        "question", "answer", "search_text", "source_version"
    )
    @classmethod
    def check_text(cls, value: str) -> str:
        return _require_nonblank(value)

    @field_validator("source_path")
    @classmethod
    def check_source_path(cls, value: str) -> str:
        return _require_relative_path(value)
