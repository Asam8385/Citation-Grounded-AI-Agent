"""Build a structured FAQ dataset from the Cedarbridge training PDFs.

Run from the project root:
    uv run python scripts/faq_dataset_builder.py

This parser expects the layout used in the supplied Cedarbridge FAQ PDFs:
- the first four text lines contain the document header;
- each FAQ begins with an identifier such as ``LEAVE-001 | SICK LEAVE``;
- the question ends with a question mark and is followed by its answer.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "faqs" / "pdfs"
OUTPUT_FILE = PROJECT_ROOT / "data" / "faqs" / "sample_faqs.json"

FAQ_HEADING = re.compile(r"^(?P<faq_id>[A-Z]+-\d{3})\s*\|\s*(?P<section>.+)$")
DOCUMENT_HEADING = re.compile(r"^(?P<document_id>CB-DOC-\d{3})\s*/\s*Page\s+\d+$")
VERSION_HEADING = re.compile(
    r"Version\s+(?P<version>[\d.]+)\s*\|\s*Effective\s+"
    r"(?P<effective_date>\d{2}\s+[A-Za-z]{3}\s+\d{4})"
)


def join_lines(lines: list[str]) -> str:
    """Join PDF-wrapped lines into searchable paragraph text."""
    return " ".join(line.strip() for line in lines if line.strip())


def parse_faqs_on_page(lines: list[str], page_number: int) -> list[dict[str, Any]]:
    """Extract complete Q&A pairs from one page of our standard FAQ layout."""
    starts = [index for index, line in enumerate(lines) if FAQ_HEADING.fullmatch(line)]
    if not starts:
        raise ValueError(f"No FAQ headings found on page {page_number}")

    records: list[dict[str, Any]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        heading = FAQ_HEADING.fullmatch(lines[start])
        assert heading is not None  # Already checked when collecting starts.

        content = [line.strip() for line in lines[start + 1 : end] if line.strip()]
        question_end = next((i for i, line in enumerate(content) if "?" in line), None)
        if question_end is None or question_end == len(content) - 1:
            raise ValueError(f"Invalid question or answer for {heading['faq_id']}")

        records.append(
            {
                "faq_id": heading["faq_id"],
                "section": heading["section"].replace("_", " ").title(),
                "question": join_lines(content[: question_end + 1]),
                "answer": join_lines(content[question_end + 1 :]),
                "source_page": page_number,
            }
        )
    return records


def extract_pdf(pdf_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one PDF and return its metadata and parsed FAQs."""
    with pymupdf.open(pdf_path) as pdf:
        document: dict[str, Any] | None = None
        all_faqs: list[dict[str, Any]] = []

        for page_number, page in enumerate(pdf, start=1):
            lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
            if len(lines) < 5:
                raise ValueError(f"Page {page_number} has too little text: {pdf_path.name}")

            title = lines[1]
            version_match = VERSION_HEADING.search(lines[2])
            document_match = DOCUMENT_HEADING.fullmatch(lines[3])
            if version_match is None or document_match is None:
                raise ValueError(f"Unrecognized PDF header: {pdf_path.name}, page {page_number}")

            document_id = document_match["document_id"]
            version = version_match["version"]
            effective_date = datetime.strptime(
                version_match["effective_date"], "%d %b %Y"
            ).date().isoformat()
            source_path = pdf_path.relative_to(OUTPUT_FILE.parent).as_posix()
            category = title.removesuffix(" - FAQ")

            if document is None:
                document = {
                    "document_id": document_id,
                    "title": title,
                    "category": category,
                    "source_path": source_path,
                    "source_version": version,
                    "effective_date": effective_date,
                    "page_count": len(pdf),
                    "faq_count": 0,
                }
            elif (document_id, version, effective_date) != (
                document["document_id"],
                document["source_version"],
                document["effective_date"],
            ):
                raise ValueError(f"Inconsistent PDF header: {pdf_path.name}, page {page_number}")

            page_faqs = parse_faqs_on_page(lines, page_number)
            for faq in page_faqs:
                faq.update(
                    {
                        "document_id": document_id,
                        "category": category,
                        "source_path": source_path,
                        "source_version": version,
                        "effective_date": effective_date,
                    }
                )
            all_faqs.extend(page_faqs)

        if document is None:
            raise ValueError(f"PDF contains no pages: {pdf_path.name}")
        document["faq_count"] = len(all_faqs)
        return document, all_faqs


def build_dataset(pdf_directory: Path = PDF_DIR) -> dict[str, Any]:
    """Process all FAQs, rejecting duplicate identifiers and malformed files."""
    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    documents: list[dict[str, Any]] = []
    faqs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_document_ids: set[str] = set()

    for pdf_path in pdf_files:
        document, extracted_faqs = extract_pdf(pdf_path)
        if document["document_id"] in seen_document_ids:
            raise ValueError(f"Duplicate document ID: {document['document_id']}")
        seen_document_ids.add(document["document_id"])

        for faq in extracted_faqs:
            if faq["faq_id"] in seen_ids:
                raise ValueError(f"Duplicate FAQ ID: {faq['faq_id']}")
            seen_ids.add(faq["faq_id"])
            faqs.append(faq)
        documents.append(document)

    return {
        "schema_version": "1.0",
        "dataset_id": "cedarbridge_faq_knowledge_base_v1",
        "company": {
            "name": "Cedarbridge Systems",
            "fictional": True,
            "description": "Synthetic FAQ knowledge base for citation-grounded AI testing.",
        },
        "dataset_notes": [
            "Sample policies and people are fictional.",
            "FAQ questions and answers are extracted from the supplied PDFs.",
            "source_path is relative to the JSON file location.",
            "source_page is the physical 1-based PDF page number.",
        ],
        "documents": documents,
        "faqs": faqs,
    }


def save_dataset(dataset: dict[str, Any], output_path: Path = OUTPUT_FILE) -> None:
    """Write human-readable UTF-8 JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    dataset = build_dataset()
    save_dataset(dataset)
    print(f"Documents processed: {len(dataset['documents'])}")
    print(f"FAQs extracted: {len(dataset['faqs'])}")
    print(f"Saved JSON to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
