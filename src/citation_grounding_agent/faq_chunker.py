from citation_grounding_agent.models import FAQDataset , EvidenceChunk , FAQRecord

def generate_chunk_id(faq: FAQRecord) -> str:

   return f"{faq.document_id}:{faq.faq_id}:v{faq.source_version}"


def format_search_text(faq: FAQRecord) -> str:

    """Add FAQ context for search without modifying the original answer."""
    return (
        f"Category: {faq.category}\n"
        f"Section: {faq.section}\n\n"
        f"Question: {faq.question}\n\n"
        f"Answer: {faq.answer}"
    )

def build_faq_chunks(dataset: FAQDataset ) -> list[EvidenceChunk]:
    """Build one evidence chunk per FAQ, preserving its source reference."""
    chunks: list[EvidenceChunk] = []

    for faq in dataset.faqs:
        chunk = EvidenceChunk(
            chunk_id=generate_chunk_id(faq),
            faq_id=faq.faq_id,
            document_id=faq.document_id,
            category=faq.category,
            section=faq.section,
            question=faq.question,
            answer=faq.answer,
            search_text=format_search_text(faq),
            source_path=faq.source_path,
            source_page=faq.source_page,
            source_version=faq.source_version,
            effective_date=faq.effective_date,
        )
        chunks.append(chunk)

    return chunks