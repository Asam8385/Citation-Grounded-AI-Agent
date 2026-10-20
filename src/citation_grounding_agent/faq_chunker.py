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

def build_faq_chunks(dataset: FAQDataset)
