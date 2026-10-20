"""Build and export citation-ready FAQ chunks.

Run from the project root:
    uv run python -m citation_grounding_agent.cli

Optional:
    uv run python -m citation_grounding_agent.cli --preview
"""

import argparse
import json
from pathlib import Path

from citation_grounding_agent.faq_chunker import build_faq_chunks
from citation_grounding_agent.faq_loader import load_FAQDataset
from citation_grounding_agent.models import EvidenceChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "faqs" / "sample_faqs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "faqs" / "faq_chunks.json"


def save_chunks(
      chunks: list[EvidenceChunk],
      dataset_id: str,
      output_path: Path
) -> None:

    """Write validated evidence chunks to a readable JSON file."""
    payload = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "chunk_count": len(chunks),
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }


    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )



def main():

    dataset = load_FAQDataset(Path(DEFAULT_INPUT))

    chunks = build_faq_chunks(dataset)


    if len(chunks) != len(dataset.faqs):
        raise ValueError("Chunk count does not match FAQ count")

    save_chunks(chunks, dataset.dataset_id, DEFAULT_OUTPUT)


    print(f"Documents: {len(dataset.documents)}")
    print(f"FAQ records: {len(dataset.faqs)}")
    print(f"Evidence chunks: {len(chunks)}")


if __name__ == "__main__":

    main()


