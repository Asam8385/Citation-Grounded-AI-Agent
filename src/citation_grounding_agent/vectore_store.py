from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from citation_grounding_agent.document_adapter import load_evidence_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "faqs" / "sample_faqs.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "faqs" / "faq_chunks.json"
DEFAULT_DB = PROJECT_ROOT / "data" / "chroma_embeddinggemma" 
DEFAULT_COLLECTION = "cedarbridge"
DEFAULT_MODEL = "google/embeddinggemma-300m"


def create_embeddings(*, device: str = "cuda") -> HuggingFaceEmbeddings:
    """Load EmbeddingGemma with its retrieval-specific query/document prompts."""
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "prompt_name": "document",
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "prompt_name": "query",
            "normalize_embeddings": True,
        },
    )


def open_vector_sotre(
    persist_directory: Path = DEFAULT_DB,
    *,
    device: str = "cuda",
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:

    persist_directory.mkdir(parents=True , exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=create_embeddings(device=device)
    )




