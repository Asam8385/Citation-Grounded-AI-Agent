from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from citation_grounding_agent.document_adapter import load_langchain_documents

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


def open_vector_store(
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




def index_documents(
    store: Chroma,
    documents: list[Document],
    *,
    rebuild: bool = False
        
) -> int:

    if not documents:
        raise ValueError("No documents were provided for indexing")

    ids: list[str] = []
    for document in documents:
        chunk_id = document.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every document needs a nonempty chunk_id")
        if document.id != chunk_id:
            raise ValueError(f"Document ID differs from chunk_id: {chunk_id}")
        ids.append(chunk_id)

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs were found")

    if rebuild:
        store.reset_collection()

    store.add_documents(documents=documents, ids=ids) 

    return len(ids)


def index_faq_chunks(
    chunks_path: Path = DEFAULT_CHUNKS,
    persist_directory: Path = DEFAULT_DB,
    *,
    device: str = "cpu",
    rebuild: bool = False,
) -> Chroma:
    """Load validated FAQ chunks and index them in Chroma."""
    documents = load_langchain_documents(chunks_path)
    store = open_vector_store(persist_directory, device=device)
    count = index_documents(store, documents, rebuild=rebuild)
    print(f"Indexed {count} FAQ chunks in {persist_directory}")
    return store
   

def semantic_search(store: Chroma, question: str, *, k: int = 3) -> list[Document]:
    """Retrieve source-bearing documents by semantic similarity."""
    if not question.strip():
        raise ValueError("The search question cannot be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    return store.similarity_search(question, k=k)


def main() -> None:
    index_faq_chunks(DEFAULT_CHUNKS, DEFAULT_DB, device="cpu", rebuild=False)


if __name__ == "__main__":
    main()
