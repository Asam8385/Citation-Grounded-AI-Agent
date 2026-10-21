import re 
from pathlib import Path


from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from citation_grounding_agent.document_adapter import load_langchain_documents

def tokenize(text: str) -> list[str]:

   return re.findall(r"[a-z0-9]+" , text.casefold())


def build_bm25_retriever(
    documents: list[Document],
    *,
    k: int = 3,
) -> BM25Retriever:
    """Create an in-memory BM25 retriever preserving document metadata."""
    if not documents:
        raise ValueError("No FAQ documents available for BM25 indexing")
    if k < 1:
        raise ValueError("k must be at least 1")

    return BM25Retriever.from_documents(
        documents,
        preprocess_func=tokenize,
        k=k,
    )
    


def load_bm25_retriever(
    chunks_path: Path,
    *,
    k: int = 3,
) -> BM25Retriever:
    """Load exported FAQ chunks and prepare a keyword retriever."""
    documents = load_langchain_documents(chunks_path)
    return build_bm25_retriever(documents, k=k)