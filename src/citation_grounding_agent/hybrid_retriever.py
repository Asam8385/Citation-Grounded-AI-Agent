from pathlib import Path
from typing import Literal


from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from citation_grounding_agent.bm25_retriever import build_bm25_retriever
from citation_grounding_agent.document_adapter import load_langchain_documents

from citation_grounding_agent.vectore_store import (
    DEFAULT_CHUNKS,
    DEFAULT_DB,
    open_vector_store,
    semantic_search,
)

SearchMode = Literal["semantic", "bm25", "hybrid"]

RRF_K = 60

class HybridRetriever:

   def __init__(
         self,
         documents: list[Document],
         vector_store: Chroma | None,
         *,
         candidate_k : int = 5
   ) -> None:
        if not documents:
            raise ValueError("At least one FAQ document is required")
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")


        self.documents_by_id: dict[str, Document] = {}

        for document in documents:
            chunk_id = document.metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Every document needs a valid chunk_id")
            if chunk_id in self.documents_by_id:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            self.documents_by_id[chunk_id] = document


        self.vector_store = vector_store
        self.bm25: BM25Retriever = build_bm25_retriever(
            documents, k=candidate_k
        )  

   @classmethod
   def from_paths(
      cls,
      chunks_path: Path = DEFAULT_CHUNKS,
      db_path: Path = DEFAULT_DB,
      *,
      device: str = "cpu",
      candidate_k: int = 5,
      enable_semantic: bool = True,
   ) -> "HybridRetriever":

       documents = load_langchain_documents(chunks_path)
       store = open_vector_store(db_path , device=device) if enable_semantic else None

       return cls(documents, store, candidate_k=candidate_k)
   

   def _canonicalize(self, documents: list[Document]) -> list[Document]:
      """Keep only chunk IDs present in the validated source export."""
      result: list[Document] = []
      for document in documents:
         chunk_id = document.metadata.get("chunk_id")
         if isinstance(chunk_id, str) and chunk_id in self.documents_by_id:
               result.append(self.documents_by_id[chunk_id])
      return result         

   


   def retrieve(
      self,
      question: str,
      *,
      k: int = 5,
      mode: SearchMode = "hybrid",
   ) -> list[Document]:
      """Find evidence and merge ranked lists with Reciprocal Rank Fusion.

      This is intentionally NOT cross-encoder or LLM reranking.
      """
      if not question.strip():
         raise ValueError("question cannot be blank")
      if k < 1:
         raise ValueError("k must be >= 1")
      if mode not in ("semantic", "bm25", "hybrid"):
         raise ValueError(f"Unsupported search mode: {mode}")

      if mode in ("semantic", "hybrid"):
         if self.vector_store is None:
               raise ValueError("Chroma is not configured; run with BM25 mode")
         dense = self._canonicalize(
               semantic_search(self.vector_store, question, k=k)
         )
      else:
         dense = []

      if mode in ("bm25", "hybrid"):
         # BM25Retriever.k is mutable, so set it per call. CLI/agent
         # share this instance sequentially, not across threads.
         self.bm25.k = k
         sparse = self._canonicalize(self.bm25.invoke(question))

      else:
         sparse = []

      # Reciprocal Rank Fusion: each list gives 1 / (RRF_K + rank) points
      # to every chunk it contains; chunks found by both lists add up.
      scores: dict[str, float] = {}
      for results in (dense, sparse):
         for rank, document in enumerate(results, start=1):
               chunk_id = str(document.metadata["chunk_id"])
               scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

      # sorted() is stable, so ties keep first-seen order (dense before sparse).
      ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
      return [self.documents_by_id[chunk_id] for chunk_id in ranked_ids]


   


   
