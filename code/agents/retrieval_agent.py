import os
from typing import List
from code.utils.logger import get_logger
from code.schemas import RetrievalResult
from code.retrieval.indices import HybridIndex
from code.retrieval.ingestion import CorpusIngestor

logger = get_logger(__name__)

class RetrievalAgent:
    """
    Agent responsible for deterministic hybrid retrieval.
    Applies Reciprocal Rank Fusion (RRF) and validates citations.
    """
    def __init__(self, index_dir: str = "data/embeddings"):
        self.index = HybridIndex()
        self.index_dir = index_dir
        self.is_loaded = False
        
    def initialize_corpus(self, directories: List[str]):
        """Ingests documents and builds indices."""
        ingestor = CorpusIngestor()
        chunks = ingestor.ingest_directories(directories)
        self.index.build(chunks)
        self.index.save(self.index_dir)
        self.is_loaded = True
        
    def load_index(self):
        """Loads index from disk."""
        if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
            self.index.load(self.index_dir)
            self.is_loaded = True
        else:
            logger.warning("No index found. Need to initialize_corpus first.")

    def retrieve(self, query: str, k: int = 5, rrf_k: int = 60) -> RetrievalResult:
        """
        Executes hybrid retrieval, applying RRF, and strictly validating sources.
        """
        if not self.is_loaded:
            logger.error("Index not loaded. Returning empty result.")
            return RetrievalResult(retrieved_chunks=[], source_documents=[], retrieval_scores=[])
            
        # Retrieve more from each to get better fusion intersection
        faiss_results = self.index.search_faiss(query, k=k*3) 
        bm25_results = self.index.search_bm25(query, k=k*3)
        
        # Calculate RRF
        rrf_scores = {}
        
        # Lower rank is better (0-indexed here, so we add 1)
        for rank, (doc_id, _) in enumerate(faiss_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # Sort deterministically by RRF score (descending), tie-break by doc_id
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
        
        retrieved_chunks = []
        source_documents = []
        retrieval_scores = []
        
        for doc_id, score in sorted_docs:
            if len(retrieved_chunks) >= k:
                break
                
            doc = self.index.get_document(doc_id)
            source_path = doc["source"]
            
            # ** Citation validation utility **
            if not os.path.exists(source_path):
                logger.error(f"Hallucinated or deleted citation detected! Dropping: {source_path}")
                continue
                
            retrieved_chunks.append(doc["text"])
            source_documents.append(source_path)
            retrieval_scores.append(score)
            
        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            source_documents=source_documents,
            retrieval_scores=retrieval_scores
        )
