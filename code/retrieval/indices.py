import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from code.utils.logger import get_logger

logger = get_logger(__name__)

class HybridIndex:
    """
    Manages both BM25 and FAISS dense vector indices.
    Allows saving and loading indices to disk.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # FAISS Index FlatL2 (which corresponds to Euclidean distance)
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        
        # BM25 Index
        self.bm25 = None
        
        # Document store: maps index -> {"text": str, "source": str}
        self.documents = []
        
    def build(self, chunks: List[Dict[str, str]]):
        """
        Builds both indices from the provided chunks deterministically.
        """
        if not chunks:
            logger.warning("No chunks provided to build index.")
            return
            
        logger.info(f"Building indices for {len(chunks)} chunks...")
        self.documents = chunks
        
        # 1. Build FAISS
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self.faiss_index.add(embeddings)
        
        # 2. Build BM25
        # Tokenize by simple split (sufficient for basic BM25 implementation)
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Indices successfully built.")
        
    def search_faiss(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Returns list of (doc_index, distance)"""
        if self.faiss_index.ntotal == 0:
            return []
            
        query_emb = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_emb, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))
        return results
        
    def search_bm25(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Returns list of (doc_index, score)"""
        if self.bm25 is None:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only return if there's some match
                results.append((int(idx), float(scores[idx])))
        return results
        
    def get_document(self, index: int) -> Dict[str, str]:
        return self.documents[index]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(save_dir, "faiss.index"))
        
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "bm25": self.bm25
            }, f)
            
    def load(self, save_dir: str):
        self.faiss_index = faiss.read_index(os.path.join(save_dir, "faiss.index"))
        with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.bm25 = data["bm25"]
