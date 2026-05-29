import os
from bs4 import BeautifulSoup
from typing import List, Dict
from code.utils.logger import get_logger
from code.retrieval.chunking import DocumentChunker

logger = get_logger(__name__)

class CorpusIngestor:
    """
    Ingests text, markdown, and html files from specified directories.
    Outputs structured overlapping chunks with source paths.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        
    def extract_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        if file_path.endswith(".html"):
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text(separator="\\n", strip=True)
            
        return content

    def ingest_directories(self, directories: List[str]) -> List[Dict[str, str]]:
        all_chunks = []
        
        for directory in directories:
            # We want to use absolute path, but since we are running from MLE-hiring,
            # directories might be relative like "data/claude". Let's convert to absolute.
            abs_dir = os.path.abspath(directory)
            if not os.path.exists(abs_dir):
                logger.warning(f"Directory not found: {abs_dir}")
                continue
                
            for root, _, files in os.walk(abs_dir):
                for file in files:
                    if file.endswith((".txt", ".md", ".html")):
                        file_path = os.path.join(root, file)
                        try:
                            text = self.extract_text(file_path)
                            chunks = self.chunker.chunk_document(text, file_path)
                            all_chunks.extend(chunks)
                        except Exception as e:
                            logger.error(f"Error ingesting {file_path}: {e}")
                            
        logger.info(f"Ingested {len(all_chunks)} total chunks from {directories}")
        return all_chunks
