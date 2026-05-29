import tiktoken
from typing import List, Dict

class DocumentChunker:
    """
    Token-based document chunker using OpenAI's tiktoken (cl100k_base).
    Splits text into chunks of roughly `chunk_size` tokens with `overlap` tokens.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def chunk_document(self, text: str, source_path: str) -> List[Dict[str, str]]:
        if not text.strip():
            return []
            
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "source": source_path
            })
            
            if end == len(tokens):
                break
                
            start += self.chunk_size - self.overlap
            
        return chunks
