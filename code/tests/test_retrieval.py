import os
import tempfile
import pytest
from code.retrieval.chunking import DocumentChunker
from code.retrieval.ingestion import CorpusIngestor
from code.agents.retrieval_agent import RetrievalAgent

def test_chunking():
    chunker = DocumentChunker(chunk_size=10, overlap=2)
    # roughly 15 words
    text = "word " * 15
    chunks = chunker.chunk_document(text, "dummy.txt")
    assert len(chunks) > 1
    assert "dummy.txt" == chunks[0]["source"]

def test_ingestion(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.html"
    p.write_text("<html><body>This is a test document about resetting passwords.</body></html>")
    
    ingestor = CorpusIngestor(chunk_size=50)
    chunks = ingestor.ingest_directories([str(d)])
    
    assert len(chunks) == 1
    assert "passwords" in chunks[0]["text"]
    assert "<html>" not in chunks[0]["text"] # HTML should be stripped

def test_retrieval_agent_deterministic_rrf(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    f1.write_text("Apples are red and sweet.")
    f2.write_text("Bananas are yellow and long.")
    
    index_dir = tmp_path / "index"
    agent = RetrievalAgent(index_dir=str(index_dir))
    agent.initialize_corpus([str(tmp_path)])
    
    res = agent.retrieve("red apples", k=1)
    
    assert len(res.retrieved_chunks) == 1
    assert "Apples" in res.retrieved_chunks[0]
    assert str(f1.absolute()) in res.source_documents[0]
    assert len(res.retrieval_scores) == 1

def test_citation_validation(tmp_path):
    f1 = tmp_path / "file1.txt"
    f1.write_text("Secret information.")
    
    index_dir = tmp_path / "index"
    agent = RetrievalAgent(index_dir=str(index_dir))
    agent.initialize_corpus([str(tmp_path)])
    
    # Delete the file to simulate hallucinated path or deleted file
    f1.unlink()
    
    res = agent.retrieve("Secret", k=1)
    # The citation validator should catch that it doesn't exist and drop it
    assert len(res.retrieved_chunks) == 0
    assert len(res.source_documents) == 0
