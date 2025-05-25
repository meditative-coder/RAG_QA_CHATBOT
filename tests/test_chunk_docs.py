import pytest
from langchain.schema import Document
from src.chunk_docs import split_documents
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def test_split_documents_chunks_correctly():
    # Create a long string that will be chunked
    long_text = "A" * (CHUNK_SIZE * 2)  # Ensures at least two chunks
    documents = [Document(page_content=long_text, metadata={"file_path": "example.py"})]

    chunks = split_documents(documents)

    # Check that the result is not empty
    assert chunks, "Expected non-empty list of chunks"

    # Check that each chunk is a Document and has appropriate size
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= CHUNK_SIZE

    # Check that metadata is preserved
    for chunk in chunks:
        assert chunk.metadata["file_path"] == "example.py"

    # Ensure some overlap exists if more than one chunk
    if len(chunks) > 1:
        assert chunks[0].page_content[-CHUNK_OVERLAP:] == chunks[1].page_content[:CHUNK_OVERLAP]
