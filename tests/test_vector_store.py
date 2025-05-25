import os
import shutil
import pytest
from langchain.schema import Document
from src.vector_store import create_vector_store
from src.config import VECTORSTORE_DIR
from langchain_community.vectorstores import FAISS

@pytest.fixture
def dummy_chunks():
    return [
        Document(page_content="def foo(): pass", metadata={"file_path": "test1.py"}),
        Document(page_content="def bar(): pass", metadata={"file_path": "test2.py"}),
    ]

@pytest.fixture
def repo_name():
    return "test_repo"

def test_create_vector_store_creates_new_store(tmp_path, dummy_chunks, repo_name):
    db_dir = tmp_path / "vector_db"
    db = create_vector_store(dummy_chunks, repo_name, str(db_dir))

    # Check that a FAISS object is returned
    assert isinstance(db, FAISS)

    # Ensure directory was created
    expected_path = db_dir / repo_name
    assert expected_path.exists()
    assert (expected_path / "index.faiss").exists()
    assert (expected_path / "index.pkl").exists()

def test_create_vector_store_raises_for_empty_chunks(repo_name):
    with pytest.raises(ValueError, match="No chunks to index"):
        create_vector_store([], repo_name)

def test_create_vector_store_loads_existing(tmp_path, dummy_chunks, repo_name):
    db_dir = tmp_path / "vector_db"
    
    # First create and save
    db1 = create_vector_store(dummy_chunks, repo_name, str(db_dir))

    # Now test loading from the same path
    db2 = create_vector_store(dummy_chunks, repo_name, str(db_dir))

    assert isinstance(db2, FAISS)
    assert db1 is not db2  # They are different instances but loaded from disk

    # Cleanup
    shutil.rmtree(str(db_dir))
