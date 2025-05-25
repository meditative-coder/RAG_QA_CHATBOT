import pytest
from unittest.mock import mock_open, patch
from src.load_code_documents import load_code_documents
from langchain.schema import Document


@patch("src.load_code_documents.glob.glob")
@patch("builtins.open", new_callable=mock_open, read_data="print('Hello world')\n")
def test_load_code_documents_success(mock_file, mock_glob):
    repo_name = "test-repo"
    repo_path = "/fake/repo/path"
    fake_file_path = "/fake/repo/path/module.py"

    # Fake one Python file
    mock_glob.return_value = [fake_file_path]

    documents = load_code_documents(repo_name, repo_path)

    assert len(documents) == 1
    doc = documents[0]
    assert isinstance(doc, Document)
    assert doc.page_content == "print('Hello world')\n"
    assert doc.metadata == {
        "repo_name": repo_name,
        "file_path": fake_file_path
    }

    mock_file.assert_called_once_with(fake_file_path, "r", encoding="utf-8")
    mock_glob.assert_called_once_with("/fake/repo/path/**/*.py", recursive=True)
