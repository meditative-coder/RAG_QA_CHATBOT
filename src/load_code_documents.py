from langchain_community.document_loaders import DirectoryLoader, PythonLoader
from langchain.schema import Document
import glob
import os


from langchain.schema import Document
import glob
import os
from typing import List

def load_code_documents(repo_name: str, repo_path: str) -> List[Document]:
    """
    Loads Python code files from the given repository path and converts them into Document objects.

    :param repo_name: Name of the repository.
    :param repo_path: Local path to the repository.
    :returns: List of Document objects representing code files.
    """
    documents: List[Document] = []
    for file_path in glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"repo_name": repo_name, "file_path": file_path}
                )
                documents.append(doc)
        except Exception:
            continue
    return documents
