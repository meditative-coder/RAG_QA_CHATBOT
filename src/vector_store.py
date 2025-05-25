import os
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from config import EMBEDDING_MODEL_NAME, VECTORSTORE_DIR

def create_vector_store(chunks: List[Document], repo_name: str, db_dir: str = VECTORSTORE_DIR) -> FAISS:
    """
    Creates or loads a FAISS vector store for the given repo chunks.

    :param chunks: List of chunked Document objects to index.
    :param repo_name: Repository name, used as directory name for saving the vector store.
    :param db_dir: Directory to store vector stores.
    :returns: FAISS vector store instance.
    :raises ValueError: If chunks list is empty.
    """
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, repo_name)

    if os.path.exists(db_path):
        print(f"Loading existing vector store for {repo_name}")
        return FAISS.load_local(
            db_path,
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
            allow_dangerous_deserialization=True
        )

    print(f"Creating new vector store for {repo_name}, chunks: {len(chunks)}")
    if not chunks:
        raise ValueError(f"No chunks to index for {repo_name}!")

    # Add unique IDs to avoid collision
    for i, doc in enumerate(chunks):
        doc.metadata["doc_id"] = f"{repo_name}_{i}_{uuid.uuid4()}"

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(db_path)
    return db
