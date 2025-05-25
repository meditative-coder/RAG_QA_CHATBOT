from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter.

    :param documents: List of Document objects to split.
    :returns: List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)
