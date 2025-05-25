from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from llm_factory import get_llm
from typing import Union
from config import CODE_QUERY_RETRIEVAL_K

def create_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Creates a RetrievalQA chain using the provided vector store and LLM.

    :param vector_store: FAISS vector store instance.
    :returns: RetrievalQA chain object.
    """
    retriever = vector_store.as_retriever(search_kwargs={'k': CODE_QUERY_RETRIEVAL_K})
    llm = get_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
