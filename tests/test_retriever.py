import pytest
from unittest.mock import MagicMock, patch
from langchain.chains import RetrievalQA
from src.retriever import create_qa_chain

@patch("src.retriever.get_llm")
def test_create_qa_chain_returns_retrievalqa(mock_get_llm):
    # Mock the FAISS vector store
    mock_faiss = MagicMock()
    mock_retriever = MagicMock()
    mock_faiss.as_retriever.return_value = mock_retriever

    # Mock the LLM returned by get_llm
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    # Patch RetrievalQA.from_chain_type to return a mock QA chain
    with patch("src.retriever.RetrievalQA.from_chain_type") as mock_from_chain_type:
        mock_qa_chain = MagicMock(spec=RetrievalQA)
        mock_from_chain_type.return_value = mock_qa_chain

        qa_chain = create_qa_chain(mock_faiss)

        # Assertions
        mock_faiss.as_retriever.assert_called_once_with(search_kwargs={'k': 3})
        mock_get_llm.assert_called_once()
        mock_from_chain_type.assert_called_once_with(llm=mock_llm, retriever=mock_retriever)

        assert isinstance(qa_chain, RetrievalQA)
