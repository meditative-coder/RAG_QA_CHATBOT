import pytest
from unittest.mock import patch, MagicMock
from src.query_classifier import is_code_related_query

@patch("src.query_classifier.llm")
def test_is_code_related_query_yes(mock_llm):
    # Mock LLM response for code-related query
    mock_response = MagicMock()
    mock_response.content = "Yes"
    mock_llm.invoke.return_value = mock_response

    query = "How does the authentication module work?"
    result = is_code_related_query(query)
    mock_llm.invoke.assert_called_once()
    assert result is True

@patch("src.query_classifier.llm")
def test_is_code_related_query_no(mock_llm):
    # Mock LLM response for non-code-related query
    mock_response = MagicMock()
    mock_response.content = "No"
    mock_llm.invoke.return_value = mock_response

    query = "What is the weather today?"
    result = is_code_related_query(query)
    mock_llm.invoke.assert_called_once()
    assert result is False

@patch("src.query_classifier.llm")
def test_is_code_related_query_case_insensitive(mock_llm):
    # Mock LLM response with different casing and whitespace
    mock_response = MagicMock()
    mock_response.content = "  YES  "
    mock_llm.invoke.return_value = mock_response

    query = "Explain the database schema"
    result = is_code_related_query(query)
    mock_llm.invoke.assert_called_once()
    assert result is True

@patch("src.query_classifier.llm")
def test_is_code_related_query_unexpected_response(mock_llm):
    # Mock LLM response with unexpected content
    mock_response = MagicMock()
    mock_response.content = "Maybe"
    mock_llm.invoke.return_value = mock_response

    query = "Is this code-related?"
    result = is_code_related_query(query)
    mock_llm.invoke.assert_called_once()
    assert result is False
