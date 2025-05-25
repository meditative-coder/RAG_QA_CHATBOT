import os
import pytest
from unittest.mock import patch, MagicMock

import src.llm_factory as llm_factory
from src.config import MODEL_PROVIDER, MODEL_NAME

@pytest.fixture(autouse=True)
def reset_llm_instance():
    # Reset the singleton instance before each test
    llm_factory._llm_instance = None
    yield
    llm_factory._llm_instance = None

@patch("src.llm_factory.ChatOllama")
def test_get_llm_ollama(mock_chat_ollama):
    with patch("config.MODEL_PROVIDER", "ollama"), patch("config.MODEL_NAME", "mistral"):
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance

        llm = llm_factory.get_llm()
        mock_chat_ollama.assert_called_once_with(model="mistral")
        assert llm == mock_instance

@patch("src.llm_factory.ChatOpenAI")
def test_get_llm_openai(mock_chat_openai):
    # Patch config.MODEL_PROVIDER and MODEL_NAME within the llm_factory module context
    with patch.object(llm_factory, "MODEL_PROVIDER", "openai"), \
         patch.object(llm_factory, "MODEL_NAME", "gpt-4"):

        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        llm = llm_factory.get_llm()
        mock_chat_openai.assert_called_once_with(model_name="gpt-4")
        assert llm == mock_instance


@patch("src.llm_factory.ChatOpenAI")
def test_get_llm_together(mock_chat_openai):
    with patch.dict(os.environ, {"TOGETHER_API_KEY": "dummy_key"}), \
         patch.object(llm_factory, "MODEL_PROVIDER", "together"), \
         patch.object(llm_factory, "MODEL_NAME", "together-model"):

        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        llm = llm_factory.get_llm()

        mock_chat_openai.assert_called_once_with(
            base_url="https://api.together.xyz/v1",
            api_key="dummy_key",
            model="together-model",
        )
        assert llm == mock_instance


def test_get_llm_unsupported_provider():
    with patch.object(llm_factory, "MODEL_PROVIDER", "unsupported"):
        with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER: unsupported"):
            llm_factory.get_llm()


def test_get_llm_singleton_behavior():
    with patch("config.MODEL_PROVIDER", "ollama"), patch("config.MODEL_NAME", "mistral"):
        first_instance = MagicMock()
        with patch("src.llm_factory.ChatOllama", return_value=first_instance):
            llm1 = llm_factory.get_llm()
            llm2 = llm_factory.get_llm()
            # Should only create instance once
            assert llm1 is llm2
