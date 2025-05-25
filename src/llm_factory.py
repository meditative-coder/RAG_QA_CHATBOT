import os
from typing import Optional
from config import MODEL_PROVIDER, MODEL_NAME

from langchain_community.chat_models import ChatOllama, ChatOpenAI

_llm_instance = None

def get_llm() -> object:
    """
    Singleton factory to get the LLM instance based on configuration.

    :returns: An instance of the configured LLM model.
    :raises ValueError: If MODEL_PROVIDER is unsupported.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if MODEL_PROVIDER == "ollama":
        _llm_instance = ChatOllama(model=MODEL_NAME)

    elif MODEL_PROVIDER == "openai":
        _llm_instance = ChatOpenAI(model_name=MODEL_NAME)

    elif MODEL_PROVIDER == "together":
        _llm_instance = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
            model=MODEL_NAME
        )

    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")

    return _llm_instance
