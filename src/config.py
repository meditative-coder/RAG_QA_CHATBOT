MODEL_PROVIDER: str = "ollama"  # or "openai", "together"
# Use below model for better code responses. Commenting this as it is large model
# MODEL_NAME = "codellama:34b-instruct
MODEL_NAME: str = "mistral"  # example model name for your LLM provider

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

VECTORSTORE_DIR: str = "../vectorstores"

RETRIEVAL_K: int = 3
CODE_QUERY_RETRIEVAL_K: int = 12
