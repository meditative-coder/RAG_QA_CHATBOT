
# Codebase QA CLI Tool

A command-line interface (CLI) tool to query multiple GitHub repositories code using powerful large language models (LLMs) with vector search. Perfect for developers wanting quick insights into codebases or repositories.

---

## Notes

- Currently, only Python files in public repos are considered for processing as the model is running locally with limited resources.

---

## Features

- Clone multiple GitHub repos at once  
- Load and chunk source code documents for efficient search  
- Create and merge vector stores with FAISS for fast retrieval  
- Classify user queries as code-related or general  
- Answer questions about code with context-aware LLM responses  
- Supports multiple LLM backends: Ollama, OpenAI, Together.xyz  

---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/meditative-coder/RAG_QA_CHATBOT
   cd RAG_QA_CHATBOT
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration:

- Copy `.env.example` to `.env` and set the variables:
  ```env
  TOGETHER_API_KEY=your_together_api_key
  OPENAI_API_KEY=your_openai_api_key
  ```

---

## Running Ollama Locally

I ran ollama locally as other provider has rate limiting.

### Steps to install and run Ollama locally:

1. **Install Ollama**

- macOS:
  ```bash
  brew install ollama
  ```

2. **Download and run an Ollama model**

- To download  and run Mistral (or other models):
  ```bash
  ollama run mistral
  ``` 
---

## Usage

Run the CLI tool:

```bash
python src/cli.py
```

You will be prompted to enter GitHub repository URLs (comma separated), for example:

```
Enter GitHub repo URLs (comma separated): https://github.com/user/repo1, https://github.com/user/repo2
```

Once the repos are processed and vector stores created, you can ask questions about the code or general programming.

Type `exit` to quit the interactive prompt.

---

## Configuration Details

- **MODEL_PROVIDER**: Specifies the LLM provider to use (e.g., "ollama", "openai", "together").
- **MODEL_NAME**: The name of the LLM model used from the selected provider.
- **EMBEDDING_MODEL_NAME**: The model name used to generate text embeddings.
- **CHUNK_SIZE**: Maximum size of document chunks for embedding.
- **CHUNK_OVERLAP**: Overlap size between consecutive document chunks.
- **VECTORSTORE_DIR**: Directory path to save/load vector stores.
- **RETRIEVAL_K**: Number of documents retrieved for general queries.
- **CODE_QUERY_RETRIEVAL_K**: Number of documents retrieved for code-related queries.


---

## Testing

Run the test suite with:

```bash
pytest tests/
```

Make sure to install the test dependencies if separate.

---

## Future Scope

1. Since this is a code-facing LLM, it can be configured to use more powerful code-centric models such as `codellama:34b-instruct` for improved accuracy and relevance.
2. Developing a user-friendly graphical interface (UI) will enhance usability and provide a better experience for end users.
3. Extend support to multiple programming languages beyond Python by improving document loaders and parsers.
4. Implement real-time incremental updates to vector stores as code repositories evolve.
5. Integrate authentication and access control to handle private repositories securely.
6. Enable deployment options for cloud for better scalability.

---
