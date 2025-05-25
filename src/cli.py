import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from typing import List


from clone_repo import clone_repos
from load_code_documents import load_code_documents
from chunk_docs import split_documents
from vector_store import create_vector_store
from retriever import create_qa_chain
from query_classifier import is_code_related_query
from llm_factory import get_llm
from config import CODE_QUERY_RETRIEVAL_K

def get_repo_urls() -> List[str]:
    """
    Prompts the user to enter comma separated GitHub repository URLs.

    :returns: List of GitHub repo URLs.
    """
    repo_urls_input = input("Enter GitHub repo URLs (comma separated): ")
    return [url.strip() for url in repo_urls_input.split(",") if url.strip()]

def process_repositories(repo_urls: List[str]):
    """
    Clones repos, loads documents, chunks, and creates vector stores.

    :param repo_urls: List of GitHub repository URLs.
    :returns: List of FAISS vector stores for each repo.
    """
    vectorstores = []
    for repo_name, repo_path in clone_repos(repo_urls):
        print(f"\nProcessing {repo_name}...")
        docs = load_code_documents(repo_name, repo_path)
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks, repo_name)
        vectorstores.append(vectorstore)
    return vectorstores

def combine_vectorstores(vectorstores):
    """
    Merges multiple vectorstores into one combined store.

    :param vectorstores: List of FAISS vectorstores.
    :returns: Combined FAISS vectorstore.
    :raises ValueError: If vectorstores list is empty.
    """
    if not vectorstores:
        raise ValueError("No vectorstores available!")
    combined = vectorstores[0]
    for vs in vectorstores[1:]:
        combined.merge_from(vs)
    return combined

def handle_code_query(query: str, vectorstore, llm) -> str:
    """
    Retrieves relevant code snippets for the query and generates an answer using the LLM.

    :param query: User question related to code.
    :param vectorstore: FAISS vectorstore instance.
    :param llm: LLM instance.
    :returns: Answer string from the LLM.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": CODE_QUERY_RETRIEVAL_K})
    docs = retriever.get_relevant_documents(query)

    grouped_docs = defaultdict(list)
    for doc in docs:
        repo = doc.metadata.get("repo_name", "Unknown Repo")
        grouped_docs[repo].append(doc)

    summaries = []
    for repo, repo_docs in grouped_docs.items():
        summaries.append(f"\n **{repo}**")
        for d in repo_docs:
            path = d.metadata.get("file_path", "unknown")
            preview = d.page_content.strip().split("\n")[0][:80]
            summaries.append(f"- `{path}`: {preview}...")

    context = "\n".join(summaries)
    prompt = (
        f"The user asked: \"{query}\"\n"
        f"Here are relevant code snippets grouped by repository:\n{context}\n\n"
        f"Answer the question in a helpful, structured way."
    )

    response = llm.invoke(prompt)
    return response.content.strip()

def interactive_loop(vectorstore, llm):
    """
    Interactive CLI loop for user queries.

    :param vectorstore: Combined FAISS vectorstore.
    :param llm: LLM instance.
    """
    print("\nAsk questions about the code across all repos or general programming (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        if not is_code_related_query(query):
            response = llm.invoke(query)
            print("\nBot:", response.content.strip())
        else:
            answer = handle_code_query(query, vectorstore, llm)
            print("\nBot:", answer)

def main():
    """
    Main entry point of the CLI app.
    """
    repo_urls = get_repo_urls()
    vectorstores = process_repositories(repo_urls)
    combined_vectorstore = combine_vectorstores(vectorstores)
    llm = get_llm()
    interactive_loop(combined_vectorstore, llm)

if __name__ == "__main__":
    main()
