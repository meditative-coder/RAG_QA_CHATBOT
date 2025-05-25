from llm_factory import get_llm

llm = get_llm()

def is_code_related_query(query: str) -> bool:
    """
    Classifies if the query is specifically about source code or repository content using the LLM.

    :param query: User query string.
    :returns: True if code-related, False otherwise.
    """
    prompt = (
        "Answer with only 'Yes' or 'No'.\n"
        "Is the following question specifically about the source code, programming logic, or repository content?\n"
        f"Question: {query}"
    )
    response = llm.invoke(prompt)
    text = response.content.strip().lower()
    return text == "yes"
