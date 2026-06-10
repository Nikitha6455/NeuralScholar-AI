from src.llm import get_llm

llm = get_llm()

response = llm.invoke(
    "What is Retrieval Augmented Generation?"
)

print(response.content)