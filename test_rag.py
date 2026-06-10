from sentence_transformers import SentenceTransformer

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_faiss_index
from src.retriever import retrieve_chunks
from src.llm import get_llm


# Load model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDF
pages = load_pdf("data/uploads/sample.pdf")

# Split
chunks = split_documents(pages)

# Embeddings
embeddings = create_embeddings(chunks)

# FAISS
index = create_faiss_index(embeddings)

# Gemini
llm = get_llm()

# User Question
query = input("Ask a question: ")

# Query Embedding
query_embedding = embedding_model.encode(query)

# Retrieve Chunks
results = retrieve_chunks(
    query_embedding,
    index,
    chunks,
    k=5
)

# Build Context
context = "\n\n".join(
    [chunk["text"] for chunk in results]
)

# Prompt
prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

# Gemini Answer
response = llm.invoke(prompt)

print("\n")
print("=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)

print("\n")
print("=" * 60)
print("SOURCES")
print("=" * 60)

for chunk in results:
    print(f"Page {chunk['page']} | Score: {chunk['score']:.2f}")