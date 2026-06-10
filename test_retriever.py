from sentence_transformers import SentenceTransformer

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_faiss_index
from src.retriever import retrieve_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")

pages = load_pdf("data/uploads/sample.pdf")

chunks = split_documents(pages)

embeddings = create_embeddings(chunks)

index = create_faiss_index(embeddings)

query = input("Ask a question: ")

query_embedding = model.encode(query)

results = retrieve_chunks(
    query_embedding,
    index,
    chunks,
    k=3
)

for i, chunk in enumerate(results, start=1):
    print("\n")
    print("=" * 50)
    print(f"Result {i}")
    print(f"Page: {chunk['page']}")
    print(chunk["text"][:500])