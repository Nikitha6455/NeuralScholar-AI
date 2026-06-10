from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import create_embeddings

pages = load_pdf("data/uploads/sample.pdf")

chunks = split_documents(pages)

embeddings = create_embeddings(chunks)

print("Total Chunks:", len(chunks))

print("Embedding Shape:")
print(embeddings.shape)