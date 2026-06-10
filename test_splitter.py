from src.pdf_loader import load_pdf
from src.text_splitter import split_documents

pages = load_pdf("data/uploads/sample.pdf")

chunks = split_documents(pages)

print("Total Pages:", len(pages))
print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0]["text"])