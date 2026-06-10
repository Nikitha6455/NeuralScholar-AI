from src.pdf_loader import load_pdf

pages = load_pdf("data/uploads/sample.pdf")

print("Total Pages:", len(pages))

print("\nFirst Page:\n")

print(pages[0]["text"][:500])