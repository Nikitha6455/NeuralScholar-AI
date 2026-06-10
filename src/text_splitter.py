from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(pages):
    """
    Split extracted PDF pages into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:

        split_texts = splitter.split_text(page["text"])

        for chunk in split_texts:

            chunks.append(
                {
                    "page": page["page"],
                    "text": chunk
                }
            )

    return chunks