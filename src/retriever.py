import numpy as np

def retrieve_chunks(query_embedding, index, chunks, k=5):

    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"]
        })

    return results