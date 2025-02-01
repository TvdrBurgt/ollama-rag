import faiss
import numpy as np


def retrieve_top_k_documents(
    document_embeddings: list[float],
    prompt_embedding: list[float],
    k: int = 5,
):
    # Convert embeddings to numpy arrays
    document_embeddings = np.array(document_embeddings).astype(np.float32)
    prompt_embedding = np.array(prompt_embedding).astype(np.float32)

    # Normalize embeddings to use cosine similarity
    faiss.normalize_L2(document_embeddings)
    faiss.normalize_L2(prompt_embedding.reshape(1, -1))

    # Create FAISS index
    index = faiss.IndexFlatIP(
        document_embeddings.shape[1]
    )  # IP = Inner Product (cosine similarity after normalization)
    index.add(document_embeddings)  # Add document embeddings to index

    # Search for the top-k most similar documents
    _, top_k_indices = index.search(prompt_embedding.reshape(1, -1), k)

    return top_k_indices[0]
