from typing import Literal

from rag.llm.base import LLMClient
from rag.retriever.faiss import retrieve_top_k_documents
from rag.retriever.static.pdf import PDFIngestor


class RAG:
    def __init__(self, generator_client: LLMClient, embeddings_client: LLMClient):
        self.generator_client = generator_client
        self.embeddings_client = embeddings_client
        self._vector_store = None

    def execute(self, prompt: str) -> str:
        # Generate embeddings for the prompt
        prompt_embeddings = self.embeddings_client.generate_embeddings(prompt)

        # Retrieve the top-k most similar documents
        document_chunks, document_embeddings = self._vector_store
        indices = retrieve_top_k_documents(
            document_embeddings=document_embeddings,
            prompt_embedding=prompt_embeddings,
            k=5,
        )

        # Provide context to the prompt
        context = [document_chunks[i] for i in indices]
        enriched_prompt = f"Context:\n{chr(10).join(context)}\n\nQuestion: {prompt}"

        return self.generator_client.generate_text(enriched_prompt)

    def provide_context(self, document_path: str, document_type: Literal["PDF"]):
        provider = None
        match document_type:
            case "PDF":
                provider = PDFIngestor
            case _:
                raise NotImplementedError(
                    "Only PDF documents are supported at the moment."
                )

        text, embeddings = provider(embeddings_client=self.embeddings_client).ingest(
            document_path
        )

        self._vector_store = (text, embeddings)
