import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.llm.base import LLMClient
from rag.retriever.static.base import Ingestor


class PDFIngestor(Ingestor):
    def __init__(self, embeddings_client: LLMClient):
        self.embeddings_client = embeddings_client

    def ingest(self, path: str) -> tuple[list[str], list[list[float]]]:
        # Read the PDF file
        with pymupdf.open(path) as document:
            text = ""
            for page in document:
                text += page.get_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # Generate embeddings for each chunk
        embeddings = [
            self.embeddings_client.generate_embeddings(chunk) for chunk in chunks
        ]

        # TODO: Store in DB

        return chunks, embeddings
