from rag.llm import OllamaEmbeddings, OllamaGenerator
from rag.rag import RAG

if __name__ == "__main__":
    # Initialize the generator and embeddings clients
    generator_client = OllamaGenerator(model="deepseek-r1:1.5b")
    embeddings_client = OllamaEmbeddings(model="all-minilm:latest")

    # Initialize the RAG model by injecting the generator and embeddings clients
    rag = RAG(generator_client=generator_client, embeddings_client=embeddings_client)
    rag.provide_context(
        document_path=r".\data\Designing Data-Intensive Applications.pdf",
        document_type="PDF",
    )

    prompt = "What is a data-intensive application?"
    response = rag.execute(prompt)
    print(response)
