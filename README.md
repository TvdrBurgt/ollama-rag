# Project - Retrieval Augmented Generation (RAG)

In this project, we make use of the OpenAI API to implement a Retrieval Augmented Generation (RAG) model. This project demonstrates the integration of large language models (LLMs) with document retrieval systems to provide contextually enriched responses.

## Overview

To connect to an LLM provider, you can either host one yourself using Ollama, or you can access your favorite provider using their API (e.g., OpenAI, Llama, Deepseek, etc.). We will use Ollama in this project because it is free, easy to use, and provides us with a good range of LLMs to choose from.

## Setup

Visit [Ollama](https://ollama.com/) and download the Ollama CLI. We want to get started swiftly, so we pull two lightweight models from the Ollama Registry:

```bash
ollama pull all-minilm
ollama pull deepseek-r1:1.5b
```

We will use `all-minilm` to generate embeddings, and `deepseek-r1:1.5b` to generate text.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/TvdrBurgt/ollama-rag
    cd ollama-rag
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install .
    ```

## Usage

1. Ensure that the Ollama CLI is running and the models are pulled.

2. Run the main script:
    ```bash
    python main.py
    ```

3. The script will provide context from the specified PDF document and generate a response to the given prompt.

## Project Structure

- `main.py`: Entry point of the application.
- `rag/`: Contains the core logic for the RAG model.
  - `llm/`: Contains the logic for connecting to LLM providers.
  - `retriever/`: Contains the logic for document retrieval and ingestion.
- `requirements/`: Contains the dependency requirements for the project.
- `data/`: Directory to store input documents.
- `README.md`: Project documentation.
