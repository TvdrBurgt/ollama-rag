# Project - Retrieval Augmented Generation (RAG)

In this project, we make use of the OpenAI API to implement a RAG model.

To connect to an LLM provider, you can either host one yourself using Ollama, or you can access your favorite provider using their API (e.g., OpenAI, Llama, Deepseek, etc.). We will use Ollama in this project because it is free, easy to use, and provides us with a good range of LLMs to choose from.

Visit [Ollama](https://ollama.com/) and download the Ollama CLI. We want to get started swiftly, so we pull two lightweight models from the Ollama Registry:

```bash
ollama pull all-minilm
ollama pull deepseek-r1:1.5b
```

We will use `all-minilm` to generate embeddings, and `deepseek-r1:1.5b` to generate text.
