from .base import LLMClient

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # required, but unused
DEFAULT_GENERATOR_MODEL = "deepseek-r1:1.5b"
DEFAULT_EMBEDDINGS_MODEL = "all-minilm:latest"


class OllamaGenerator(LLMClient):
    def __init__(self, model: str = DEFAULT_GENERATOR_MODEL):
        super().__init__(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, model=model)

    def generate(self, prompt: str) -> str:
        return self.generate_text(prompt)


class OllamaEmbeddings(LLMClient):
    def __init__(self, model: str = DEFAULT_EMBEDDINGS_MODEL):
        super().__init__(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, model=model)

    def generate(self, text: str) -> list:
        return self.generate_embeddings(text)
