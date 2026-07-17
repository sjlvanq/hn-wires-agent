from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings
from config.settings import settings


class OllamaEmbeddings:
    """Wrapper for Ollama embeddings optimized for small models."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        """
        Initialize Ollama embeddings.

        Args:
            model: Ollama model name. Defaults to settings.ollama_embedding_model.
            base_url: Ollama base URL. Defaults to settings.ollama_base_url.
        """
        self.model = model or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self._embeddings = LCOllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._embeddings.embed_documents(texts)
