from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings
from config.settings import settings


class OllamaEmbeddings:
    """Thin wrapper around :class:`langchain_ollama.OllamaEmbeddings`.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        """Create an embeddings instance.

        Parameters
        ----------
        model : str | None
            Name of the Ollama embedding model.  Defaults to
            :data:`settings.ollama_embedding_model`.
        base_url : str | None
            Base URL for the Ollama server.  Defaults to
            :data:`settings.ollama_base_url`.
        """
        self.model = model or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self._embeddings = LCOllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding for a single piece of text.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of documents.

        Parameters
        ----------
        texts : list[str]
            List of strings to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors.
        """
        return self._embeddings.embed_documents(texts)
