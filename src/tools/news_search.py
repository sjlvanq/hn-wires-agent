from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from database import NewsRepository
from embeddings import OllamaEmbeddings


class SearchSimilarByEmbeddingInput(BaseModel):
    """Input schema for similarity search tool."""

    query: str = Field(description="Query text to find similar news")
    top_k: int = Field(default=5, description="Number of similar results to return")


class SearchSimilarByEmbeddingTool:
    """Tool for searching similar news posts using vector embeddings."""

    name = "search_similar"
    description = "Search for Hacker News posts similar to a query using semantic search with embeddings."

    def __init__(
        self,
        repository: Optional[NewsRepository] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
    ):
        """
        Initialize similarity search tool.

        Args:
            repository: News repository instance. If None, creates new instance.
            embeddings: Embeddings instance. If None, creates new instance.
        """
        self.repository = repository or NewsRepository()
        self.embeddings = embeddings or OllamaEmbeddings()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search similar posts using vector similarity.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            list[dict] with search results.
        """
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)

        # Perform vector search
        results = self.repository.wires_vector_search(query_embedding, top_k)

        if not results:
            return []

        output = []
        for i, post in enumerate(results, 1):
            output.append(
                {
                    #"rank": i,
                    "id": post.get("post_id"),
                    "title": post.get("title"),
                    "author": post.get("author"),
                    "url": post.get("url"),
                    "similarity": post.get("similarity"),
                    "summary": post.get("summary")
                }
            )

        return output

    def as_tool(self) -> StructuredTool:
        """Convert to LangChain StructuredTool."""
        return StructuredTool.from_function(
            func=self.search,
            name=self.name,
            description=self.description,
            args_schema=SearchSimilarByEmbeddingInput,
        )
