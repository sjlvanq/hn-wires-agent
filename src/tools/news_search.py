from typing import Optional
import warnings

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from database import NewsRepository
from embeddings import OllamaEmbeddings


class SearchSimilarByEmbeddingInput(BaseModel):
    """Pydantic model describing the arguments for the embedding search tool.
    """

    query: str = Field(description="Query text to find similar news")
    top_k: int = Field(default=3, description="Number of similar results to return")


class SearchSimilarByEmbeddingTool:
    """Encapsulates vector‑based semantic search over news posts.
    """

    name = "search_similar"
    description = "Search for Hacker News posts similar to a query using semantic search with embeddings."

    def __init__(
        self,
        repository: Optional[NewsRepository] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
    ):
        """Create a new :class:`SearchSimilarByEmbeddingTool`.

        Parameters
        ----------
        repository : NewsRepository | None
            Repository for accessing stored news.  Defaults to a new instance.
        embeddings : OllamaEmbeddings | None
            Embedding generator.  Defaults to a new instance.
        """
        self.repository = repository or NewsRepository()
        self.embeddings = embeddings or OllamaEmbeddings()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return posts similar to *query*.

        Parameters
        ----------
        query : str
            Text whose semantic similarity to stored posts is evaluated.
        top_k : int, default 3
            Number of top results to return.

        Returns
        -------
        list[dict]
            List of dictionaries describing each matching post.
        """
        # Deprecation notice: prefer calling the StructuredTool via `as_tool().run(...)`
        warnings.warn(
            "SearchSimilarByEmbeddingTool.search() is deprecated; use as_tool().run({'query':..., 'top_k':...}) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        """Exposed as a :class:`StructuredTool` for LangChain adapters.
        """
        return StructuredTool.from_function(
            func=self.search,
            name=self.name,
            description=self.description,
            args_schema=SearchSimilarByEmbeddingInput,
        )


class SearchSimilarByKeywordInput(BaseModel):
    """Pydantic schema for the keyword‑based similarity search tool.
    """

    keyword_id: int = Field(description="ID of the keyword to use for similarity search")
    top_k: int = Field(default=5, description="Number of similar results to return")


class SearchSimilarByKeywordTool:
    """Search for posts using the vector similarity of keyword embeddings.
    """

    name = "search_similar_by_keyword"
    description = "Search for Hacker News posts similar to a keyword using vector similarity search over keywords."

    def __init__(
        self,
        repository: Optional[NewsRepository] = None,
    ):
        """Create a new :class:`SearchSimilarByKeywordTool`.

        Parameters
        ----------
        repository : NewsRepository | None
            Repository for accessing news data.  Defaults to a new instance.
        """
        self.repository = repository or NewsRepository()

    def search(self, keyword_id: int, top_k: int = 5) -> list[dict]:
        """Return posts similar to *keyword_id*.

        Parameters
        ----------
        keyword_id : int
            Identifier of the keyword to query.
        top_k : int, default 5
            Number of results to return.

        Returns
        -------
        list[dict]
            List of dictionaries describing each matching post.
        """
        # Deprecation notice: prefer calling the StructuredTool via `as_tool().run(...)`
        warnings.warn(
            "SearchSimilarByKeywordTool.search() is deprecated; use as_tool().run({'keyword_id':..., 'top_k':...}) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Perform vector search using keyword_id
        results = self.repository.keywords_vector_search(keyword_id, top_k)

        if not results:
            return []

        output = []
        for i, post in enumerate(results, 1):
            output.append(
                {
                    "id": post.get("post_id"),
                    "title": post.get("title"),
                    "author": post.get("author"),
                    "url": post.get("url"),
                    "similarity": post.get("similarity"),
                    "matched_keyword": post.get("matched_keyword"),
                    "all_post_keywords": post.get("all_post_keywords")
                }
            )

        return output

    def as_tool(self) -> StructuredTool:
        """Expose as a :class:`StructuredTool` for LangChain integration.
        """
        return StructuredTool.from_function(
            func=self.search,
            name=self.name,
            description=self.description,
            args_schema=SearchSimilarByKeywordInput,
        )
