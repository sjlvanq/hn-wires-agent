from typing import Optional
import math

from config.settings import settings
from database.connection import DatabaseConnection

import logging
logger = logging.getLogger(__name__)

class NewsRepository:
    """Repository for accessing news data from SQLite database."""

    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """
        Initialize news repository.

        Args:
            db_connection: Database connection instance. If None, creates new instance.
        """
        self.db = db_connection or DatabaseConnection()

    def _get_post_by_id(self, post_id: int) -> Optional[dict]:
        """
        Retrieve a post by its ID (internal method).

        Args:
            post_id: The post ID.

        Returns:
            Dictionary with post data or None if not found.
        """
        query = """
            SELECT id, author, descendants, score, time, title, text, url
            FROM posts
            WHERE id = ?
        """
        results = self.db.execute_query(query, (post_id,))
        if results:
            row = results[0]
            return {
                "id": row["id"],
                "author": row["author"],
                "descendants": row["descendants"],
                "score": row["score"],
                "time": row["time"],
                "title": row["title"],
                "text": row["text"],
                "url": row["url"],
            }
        return None

    def _get_wire_by_post_id(self, post_id: int) -> Optional[dict]:
        """
        Retrieve wire data for a post (internal method).

        Args:
            post_id: The post ID.

        Returns:
            Dictionary with wire data or None if not found.
        """
        query = """
            SELECT id, post_id, topic, summary
            FROM wires
            WHERE post_id = ?
        """
        results = self.db.execute_query(query, (post_id,))
        if results:
            row = results[0]
            return {
                "id": row["id"],
                "post_id": row["post_id"],
                "topic": row["topic"],
                "summary": row["summary"],
            }
        return None

    def get_post_with_wire(self, post_id: int) -> Optional[dict]:
        """
        Retrieve post with its associated wire data.

        Args:
            post_id: The post ID.

        Returns:
            Dictionary with post and wire data or None if not found.
        """
        post = self._get_post_by_id(post_id)
        if not post:
            return None

        wire = self._get_wire_by_post_id(post_id)
        if wire:
            post["wire"] = wire.get('summary')
        return post

    def vector_search(self, embedding: list[float], top_k: Optional[int] = None) -> list[dict]:
        """
        Perform vector similarity search using sqlite-vec.

        Args:
            embedding: Query embedding vector.
            top_k: Number of results to return. Defaults to settings.vector_search_top_k.

        Returns:
            List of matching posts with similarity scores.
        """

        if len(embedding) != settings.vector_search_vector_k:
            raise ValueError(f"Embedding vector must have length {settings.vector_search_vector_k}, got {len(embedding)}")

        if top_k is None:
            top_k = settings.vector_search_top_k

        # Convert embedding to string format for sqlite-vec
        embedding_str = ",".join(map(str, embedding))

        query = f"""
            SELECT
                v.rowid,
                v.distance,
                w.summary,
                w.post_id,
                p.author, p.descendants, p.score, p.time, p.title, p.text, p.url
            FROM vec_wires v
            INNER JOIN wires w ON v.rowid = w.id
            INNER JOIN posts p ON w.post_id = p.id
            WHERE w.summary != '' AND v.k = ? AND v.embedding MATCH ?
            ORDER BY v.distance
        """
        
        try:
            results = self.db.execute_query(query, (top_k, "["+embedding_str+"]"))
        except Exception as e:
            logger.error(f"Error executing vector search query: {e}")
            return []

        if not results:
            return []
        
        # DEBUG
        logger.debug(f"vector_search db.execute_query:")
        for row in results: logger.debug(dict(row))

        # Get full post data for each result
        posts = []
        for row in results:
            distance = row["distance"]
            # Use exponential decay for better similarity score scaling
            # Distance 0 -> similarity 1.0, Distance 1 -> similarity 0.37, Distance 2 -> similarity 0.14
            similarity = math.exp(-distance)
            post = dict(row)
            post["similarity"] = similarity
            posts.append(post)

        return posts

    def get_post_keywords(self, post_id: int) -> list[str]:
        """
        Retrieve keywords for a post.

        Args:
            post_id: The post ID.

        Returns:
            List of keywords.
        """
        query = """
            SELECT keyword
            FROM post_keywords
            WHERE post_id = ?
        """
        results = self.db.execute_query(query, (post_id,))
        return [row["keyword"] for row in results]
