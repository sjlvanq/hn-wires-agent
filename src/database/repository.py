from typing import Optional
import math

from config.settings import settings
from database.connection import DatabaseConnection

import logging
logger = logging.getLogger(__name__)

class NewsRepositoryException(Exception):
    """Custom exception for NewsRepository errors."""
    pass

class NewsRepository:
    """Repository for accessing news data from SQLite database."""

    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """Create a new :class:`NewsRepository`.

        Parameters
        ----------
        db_connection:
            Existing :class:`DatabaseConnection` instance.  If omitted a new
            connection will be established internally.  This allows callers to
            share a single database connection across several repositories.
        """
        self.db = db_connection or DatabaseConnection()

    def _get_post_by_id(self, post_id: int) -> Optional[dict]:
        """Return a post by its numeric ``id``.

        Parameters
        ----------
        post_id: int
            The id of the post to look up.

        Returns
        -------
        Optional[dict]
            A dictionary containing the post fields – ``id``, ``author`` and
            ``title`` – or ``None`` if the id does not exist.
        """
        query = """
            SELECT id, author, descendants, score, time, title, text, url
            FROM posts
            WHERE id = ?
        """
        try:
            results = self.db.execute_query(query, (post_id,))
        except Exception as e:
            logger.error(f"Error occurred while fetching post by ID {post_id}: {e}")
            raise NewsRepositoryException("Error occurred while fetching post by ID") from e

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
        """Return the ``wires`` record attached to a post.

        Parameters
        ----------
        post_id: int
            Identifier of the post whose wire is requested.

        Returns
        -------
        Optional[dict]
            ``None`` if the post has no associated wire; otherwise a mapping
            containing ``id``, ``post_id``, ``topic`` and ``summary``.
        """
        query = """
            SELECT id, post_id, topic, summary
            FROM wires
            WHERE post_id = ?
        """
        try:
            results = self.db.execute_query(query, (post_id,))
        except Exception as e:
            logger.error(f"Error occurred while fetching wire by post ID {post_id}: {e}")
            raise NewsRepositoryException("Error occurred while fetching wire by post ID") from e

        if results:
            row = results[0]
            return {
                "id": row["id"],
                "post_id": row["post_id"],
                "topic": row["topic"],
                "summary": row["summary"],
            }
        return None

    def _add_similarity_scores(self, results: list[dict]) -> list[dict]:
        """Add an exponential *similarity* field to result rows.

        The ``sqlite-vec`` module returns a ``distance`` value that decreases as
        vectors become more similar.  To make the value more intuitive we
        convert it with an exponential decay function: ``similarity =
        exp(-distance)``.

        Parameters
        ----------
        results: list[dict]
            Query results that each contain a ``distance`` key.

        Returns
        -------
        list[dict]
            The same list of dictionaries, each augmented with a ``similarity``
            key.
        """
        normalized = [dict(r) for r in results]
        for row in normalized:
            distance = row.get("distance", 0)
            # Convert raw distance to a similarity score with exponential decay.
            similarity = math.exp(-distance)
            row["similarity"] = similarity
        return normalized


    def get_post_with_wire(self, post_id: int) -> Optional[dict]:
        """Return a post enriched with wire information when available.

        Parameters
        ----------
        post_id: int
            Identifier of the post.

        Returns
        -------
        Optional[dict]
            ``None`` if the post does not exist.  Otherwise a dictionary with
            all post columns and an optional ``wire`` key containing the
            wire summary.
        """
        try:
            post = self._get_post_by_id(post_id)
        except Exception as e:
            logger.error(f"Error occurred while fetching post by ID {post_id}: {e}")
            raise NewsRepositoryException("Error occurred while fetching post by ID") from e

        if not post:
            return None

        try:
            wire = self._get_wire_by_post_id(post_id)
        except Exception as e:
            logger.error(f"Error occurred while fetching wire by post ID {post_id}: {e}")
            raise NewsRepositoryException("Error occurred while fetching wire by post ID") from e
        
        if wire:
            post["wire"] = wire.get('summary')
        
        return post

    def keywords_vector_search(self, keyword_id: int, top_k: Optional[int] = None) -> list[dict]:
        """Search posts that are close to the vector of ``keyword_id``.

        Parameters
        ----------
        keyword_id: int
            Identifier of a keyword row whose embedding is used as a query.
        top_k: int | None
            Number of top‑ranked results to return.  ``None`` defaults to
            :pyattr:`settings.wires_vector_search_top_k`.

        Returns
        -------
        list[dict]
            A list of matching post records augmented with a ``similarity``
            field.  Each item contains the post data, the matched keyword and
            the full set of keywords for that post.
        """
        if top_k is None:
            top_k = settings.wires_vector_search_top_k

        query = f"""
            SELECT
                dvk.post_id,
                dvk.keyword AS matched_keyword,
                dvk.distance,
                p.author,
                p.time,
                p.title,
                p.url,
                (
                    SELECT GROUP_CONCAT(keyword, ', ')
                    FROM keywords
                    WHERE post_id = dvk.post_id
                ) AS all_post_keywords
            FROM (
                SELECT
                    v.rowid AS keyword_id,
                    v.distance,
                    k.post_id,
                    k.keyword
                FROM vec_keywords v
                INNER JOIN keywords k ON v.rowid = k.id
                WHERE v.k = ?
                AND v.embedding MATCH (SELECT embedding FROM vec_keywords WHERE rowid = ?)
            ) dvk
            INNER JOIN posts p ON dvk.post_id = p.id
            ORDER BY dvk.distance;
        """

        try:
            results = self.db.execute_query(query, (top_k, keyword_id))
        except Exception as e:
            logger.error(f"Error executing wires vector search query: {e}")
            raise NewsRepositoryException("Error occurred while executing wires vector search") from e

        if not results:
            return []

        # DEBUG
        logger.debug(f"keywords_vector_search db.execute_query:")
        for row in results: logger.debug(dict(row))

        keywords = self._add_similarity_scores(results)
        return keywords

    def wires_vector_search(self, embedding: list[float], top_k: Optional[int] = None) -> list[dict]:
        """Search *wires* on a user supplied embedding.

        Parameters
        ----------
        embedding: list[float]
            The query vector to compare against the ``vector`` column in
            ``vec_wires``.
        top_k: int | None
            Maximum number of results to return; ``None`` falls back to
            :pyattr:`settings.wires_vector_search_top_k`.

        Returns
        -------
        list[dict]
            Posts that match the embedding ranked by distance, each augmented
            with a ``similarity`` key derived from the exponential decay of
            distance.
        """
        if len(embedding) != settings.vector_search_vector_k:
            raise ValueError(f"Embedding vector must have length {settings.vector_search_vector_k}, got {len(embedding)}")

        if top_k is None:
            top_k = settings.wires_vector_search_top_k

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
            logger.error(f"Error executing wires vector search query: {e}")
            raise NewsRepositoryException("Error occurred while executing wires vector search") from e

        if not results:
            return []
        
        # DEBUG
        logger.debug(f"wires_vector_search db.execute_query:")
        for row in results: logger.debug(dict(row))

        posts = self._add_similarity_scores(results)

        return posts

    def get_post_keywords(self, post_id: int) -> list[dict]:
        """Return the list of keywords attached to a post.

        Parameters
        ----------
        post_id: int
            Identifier of the post whose keywords we want.

        Returns
        -------
        list[dict]
            Each dictionary contains the primary key ``id`` and the keyword
            string under the key ``keyword``.  The list may be empty if the
            post has no keywords.
        """
        query = """
            SELECT id, keyword
            FROM keywords
            WHERE post_id = ?
        """
        try:
            results = self.db.execute_query(query, (post_id,))
        except Exception as e:
            logger.error(f"Error occurred while fetching keywords for post ID {post_id}: {e}")
            raise NewsRepositoryException("Error occurred while fetching keywords for post ID") from e
        if results:
            return [{"id": row["id"], "keyword": row["keyword"]} for row in results]
        return []
