import sqlite3
import sqlite_vec

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from config.settings import settings


class DatabaseConnection:
    """Manages SQLite database connections with sqlite-vec support."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file. Defaults to settings.database_path_absolute.
        """
        self.db_path = db_path or settings.database_path_absolute

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection with vec extension loaded.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of result rows.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            Number of affected rows.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount
