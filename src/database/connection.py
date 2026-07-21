import sqlite3
import sqlite_vec

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from config.settings import settings


class DatabaseConnection:
    """SQLite connection wrapper that enables the ``sqlite_vec`` extension.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Create a new :class:`DatabaseConnection`.

        Parameters
        ----------
        db_path : Path | None
            Path to the SQLite database file.  If *None*, the path from
            :data:`settings.database_path_absolute` is used.
        """
        self.db_path = db_path or settings.database_path_absolute

    @contextmanager
    def get_connection(self):
        """Yield a database connection with the vector extension enabled.

        Yields
        ------
        sqlite3.Connection
            Connection object with the ``sqlite_vec`` extension loaded.
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
        """Run a ``SELECT`` query and return all fetched rows.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : tuple, optional
            Parameters passed to ``cursor.execute``.

        Returns
        -------
        list[sqlite3.Row]
            List of all rows returned by the query.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Run an ``INSERT``, ``UPDATE`` or ``DELETE`` statement.

        Parameters
        ----------
        query : str
            SQL statement to execute.
        params : tuple, optional
            Parameters passed to ``cursor.execute``.

        Returns
        -------
        int
            Number of rows affected by the statement.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount
