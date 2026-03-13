from __future__ import annotations

from collections.abc import Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool

from config import get_settings

settings = get_settings()

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=settings.database_url,
        )
    return _pool


def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
