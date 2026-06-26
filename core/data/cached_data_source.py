"""
Cached data source wrapper.

Provides automatic caching for any DataSource implementation.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import duckdb
import hashlib
import json
import time

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = str(PROJECT_ROOT / "data" / "cache")



class CachedDataSource:
    """Wrapper that adds caching to any DataSource.

    Cache structure:
    cache_dir/
    ├── metadata.db              # DuckDB with inventory
    └── data/                    # Parquet files by hash
        └── {method_hash}/
            ├── klines.parquet.gz
            ├── funding.parquet.gz
            └── oi.parquet.gz

    Usage:
        src = BinanceFuturesSource("BTCUSDT")
        cached_src = CachedDataSource(src, cache_dir=DEFAULT_CACHE_DIR)
        # Now all calls are cached
        df = cached_src.get_klines(interval="4h", ...)
    """

    def __init__(self, source, cache_dir: str = DEFAULT_CACHE_DIR):
        """Wrap a DataSource with caching.

        Args:
            source: DataSource instance (BinanceFuturesSource, etc.)
            cache_dir: Cache directory path
        """
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.db_path = self.cache_dir / "metadata.db"
        self._init_db()

    def _init_db(self):
        """Initialize DuckDB metadata."""
        try:
            con = duckdb.connect(str(self.db_path))
            con.execute("""
                CREATE TABLE IF NOT EXISTS cache_inventory (
                    query_hash TEXT PRIMARY KEY,
                    symbol TEXT,
                    method TEXT,
                    params TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP,
                    record_count INTEGER
                )
            """)
            con.close()
        except Exception:
            pass  # Metadata not critical, cache still works with parquet files

    def _get_query_hash(self, method: str, interval: str = None, **kwargs) -> str:
        """Generate hash for query parameters.

        Datetimes are truncated to interval boundary for consistent hashing.
        For example: interval='4h', time='15:45' → '12:00'
        """
        def truncate_dt(dt: datetime, interval: str) -> datetime:
            """Truncate datetime to interval boundary."""
            if not interval:
                # If no interval, truncate to hour
                return dt.replace(minute=0, second=0, microsecond=0)

            # Parse interval (e.g., '4h', '1h', '15m', '1d')
            interval = interval.lower()
            if interval.endswith('h'):
                hours = int(interval[:-1])
                # Truncate to hour boundary, then subtract remainder
                truncated = dt.replace(minute=0, second=0, microsecond=0)
                hour_offset = truncated.hour % hours
                truncated = truncated.replace(hour=truncated.hour - hour_offset)
                return truncated
            elif interval.endswith('m'):
                minutes = int(interval[:-1])
                truncated = dt.replace(second=0, microsecond=0)
                minute_offset = truncated.minute % minutes
                truncated = truncated.replace(minute=truncated.minute - minute_offset)
                return truncated
            elif interval.endswith('d'):
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                return dt.replace(minute=0, second=0, microsecond=0)

        # Normalize kwargs for consistent hashing
        normalized = []
        for k, v in sorted(kwargs.items()):
            if isinstance(v, datetime):
                # Truncate datetime to interval boundary
                v = truncate_dt(v, interval)
                v = v.isoformat()
            elif v is not None:
                v = str(v)
            normalized.append((k, v))
        params_str = f"{self.source.symbol}_{method}_{normalized}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, query_hash: str) -> Path:
        """Get cache file path for a query."""
        return self.data_dir / f"{query_hash}.parquet.gz"

    def _load_from_cache(self, query_hash: str) -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        cache_path = self._get_cache_path(query_hash)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            # Try to update last_used, but don't fail if locked
            try:
                con = duckdb.connect(str(self.db_path))
                con.execute("""
                    UPDATE cache_inventory
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE query_hash = ?
                """, [query_hash])
                con.close()
            except Exception:
                pass
            return df
        except Exception:
            return None

    def _save_to_cache(self, query_hash: str, df: pd.DataFrame, method: str, params: dict):
        """Save data to cache."""
        if df.empty:
            return

        cache_path = self._get_cache_path(query_hash)
        df.to_parquet(cache_path, index=False, compression="gzip")

        # Try to update metadata, but don't fail if locked
        try:
            con = duckdb.connect(str(self.db_path))
            con.execute("""
                INSERT OR REPLACE INTO cache_inventory
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, [query_hash, self.source.symbol, method, str(params), len(df)])
            con.close()
        except Exception:
            pass  # Metadata not critical

    def get_klines(
        self,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get klines with caching."""
        params = {"start": start_time, "end": end_time, "limit": limit}
        query_hash = self._get_query_hash("klines", interval=interval, **params)

        if use_cache:
            cached = self._load_from_cache(query_hash)
            if cached is not None:
                return cached

        df = self.source.get_klines(interval, start_time, end_time, limit)

        if use_cache and not df.empty:
            self._save_to_cache(query_hash, df, "klines", params)

        return df

    def get_funding_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get funding rate with caching."""
        params = {"start": start_time, "end": end_time, "limit": limit}
        query_hash = self._get_query_hash("funding", interval=None, **params)

        if use_cache:
            cached = self._load_from_cache(query_hash)
            if cached is not None:
                return cached

        df = self.source.get_funding_rate(start_time, end_time, limit)

        if use_cache and not df.empty:
            self._save_to_cache(query_hash, df, "funding", params)

        return df

    def get_open_interest(
        self,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get open interest with caching."""
        params = {"start": start_time, "end": end_time, "limit": limit}
        query_hash = self._get_query_hash("oi", interval=interval, **params)

        if use_cache:
            cached = self._load_from_cache(query_hash)
            if cached is not None:
                return cached

        df = self.source.get_open_interest(interval, start_time, end_time, limit)

        if use_cache and not df.empty:
            self._save_to_cache(query_hash, df, "oi", params)

        return df

    def has_funding_data(self) -> bool:
        """Proxy to source."""
        return self.source.has_funding_data()

    def has_oi_data(self) -> bool:
        """Proxy to source."""
        return self.source.has_oi_data()

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(exist_ok=True)

        # Re-init DB
        self._init_db()

    def get_cache_stats(self) -> pd.DataFrame:
        """Get cache statistics."""
        try:
            con = duckdb.connect(str(self.db_path))
            stats = con.execute("""
                SELECT
                    method,
                    symbol,
                    COUNT(*) as num_queries,
                    SUM(record_count) as total_records,
                    MAX(last_used) as last_used
                FROM cache_inventory
                GROUP BY method, symbol
            """).fetchdf()
            con.close()
            return stats
        except Exception:
            return pd.DataFrame()
