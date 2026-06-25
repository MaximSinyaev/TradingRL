"""
Multi-source data loader with smart caching.

Features:
- Multiple exchange support (Binance, Bybit, etc.)
- Incremental updates (only download missing dates)
- DuckDB metadata for fast date range queries
- Per-symbol caching with automatic deduplication
"""

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal
from pathlib import Path
import duckdb

from core.data.data_sources import DataSource, DataSourceFactory, BINANCE_FUTURES, BYBIT_FUTURES


class SmartCacheLoader:
    """Loader with smart caching and incremental updates.

    Cache structure:
    cache_dir/
    ├── metadata.db              # DuckDB with date ranges
    └── data/                    # Parquet files by config hash
        └── {hash}/
            ├── BTCUSDT_2020.parquet.gz
            ├── BTCUSDT_2021.parquet.gz
            └── ...
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        primary_source: str = BINANCE_FUTURES,
        oi_source: Optional[str] = None,  # Alternative source for OI data
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.db_path = self.cache_dir / "metadata.db"
        self.primary_source = primary_source
        self.oi_source = oi_source or primary_source

        # Initialize metadata DB
        self._init_db()

    def _init_db(self):
        """Initialize DuckDB metadata tables."""
        con = duckdb.connect(str(self.db_path))

        # Data inventory table
        con.execute("""
            CREATE TABLE IF NOT EXISTS data_inventory (
                config_hash TEXT PRIMARY KEY,
                symbol TEXT,
                source_type TEXT,
                interval TEXT,
                fetch_funding INTEGER,
                fetch_oi INTEGER,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # Date ranges table
        con.execute("""
            CREATE TABLE IF NOT EXISTS date_ranges (
                config_hash TEXT,
                symbol TEXT,
                year INTEGER,
                file_path TEXT,
                min_date DATE,
                max_date DATE,
                record_count INTEGER,
                PRIMARY KEY (config_hash, symbol, year)
            )
        """)

        con.close()

    def _get_config_hash(self, symbol: str, source: str, interval: str,
                        fetch_funding: bool, fetch_oi: bool) -> str:
        """Generate hash for configuration."""
        import hashlib
        config_str = f"{symbol}_{source}_{interval}_{fetch_funding}_{fetch_oi}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def get_existing_dates(self, symbol: str, source: str, interval: str,
                          fetch_funding: bool, fetch_oi: bool) -> set:
        """Query existing date ranges from metadata.

        Returns:
            Set of dates that already exist in cache
        """
        config_hash = self._get_config_hash(symbol, source, interval, fetch_funding, fetch_oi)

        con = duckdb.connect(str(self.db_path))
        result = con.execute("""
            SELECT DISTINCT min_date, max_date
            FROM date_ranges
            WHERE config_hash = ?
        """, [config_hash]).fetchall()

        con.close()

        if not result:
            return set()

        # Convert date ranges to set of dates
        dates = set()
        for min_date, max_date in result:
            if min_date and max_date:
                current = min_date
                while current <= max_date:
                    dates.add(current)
                    current += timedelta(days=1)

        return dates

    def download(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        interval: str = "4h",
        fetch_funding: bool = True,
        fetch_oi: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Download data with smart caching.

        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to now
            interval: Candle interval
            fetch_funding: Include funding rates
            fetch_oi: Include open interest
            force_refresh: Ignore cache and re-download

        Returns:
            Combined DataFrame with all data
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else datetime.now(timezone.utc)

        # Check what we already have
        if not force_refresh:
            existing_dates = self.get_existing_dates(
                symbol, self.primary_source, interval, fetch_funding, fetch_oi
            )
        else:
            existing_dates = set()

        # Determine what dates we need
        needed_dates = set()
        current = start.date()
        while current <= end.date():
            if current not in existing_dates:
                needed_dates.add(current)
            current += timedelta(days=1)

        if not needed_dates:
            print("✅ All data in cache, loading from disk...")
            df = self._load_cached_data(
                symbol, self.primary_source, interval, fetch_funding, fetch_oi
            )
            if not df.empty:
                df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
            return df

        print(f"📥 Need to download {len(needed_dates)} missing dates")

        # Download missing data
        primary_src = DataSourceFactory.create(self.primary_source, symbol)
        oi_src = DataSourceFactory.create(self.oi_source, symbol)

        # Download by year for better organization
        years_to_fetch = sorted({d.year for d in needed_dates})
        dfs_by_year = {}

        for year in years_to_fetch:
            year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
            year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

            # Clip to actual requested range
            dl_start = max(year_start, start)
            dl_end = min(year_end, end)

            print(f"  📅 Downloading {year}...")

            df = self._download_range(
                primary_src, oi_src, dl_start, dl_end, interval, fetch_funding, fetch_oi
            )

            if not df.empty:
                dfs_by_year[year] = df

        # Load existing cached data and merge
        final_df = self._load_cached_data(
            symbol, self.primary_source, interval, fetch_funding, fetch_oi
        )

        for year, df in dfs_by_year.items():
            final_df = pd.concat([final_df, df], ignore_index=True)

        # Deduplicate and sort
        if not final_df.empty:
            final_df = final_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

            # Save by year (only new data to avoid overwriting unchanged cache)
            if dfs_by_year:
                new_data_df = pd.concat(dfs_by_year.values(), ignore_index=True)
                self._save_by_year(new_data_df, symbol, interval, fetch_funding, fetch_oi)

            # Update metadata
            self._update_metadata(final_df, symbol, interval, fetch_funding, fetch_oi)
            
            # Clip to requested range before returning
            final_df = final_df[(final_df["timestamp"] >= start) & (final_df["timestamp"] <= end)]

        return final_df

    def _download_range(
        self,
        primary_src: DataSource,
        oi_src: DataSource,
        start: datetime,
        end: datetime,
        interval: str,
        fetch_funding: bool,
        fetch_oi: bool,
    ) -> pd.DataFrame:
        """Download data for a specific date range."""
        try:
            result_df = primary_src.get_klines(
                interval=interval,
                start_time=start,
                end_time=end,
                limit=1000
            )
        except Exception as e:
            print(f"❌ Error downloading klines: {e}")
            result_df = pd.DataFrame()

        # Fetch funding rates
        if fetch_funding and primary_src.has_funding_data():
            print(f"    📊 Downloading funding rates...")
            df_funding = self._download_until_complete(
                primary_src.get_funding_rate,
                start, end
            )

            if not df_funding.empty:
                result_df = pd.merge_asof(
                    result_df.sort_values("timestamp"),
                    df_funding.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward"
                )

        # Fetch OI (potentially from different source)
        if fetch_oi and oi_src.has_oi_data():
            if oi_src.config.name != primary_src.config.name:
                print(f"    📊 Downloading OI from {oi_src.config.name}...")

            df_oi = self._download_until_complete(
                lambda st, et: oi_src.get_open_interest(interval, st, et),
                start, end
            )

            if not df_oi.empty:
                result_df = pd.merge_asof(
                    result_df.sort_values("timestamp"),
                    df_oi.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward"
                )

        # Forward fill NaNs
        if fetch_funding and "fundingRate" in result_df.columns:
            result_df["fundingRate"] = result_df["fundingRate"].ffill().fillna(0)

        if fetch_oi and "sumOpenInterest" in result_df.columns:
            result_df["sumOpenInterest"] = result_df["sumOpenInterest"].ffill()

        return result_df

    def _download_until_complete(self, fetch_fn, start: datetime, end: datetime) -> pd.DataFrame:
        """Download data (underlying source handles pagination)."""
        try:
            df = fetch_fn(start, end)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return pd.DataFrame().drop_duplicates(subset=["timestamp"])

    def _load_cached_data(
        self, symbol: str, source: str, interval: str,
        fetch_funding: bool, fetch_oi: bool
    ) -> pd.DataFrame:
        """Load all cached data for a configuration."""
        config_hash = self._get_config_hash(symbol, source, interval, fetch_funding, fetch_oi)

        cache_path = self.data_dir / config_hash
        if not cache_path.exists():
            return pd.DataFrame()

        parquet_files = list(cache_path.glob("*.parquet.gz"))
        if not parquet_files:
            return pd.DataFrame()

        dfs = []
        for f in sorted(parquet_files):
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                print(f"⚠️  Error reading {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _save_by_year(
        self, df: pd.DataFrame, symbol: str, interval: str,
        fetch_funding: bool, fetch_oi: bool
    ):
        """Save data partitioned by year."""
        config_hash = self._get_config_hash(
            symbol, self.primary_source, interval, fetch_funding, fetch_oi
        )

        cache_path = self.data_dir / config_hash
        cache_path.mkdir(exist_ok=True)

        # Extract year from timestamp
        df["year"] = pd.to_datetime(df["timestamp"]).dt.year

        for year, year_df in df.groupby("year"):
            file_path = cache_path / f"{symbol}_{year}.parquet.gz"

            # Load existing and merge
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                merged = pd.concat([existing, year_df]).drop_duplicates(subset=["timestamp"])
            else:
                merged = year_df

            merged = merged.sort_values("timestamp").drop(columns=["year"])
            merged.to_parquet(file_path, index=False, compression="gzip")
            print(f"    💾 Saved {year}: {file_path.name}")

    def _update_metadata(
        self, df: pd.DataFrame, symbol: str, interval: str,
        fetch_funding: bool, fetch_oi: bool
    ):
        """Update metadata DB with current data."""
        config_hash = self._get_config_hash(
            symbol, self.primary_source, interval, fetch_funding, fetch_oi
        )

        con = duckdb.connect(str(self.db_path))

        # Update config record
        now = datetime.now(timezone.utc)
        con.execute("""
            INSERT OR REPLACE INTO data_inventory
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            config_hash, symbol, self.primary_source, interval,
            int(fetch_funding), int(fetch_oi), now, now
        ])

        # Update per-year records
        df["year"] = pd.to_datetime(df["timestamp"]).dt.year
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        for year, year_df in df.groupby("year"):
            file_path = self.data_dir / config_hash / f"{symbol}_{year}.parquet.gz"

            min_date = year_df["date"].min()
            max_date = year_df["date"].max()
            count = len(year_df)

            con.execute("""
                INSERT OR REPLACE INTO date_ranges
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [config_hash, symbol, year, str(file_path), min_date, max_date, count])

        con.close()

    def get_stats(self) -> pd.DataFrame:
        """Get cache statistics."""
        con = duckdb.connect(str(self.db_path))
        stats = con.execute("""
            SELECT
                di.symbol,
                di.source_type,
                di.interval,
                SUM(dr.record_count) as total_records,
                MIN(dr.min_date) as from_date,
                MAX(dr.max_date) as to_date
            FROM data_inventory di
            JOIN date_ranges dr ON di.config_hash = dr.config_hash
            GROUP BY di.symbol, di.source_type, di.interval
            ORDER BY di.symbol, di.source_type
        """).fetchdf()
        con.close()

        return stats


# Legacy compatibility wrapper
class MultiSourceDataLoader:
    """Wrapper for legacy compatibility."""

    def __init__(
        self,
        symbols: list[str],
        primary_source: str = BINANCE_FUTURES,
        oi_source: Optional[str] = None,
        cache_dir: str = "data/cache"
    ):
        self.symbols = [s.upper() for s in symbols]
        self.loader = SmartCacheLoader(
            cache_dir=cache_dir,
            primary_source=primary_source,
            oi_source=oi_source
        )

    def download_all(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "4h",
        fetch_funding: bool = True,
        fetch_oi: bool = True
    ) -> dict[str, pd.DataFrame]:
        """Download all symbols."""
        results = {}
        for symbol in self.symbols:
            print(f"\n{'='*60}")
            print(f"Processing {symbol}...")
            print(f"{'='*60}")
            results[symbol] = self.loader.download(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                fetch_funding=fetch_funding,
                fetch_oi=fetch_oi
            )
        return results
