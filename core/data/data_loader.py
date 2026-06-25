"""
Simple one-line data loader with incremental caching.

Usage:
    from core.data.data_loader import load_crypto_data

    df = load_crypto_data(
        symbol="BTCUSDT",
        start_date="2020-01-01",
        interval="4h"
    )
"""

from datetime import datetime, timezone, timezone
from typing import Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.data.data_sources import DataSourceFactory, BINANCE_FUTURES, BYBIT_FUTURES
from core.data.multi_source_loader import SmartCacheLoader


def load_crypto_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    interval: str = "4h",
    source: str = "bybit_futures",
    cache_dir: str = "data/cache",
    use_cache: bool = True
) -> pd.DataFrame:
    """Load complete crypto data in one line with incremental caching.

    Automatically fetches and merges:
    - Klines (OHLCV)
    - Funding rates
    - Open interest

    Incremental caching:
    - First call: downloads all data
    - Subsequent calls: only downloads missing dates
    - Data is cached per-year for efficient updates

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD" (default: now)
        interval: Candle interval ("1h", "4h", "1d", etc.)
        source: Data source ("bybit_futures" or "binance_futures")
        cache_dir: Cache directory
        use_cache: Use cached data (default: True)

    Returns:
        DataFrame ready for feature generator:
        - timestamp, open, high, low, close, volume
        - fundingRate (ffilled)
        - sumOpenInterest (ffilled)
    """
    if use_cache:
        # Use SmartCacheLoader with incremental updates
        loader = SmartCacheLoader(
            cache_dir=cache_dir,
            primary_source=source,
            oi_source=source  # Use same source for OI
        )

        df = loader.download(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            fetch_funding=True,
            fetch_oi=True,
            force_refresh=False
        )

        return df
    else:
        # Direct download without cache
        src = DataSourceFactory.create(source, symbol)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else datetime.now(timezone.utc)

        print(f"📥 Loading {symbol} from {start_date}...")

        # Parallel load with ThreadPoolExecutor
        def load_klines():
            return src.get_klines(
                interval=interval,
                start_time=start_dt,
                end_time=end_dt,
                limit=1000
            )

        def load_funding():
            return src.get_funding_rate(
                start_time=start_dt,
                end_time=end_dt,
                limit=1000
            )

        def load_oi():
            return src.get_open_interest(
                interval=interval,
                start_time=start_dt,
                end_time=end_dt,
                limit=200
            )

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(load_klines): "klines",
                executor.submit(load_funding): "funding",
                executor.submit(load_oi): "oi"
            }

            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    results[data_type] = future.result()
                except Exception as e:
                    print(f"❌ Error loading {data_type}: {e}")
                    results[data_type] = pd.DataFrame()

        df_klines = results.get("klines", pd.DataFrame())
        df_funding = results.get("funding", pd.DataFrame())
        df_oi = results.get("oi", pd.DataFrame())

        if df_klines.empty:
            print(f"❌ No klines data")
            return pd.DataFrame()

        # Merge
        result = df_klines

        if not df_funding.empty:
            result = pd.merge_asof(
                result.sort_values("timestamp"),
                df_funding.sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )

        if not df_oi.empty:
            result = pd.merge_asof(
                result.sort_values("timestamp"),
                df_oi.sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )

        # Forward fill
        if "fundingRate" in result.columns:
            result["fundingRate"] = result["fundingRate"].ffill().fillna(0)

        if "sumOpenInterest" in result.columns:
            result["sumOpenInterest"] = result["sumOpenInterest"].ffill()

        result = result.sort_values("timestamp").reset_index(drop=True)

        print(f"✅ Loaded {len(result)} records")
        return result


def load_multi_crypto_data(
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    interval: str = "4h",
    source: str = "bybit_futures",
    cache_dir: str = "data/cache",
    use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    """Load data for multiple crypto symbols.
    
    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD" (default: now)
        interval: Candle interval ("1h", "4h", "1d", etc.)
        source: Data source ("bybit_futures" or "binance_futures")
        cache_dir: Cache directory
        use_cache: Use cached data (default: True)
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    results = {}
    for sym in symbols:
        print(f"\n{'='*40}")
        print(f"🔄 Processing {sym}...")
        print(f"{'='*40}")
        
        df = load_crypto_data(
            symbol=sym,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            source=source,
            cache_dir=cache_dir,
            use_cache=use_cache
        )
        if not df.empty:
            results[sym] = df
            
    return results
