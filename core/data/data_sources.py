"""
Data sources abstraction for crypto market data.

Provides unified interface for different exchanges (Binance, Bybit, etc.)
with support for:
- Klines (OHLCV)
- Funding rates
- Open interest
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import pandas as pd
import requests
import time



@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    base_url: str
    rate_limit_delay: float = 0.3  # seconds between requests

    # Optional endpoints (None if not supported)
    klines_endpoint: Optional[str] = None
    funding_endpoint: Optional[str] = None
    oi_endpoint: Optional[str] = None


class DataSource(ABC):
    """Abstract base class for crypto data sources.

    All data sources must implement these methods:
    - get_klines: Fetch OHLCV candlestick data
    - get_funding_rate: Fetch funding rate history (futures only)
    - get_open_interest: Fetch open interest history (futures only)
    """

    def __init__(self, config: DataSourceConfig, symbol: str):
        self.config = config
        self.symbol = symbol.upper()
        self._last_request_time = 0

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, params: dict) -> list:
        """Make HTTP request with rate limiting and error handling."""
        self._rate_limit()
        url = f"{self.config.base_url}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _chunked_request(
        self,
        request_fn: callable,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        chunk_days: int = 30
    ) -> pd.DataFrame:
        """Execute time-chunked requests for large date ranges.

        Many APIs limit records per request (e.g., 1000 for klines, 200 for funding).
        When requesting large ranges, APIs return only recent records.
        This helper chunks requests by time interval.

        Args:
            request_fn: Function that takes (start_time, end_time) and returns DataFrame
            start_time: Range start
            end_time: Range end
            chunk_days: Days per chunk (default 30)

        Returns:
            Combined DataFrame with all chunks
        """
        # No range or small range: single request
        if not start_time or not end_time:
            return request_fn(start_time, end_time)

        time_delta = end_time - start_time
        chunk_size = timedelta(days=chunk_days)

        if time_delta <= chunk_size:
            return request_fn(start_time, end_time)

        # Large range: chunk it
        all_data = []
        current_start = start_time

        while current_start < end_time:
            chunk_end = min(current_start + chunk_size, end_time)

            df_chunk = request_fn(current_start, chunk_end)

            if not df_chunk.empty:
                all_data.append(df_chunk)

            current_start = chunk_end

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return result

    @abstractmethod
    def get_klines(
        self,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV klines data."""
        pass

    @abstractmethod
    def get_funding_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch funding rate history (futures only)."""
        pass

    @abstractmethod
    def get_open_interest(
        self,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """Fetch open interest history (futures only)."""
        pass

    def has_funding_data(self) -> bool:
        """Check if this source supports funding rate data."""
        return self.config.funding_endpoint is not None

    def has_oi_data(self) -> bool:
        """Check if this source supports open interest data."""
        return self.config.oi_endpoint is not None



class DataSourceFactory:
    """Factory for creating data source instances."""

    _sources = {}

    @classmethod
    def create(cls, source: str, symbol: str) -> DataSource:
        source_lower = source.lower()
        if source_lower == BINANCE_FUTURES:
            from core.data.binance_source import BinanceFuturesSource
            return BinanceFuturesSource(symbol)
        elif source_lower == BYBIT_FUTURES:
            from core.data.bybit_source import BybitFuturesSource
            return BybitFuturesSource(symbol)
        else:
            raise ValueError(f"Unknown data source: {source}")

    @classmethod
    def register_source(cls, name: str, source_class: type):
        cls._sources[name.lower()] = source_class


# Convenience instances matching legacy API
BINANCE_FUTURES = "binance_futures"
BYBIT_FUTURES = "bybit_futures"

