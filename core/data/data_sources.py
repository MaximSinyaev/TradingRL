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
from tqdm import tqdm


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


class BybitFuturesSource(DataSource):
    """Bybit USDT Perpetual Futures data source.

    Endpoints:
    - Klines: /v5/market/kline
    - Funding: /v5/market/funding-history
    - OI: /v5/market/open-interest (historical available!)

    Bybit advantages:
    - OI historical data available (unlike Binance's 30-day limit)
    - More generous rate limits
    """

    # Bybit interval mapping
    INTERVAL_MAP = {
        '1': '1', '3': '3', '5': '5', '15': '15', '30': '30',
        '60': '60', '120': '120', '240': '240', '360': '360',
        '480': '480', '720': '720', 'D': 'D',
        'W': 'W', 'M': 'M'
    }

    # Reverse mapping for user-friendly intervals
    USER_INTERVALS = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240', '6h': '360',
        '12h': '480', '1d': 'D', '1w': 'W', '1M': 'M'
    }

    def __init__(self, symbol: str):
        # Bybit uses different symbol format (e.g., BTCUSDT -> BTCUSDT)
        config = DataSourceConfig(
            name="bybit_futures",
            base_url="https://api.bybit.com",
            rate_limit_delay=0.1,  # Bybit has more generous limits
            klines_endpoint="/v5/market/kline",
            funding_endpoint="/v5/market/funding/history",
            oi_endpoint="/v5/market/open-interest"
        )
        super().__init__(config, symbol)

    def _normalize_interval(self, interval: str) -> str:
        """Convert user-friendly interval to Bybit format."""
        norm = self.USER_INTERVALS.get(interval)
        if not norm:
            raise ValueError(f"Invalid interval {interval}. Valid: {list(self.USER_INTERVALS.keys())}")
        return norm

    def _chunked_time_request(
        self,
        endpoint: str,
        params_builder: callable,
        response_parser: callable,
        start_time: datetime,
        end_time: datetime,
        chunk_days: int = 60
    ) -> pd.DataFrame:
        all_data = []
        chunk_size = timedelta(days=chunk_days)
        current_start = start_time

        while current_start < end_time:
            chunk_end = min(current_start + chunk_size, end_time)

            params = params_builder(current_start, chunk_end)

            data = self._make_request(endpoint, params)

            if not data or data.get("retCode") != 0:
                break

            response_list = data.get("result", {}).get("list", [])
            if not response_list:
                current_start = chunk_end
                continue

            df = response_parser(response_list)

            if not df.empty:
                all_data.append(df)

            current_start = chunk_end

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return result

    def get_klines(
        self,
        interval: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        bybit_interval = self._normalize_interval(interval)
        all_data = []

        if not start_time:
            start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        if not end_time:
            end_time = datetime.now(timezone.utc)

        current_start = start_time
        request_limit = min(limit, 1000)

        # Calculate total hours for progress tracking
        total_hours = int((end_time - start_time).total_seconds() / 3600)

        with tqdm(total=total_hours, desc="Klines   ", position=0, leave=False, unit="h") as p_bar:
            while current_start < end_time:
                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "interval": bybit_interval,
                    "start": int(current_start.timestamp() * 1000),
                    "limit": request_limit
                }

                data = self._make_request(self.config.klines_endpoint, params)

                if not data or data.get("retCode") != 0:
                    break

                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break

                # Bybit returns newest first, reverse for chronological order
                klines.reverse()

                df = pd.DataFrame(klines, columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "turnover"
                ])

                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)

                numeric_columns = ["open", "high", "low", "close", "volume", "turnover"]
                df[numeric_columns] = df[numeric_columns].astype(float)

                # Add compatibility columns
                df["quote_asset_volume"] = df["turnover"]
                df["num_trades"] = None
                df["taker_buy_base_volume"] = None
                df["taker_buy_quote_volume"] = None

                # Filter to end_time
                df = df[df["timestamp"] <= end_time]

                if df.empty:
                    break

                all_data.append(df)

                # Update progress bar
                hours_loaded = int((df["timestamp"].iloc[-1] - start_time).total_seconds() / 3600)
                p_bar.update(hours_loaded - p_bar.n)

                # If got less than limit, we're done
                if len(klines) < request_limit:
                    break

                # Continue from last record + 1ms
                last_timestamp = df["timestamp"].iloc[-1]
                if last_timestamp <= current_start - timedelta(milliseconds=1):
                    # We didn't advance! Bybit returned older or same records
                    break
                current_start = last_timestamp + timedelta(milliseconds=1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return result

    def get_funding_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200
    ) -> pd.DataFrame:
        all_data = []

        if not start_time:
            start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        if not end_time:
            end_time = datetime.now(timezone.utc)

        # Use 60-day chunks to avoid API ignoring start parameter
        chunk_size = timedelta(days=60)
        current_start = start_time
        request_limit = min(limit, 200)

        # Calculate total days for progress tracking
        total_days = (end_time - start_time).days

        with tqdm(total=total_days, desc="Funding  ", position=1, leave=False, unit="d") as p_bar:
            while current_start < end_time:
                chunk_end = min(current_start + chunk_size, end_time)

                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "limit": request_limit,
                    "startTime": int(current_start.timestamp() * 1000),
                    "endTime": int(chunk_end.timestamp() * 1000)
                }

                data = self._make_request(self.config.funding_endpoint, params)

                if not data or data.get("retCode") != 0:
                    break

                funding_list = data.get("result", {}).get("list", [])
                if not funding_list:
                    current_start = chunk_end
                    p_bar.update((chunk_end - current_start).days)
                    continue

                funding_list.reverse()

                df = pd.DataFrame(funding_list)
                df["timestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
                df["fundingRate"] = df["fundingRate"].astype(float)
                df = df[["timestamp", "fundingRate"]]

                all_data.append(df)

                # Update progress
                days_loaded = (chunk_end - start_time).days
                p_bar.update(days_loaded - p_bar.n)

                # Move to next chunk
                current_start = chunk_end

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return result

    def get_open_interest(
        self,
        interval: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200
    ) -> pd.DataFrame:
        if interval not in self.USER_INTERVALS:
            raise ValueError(f"Invalid interval {interval}. Valid: {list(self.USER_INTERVALS.keys())}")

        all_data = []

        if not start_time:
            start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        if not end_time:
            end_time = datetime.now(timezone.utc)

        # Use 60-day chunks to avoid API ignoring start parameter
        chunk_size = timedelta(days=60)
        current_start = start_time
        request_limit = min(limit, 200)

        # Calculate total days for progress tracking
        total_days = (end_time - start_time).days

        with tqdm(total=total_days, desc="OI       ", position=2, leave=False, unit="d") as p_bar:
            while current_start < end_time:
                chunk_end = min(current_start + chunk_size, end_time)

                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "intervalTime": interval,
                    "limit": request_limit,
                    "startTime": int(current_start.timestamp() * 1000),
                    "endTime": int(chunk_end.timestamp() * 1000)
                }

                data = self._make_request(self.config.oi_endpoint, params)

                if not data or data.get("retCode") != 0:
                    break

                oi_list = data.get("result", {}).get("list", [])
                if not oi_list:
                    current_start = chunk_end
                    p_bar.update((chunk_end - current_start).days)
                    continue

                oi_list.reverse()

                df = pd.DataFrame(oi_list)
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
                df["sumOpenInterest"] = df["openInterest"].astype(float)
                df["sumOpenInterestValue"] = None
                df = df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]

                all_data.append(df)

                # Update progress
                days_loaded = (chunk_end - start_time).days
                p_bar.update(days_loaded - p_bar.n)

                # Move to next chunk
                current_start = chunk_end

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return result


class DataSourceFactory:
    """Factory for creating data source instances."""

    _sources = {
        "bybit_futures": BybitFuturesSource,
    }

    @classmethod
    def create(cls, source_type: str, symbol: str) -> DataSource:
        source_class = cls._sources.get(source_type.lower())
        if not source_class:
            raise ValueError(f"Unknown source type: {source_type}. Available: {list(cls._sources.keys())}")
        return source_class(symbol)

    @classmethod
    def register_source(cls, name: str, source_class: type):
        cls._sources[name.lower()] = source_class


# Convenience instances matching legacy API
BINANCE_FUTURES = "binance_futures"
BYBIT_FUTURES = "bybit_futures"

from core.data.binance_source import BinanceFuturesSource
DataSourceFactory.register_source("binance_futures", BinanceFuturesSource)
