import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from core.data.data_sources import DataSource, DataSourceConfig

class BinanceFuturesSource(DataSource):
    """Binance USDT-Margined Futures data source.

    Endpoints:
    - Klines: /fapi/v1/klines
    - Funding: /fapi/v1/fundingRate
    - OI: /futures/data/openInterestHist (30 days only!)
    """

    # Supported intervals for each endpoint
    KLINE_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
                      '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    OI_INTERVALS = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']

    def _validate_oi_interval(self, interval: str) -> bool:
        """Check if interval is valid for OI endpoint."""
        return interval in self.OI_INTERVALS

    def __init__(self, symbol: str):
        config = DataSourceConfig(
            name="binance_futures",
            base_url="https://fapi.binance.com",
            rate_limit_delay=0.3,
            klines_endpoint="/fapi/v1/klines",
            funding_endpoint="/fapi/v1/fundingRate",
            oi_endpoint="/futures/data/openInterestHist"
        )
        super().__init__(config, symbol)

    def get_klines(
        self,
        interval: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        if interval not in self.KLINE_INTERVALS:
            raise ValueError(f"Invalid interval {interval}. Valid: {self.KLINE_INTERVALS}")

        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = self._make_request(self.config.klines_endpoint, params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        numeric_columns = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_volume",
            "taker_buy_quote_volume"
        ]
        df[numeric_columns] = df[numeric_columns].astype(float)
        df["num_trades"] = df["num_trades"].astype(int)
        df.drop(columns=["ignore"], inplace=True)

        return df

    def get_funding_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        params = {
            "symbol": self.symbol,
            "limit": limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = self._make_request(self.config.funding_endpoint, params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "fundingRate"]]

        return df

    def get_open_interest(
        self,
        interval: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        if not self._validate_oi_interval(interval):
            raise ValueError(f"Invalid OI interval {interval}. Valid: {self.OI_INTERVALS}")

        params = {
            "symbol": self.symbol,
            "period": interval,  # OI uses 'period' not 'interval'
            "limit": limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = self._make_request(self.config.oi_endpoint, params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        df = df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]

        return df

