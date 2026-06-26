import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = str(PROJECT_ROOT / "data" / "cache")


@dataclass
class MarketSource:
    name: str
    base_url: str
    klines_endpoint: str
    funding_endpoint: Optional[str] = None
    oi_endpoint: Optional[str] = None

SPOT_MARKET = MarketSource(
    name="spot",
    base_url="https://api.binance.com",
    klines_endpoint="/api/v3/klines"
)

FUTURES_MARKET = MarketSource(
    name="futures",
    base_url="https://fapi.binance.com",
    klines_endpoint="/fapi/v1/klines",
    funding_endpoint="/fapi/v1/fundingRate",
    oi_endpoint="/futures/data/openInterestHist"
)

class BinanceKlinesDownloader:
    """Downloader для данных с Binance (Spot & Futures).

    Поддерживает:
    - Кэширование в parquet.gz формате
    - Фьючерсы и Спот
    - Загрузку Funding Rate и Open Interest (для фьючерсов)
    """

    def __init__(self, symbol: str = "BTCUSDT", market: MarketSource = FUTURES_MARKET, cache_dir: str = DEFAULT_CACHE_DIR):
        self.symbol = symbol.upper()
        self.limit = 1000  # Binance max limit per request (for klines/funding)
        self.oi_limit = 500 # max limit for OI
        self.market = market
        self.cache_dir = cache_dir
        self.symbol_dir = os.path.join(cache_dir, self.symbol)
        os.makedirs(self.symbol_dir, exist_ok=True)

    def _get_klines(self, interval: str, start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> list:
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": self.limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        url = f"{self.market.base_url}{self.market.klines_endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _get_funding_rate(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> list:
        if not self.market.funding_endpoint:
            return []
            
        params = {
            "symbol": self.symbol,
            "limit": self.limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        url = f"{self.market.base_url}{self.market.funding_endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _get_open_interest(self, period: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> list:
        if not self.market.oi_endpoint:
            return []

        # Map interval to OI period. OI endpoint supports: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
        params = {
            "symbol": self.symbol,
            "period": period,
            "limit": self.oi_limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        url = f"{self.market.base_url}{self.market.oi_endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _get_cache_path(self, start_date: str, end_date: Optional[str], interval: str, fetch_funding: bool, fetch_oi: bool) -> str:
        end_date_str = end_date if end_date else "now"
        market_str = self.market.name
        
        suffix = ""
        if fetch_funding: suffix += "_fund"
        if fetch_oi: suffix += "_oi"
        
        filename = f"{self.symbol}_{market_str}_{interval}_{start_date}_to_{end_date_str}{suffix}.parquet.gz"
        return os.path.join(self.symbol_dir, filename)

    def download(self, start_date: str, end_date: Optional[str] = None,
                 interval: str = "4h", fetch_funding: bool = True, fetch_oi: bool = True) -> pd.DataFrame:
        """Скачивает данные с Binance.

        Args:
            start_date: Дата начала в формате 'YYYY-MM-DD'
            end_date: Дата окончания в формате 'YYYY-MM-DD' (опционально)
            interval: Интервал свечей (по дефолту 4h)
            fetch_funding: Скачивать ли историю Funding Rate (только фьючерсы)
            fetch_oi: Скачивать ли историю Open Interest (только фьючерсы)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else datetime.now(timezone.utc)
        end_date_str = end_date if end_date else end.strftime("%Y-%m-%d")

        # Disable fetching extras if not futures
        if self.market.name == "spot":
            fetch_funding = False
            fetch_oi = False

        cache_path = self._get_cache_path(start_date, end_date_str, interval, fetch_funding, fetch_oi)

        if os.path.exists(cache_path):
            print(f"📦 Чтение из кэша: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"📥 Скачивание {self.symbol} [{self.market.name}] | {start_date} → {end_date_str}")
        
        df_klines = self._download_klines(start, end, interval)
        if df_klines.empty:
            return pd.DataFrame()

        result_df = df_klines

        if fetch_funding:
            print(f"📥 Скачивание Funding Rate...")
            df_funding = self._download_funding(start, end)
            if not df_funding.empty:
                # Merge logic: asof merge
                result_df = pd.merge_asof(
                    result_df.sort_values("timestamp"),
                    df_funding.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward"
                )
                
        if fetch_oi:
            print(f"📥 Скачивание Open Interest ({interval})...")
            oi_period = interval
            valid_oi_periods = ["5m","15m","30m","1h","2h","4h","6h","12h","1d"]
            if oi_period not in valid_oi_periods:
                print(f"⚠️  Интервал {interval} не поддерживается для OI. Пропуск.")
            else:
                df_oi = self._download_oi(start, end, oi_period)
                if not df_oi.empty:
                    result_df = pd.merge_asof(
                        result_df.sort_values("timestamp"),
                        df_oi.sort_values("timestamp"),
                        on="timestamp",
                        direction="backward"
                    )

        result_df = result_df.sort_values("timestamp").reset_index(drop=True)
        
        # FFill NaNs in funding and OI
        if fetch_funding and "fundingRate" in result_df.columns:
            result_df["fundingRate"] = result_df["fundingRate"].ffill().fillna(0)
        
        if fetch_oi and "sumOpenInterest" in result_df.columns:
             result_df["sumOpenInterest"] = result_df["sumOpenInterest"].ffill()
             if "sumOpenInterestValue" in result_df.columns:
                 result_df["sumOpenInterestValue"] = result_df["sumOpenInterestValue"].ffill()

        print(f"✅ Итого: {len(result_df):,} записей")
        result_df.to_parquet(cache_path, index=False, compression="gzip")
        print(f"💾 Сохранено: {cache_path}")

        return result_df

    def _download_klines(self, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        all_data = []
        current_start = start
        
        estimated_minutes = (end - start).total_seconds() / 60
        estimated_requests = max(1, int(estimated_minutes / 1000))

        with tqdm(total=estimated_requests, desc=f"Klines", unit="req") as pbar:
            while current_start < end:
                try:
                    data = self._get_klines(interval=interval, start_time=current_start)
                    if not data:
                        break

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

                    all_data.append(df)

                    last_time = df["timestamp"].iloc[-1]
                    current_start = last_time + timedelta(milliseconds=1)
                    pbar.update(1)
                    time.sleep(0.3)

                except Exception as e:
                    print(f"❌ Ошибка Klines: {e}")
                    break

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        return df.drop_duplicates(subset=["timestamp"])

    def _download_funding(self, start: datetime, end: datetime) -> pd.DataFrame:
        all_data = []
        current_start = start
        while current_start < end:
            try:
                data = self._get_funding_rate(start_time=current_start)
                if not data:
                    break
                
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
                df["fundingRate"] = df["fundingRate"].astype(float)
                df = df[["timestamp", "fundingRate"]]
                
                all_data.append(df)
                
                last_time = df["timestamp"].iloc[-1]
                if current_start == last_time + timedelta(milliseconds=1):
                    break
                current_start = last_time + timedelta(milliseconds=1)
                time.sleep(0.3)
            except Exception as e:
                print(f"❌ Ошибка Funding: {e}")
                break
                
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["timestamp"])

    def _download_oi(self, start: datetime, end: datetime, period: str) -> pd.DataFrame:
        all_data = []
        current_start = start
        while current_start < end:
            try:
                data = self._get_open_interest(period=period, start_time=current_start)
                if not data:
                    break
                
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
                df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
                df = df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]
                
                all_data.append(df)
                
                last_time = df["timestamp"].iloc[-1]
                if current_start == last_time + timedelta(milliseconds=1):
                    break
                current_start = last_time + timedelta(milliseconds=1)
                time.sleep(0.3)
            except Exception as e:
                print(f"❌ Ошибка OI: {e}")
                break
                
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["timestamp"])

class MultiSymbolDataLoader:
    """Загрузчик для нескольких символов одновременно."""

    def __init__(self, symbols: list[str], market: MarketSource = FUTURES_MARKET, cache_dir: str = DEFAULT_CACHE_DIR):
        self.symbols = [s.upper() for s in symbols]
        self.market = market
        self.downloaders = {
            symbol: BinanceKlinesDownloader(symbol, market, cache_dir)
            for symbol in self.symbols
        }

    def download_all(self, start_date: str, end_date: Optional[str] = None,
                     interval: str = "4h", fetch_funding: bool = True, fetch_oi: bool = True) -> dict[str, pd.DataFrame]:
        results = {}
        for symbol, downloader in self.downloaders.items():
            print(f"\n{'='*60}")
            results[symbol] = downloader.download(start_date, end_date, interval, fetch_funding, fetch_oi)
            print(f"{'='*60}\n")
        return results

    def get_combined_stats(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        stats = []
        for symbol, df in data.items():
            if not df.empty:
                stats.append({
                    "symbol": symbol,
                    "rows": len(df),
                    "start": df["timestamp"].min(),
                    "end": df["timestamp"].max(),
                    "missing_pct": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                })
        return pd.DataFrame(stats)
