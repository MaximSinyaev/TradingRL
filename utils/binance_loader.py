import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone

class BinanceKlinesDownloader:
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, symbol="BTCUSDT", cache_dir="binance_data_cache"):
        self.symbol = symbol.upper()
        self.limit = 1000  # Binance max limit per request
        self.cache_dir = cache_dir
        self.symbol_dir = os.path.join(self.cache_dir, self.symbol)
        os.makedirs(self.symbol_dir, exist_ok=True)

    def _get_klines(self, interval, start_time=None, end_time=None):
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": self.limit
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def _get_cache_path(self, start_date: str, end_date: str, interval: str) -> str:
        filename = f"{self.symbol}_{interval}_{start_date}_to_{end_date}.parquet.gz"
        return os.path.join(self.symbol_dir, filename)

    def download(self, start_date: str, end_date: str = None, interval: str = "1m"):
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else datetime.now(timezone.utc)
        end_date_str = end_date if end_date else end.strftime("%Y-%m-%d")

        cache_path = self._get_cache_path(start_date, end_date_str, interval)

        if os.path.exists(cache_path):
            print(f"Чтение данных из кэша: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Скачивание данных с Binance для {self.symbol}...")
        
        print(f"Период: {start_date} до {end_date_str}")

        all_data = []
        while start < end:
            try:
                data = self._get_klines(interval=interval, start_time=start)
                if not data:
                    break
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "num_trades",
                    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                numeric_columns = ["open", "high", "low", "close", "volume", 
                                "quote_asset_volume", "taker_buy_base_volume", 
                                "taker_buy_quote_volume"]
                df[numeric_columns] = df[numeric_columns].astype(float)
                df["num_trades"] = df["num_trades"].astype(int)

                all_data.append(df)

                last_time = df["timestamp"].iloc[-1]
                start = last_time + timedelta(milliseconds=1)
                time.sleep(0.5)
            except Exception as e:
                print(f"Ошибка при получении данных: {e}")
                break

        result_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        if not result_df.empty:
            result_df.to_parquet(cache_path, index=False, compression="gzip")
            print(f"Данные сохранены в {cache_path}")

        return result_df