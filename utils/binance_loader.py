import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from tqdm import tqdm

class BinanceKlinesDownloader:
    """Downloader для klines данных с Binance.

    Поддерживает:
    - Кэширование в parquet.gz формате
    - Множественные символы
    - Произвольные временные интервалы
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, symbol: str = "BTCUSDT", cache_dir: str = "binance_data_cache"):
        self.symbol = symbol.upper()
        self.limit = 1000  # Binance max limit per request
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

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def _get_cache_path(self, start_date: str, end_date: Optional[str], interval: str) -> str:
        end_date_str = end_date if end_date else "now"
        filename = f"{self.symbol}_{interval}_{start_date}_to_{end_date_str}.parquet.gz"
        return os.path.join(self.symbol_dir, filename)

    def download(self, start_date: str, end_date: Optional[str] = None,
                 interval: str = "1m") -> pd.DataFrame:
        """Скачивает klines данные с Binance.

        Args:
            start_date: Дата начала в формате 'YYYY-MM-DD'
            end_date: Дата окончания в формате 'YYYY-MM-DD' (опционально)
            interval: Интервал свечей (1m, 5m, 1h, etc.)

        Returns:
            DataFrame с OHLCV данными
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else datetime.now(timezone.utc)
        end_date_str = end_date if end_date else end.strftime("%Y-%m-%d")

        cache_path = self._get_cache_path(start_date, end_date_str, interval)

        if os.path.exists(cache_path):
            print(f"📦 Чтение из кэша: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"📥 Скачивание {self.symbol} | {start_date} → {end_date_str}")
        print(f"⏱️  Ожидаемое время: ~{(end - start).days * 0.5:.1f} сек")

        all_data = []
        current_start = start
        total_requests = 0

        # Оценка количества запросов
        estimated_minutes = (end - start).total_seconds() / 60
        estimated_requests = max(1, int(estimated_minutes / 1000))

        with tqdm(total=estimated_requests, desc=f"{self.symbol}", unit="req") as pbar:
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

                    all_data.append(df)

                    last_time = df["timestamp"].iloc[-1]
                    current_start = last_time + timedelta(milliseconds=1)
                    total_requests += 1

                    pbar.update(1)
                    pbar.set_postfix({"candles": len(all_data) * 1000})

                    time.sleep(0.3)  # Respect rate limits

                except Exception as e:
                    print(f"❌ Ошибка: {e}")
                    break

        if not all_data:
            print("⚠️  Не удалось получить данные")
            return pd.DataFrame()

        result_df = pd.concat(all_data, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        # Remove duplicates
        result_df = result_df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        print(f"✅ Итого: {len(result_df):,} свечей")
        print(f"📊 Период: {result_df['timestamp'].min()} → {result_df['timestamp'].max()}")

        result_df.to_parquet(cache_path, index=False, compression="gzip")
        print(f"💾 Сохранено: {cache_path}")

        return result_df


class MultiSymbolDataLoader:
    """Загрузчик для нескольких символов одновременно."""

    def __init__(self, symbols: list[str], cache_dir: str = "binance_data_cache"):
        self.symbols = [s.upper() for s in symbols]
        self.downloaders = {
            symbol: BinanceKlinesDownloader(symbol, cache_dir)
            for symbol in self.symbols
        }

    def download_all(self, start_date: str, end_date: Optional[str] = None,
                     interval: str = "1m") -> dict[str, pd.DataFrame]:
        """Скачивает данные для всех символов.

        Returns:
            Dict {symbol: DataFrame}
        """
        results = {}
        for symbol, downloader in self.downloaders.items():
            print(f"\n{'='*60}")
            results[symbol] = downloader.download(start_date, end_date, interval)
            print(f"{'='*60}\n")
        return results

    def get_combined_stats(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Возвращает статистику по всем символам."""
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
