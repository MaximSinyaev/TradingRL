#!/usr/bin/env python3
"""
Скрипт для пакетной загрузки данных для мульти-ассет обучения
и предварительного расчета всех фичей (state_vector) через FeatureGenerator.

Поддерживает Bybit Futures. Кэширует результат в папку smart_data_cache/.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '.')

from core.data.data_loader import load_multi_crypto_data
from core.features.feature_generator import FeatureGenerator

# Конфигурация
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
INTERVAL = "4h"
START_DATE = "2020-01-01" # Берем большой запас для прогрева индикаторов
SOURCE = "bybit_futures"
CACHE_DIR = "smart_data_cache"

def process_symbol(symbol: str, df: pd.DataFrame, output_dir: Path) -> bool:
    """Прогоняет сырой датафрейм через FeatureGenerator и сохраняет."""
    print(f"[{symbol}] Начало генерации фичей (исходных строк: {len(df)})...")
    try:
        generator = FeatureGenerator(
            price_col='close',
            volume_col='volume',
            ema_span=20,
            extra_ema_spans=[50, 100],
            use_ema_diffs=True,
            d_frac=0.5,
            hmm_path=None # Пока отключим HMM для ускорения, либо можно включить, если модель обучена
        )
        
        features_df = generator.transform(df)
        
        # Сохраняем в Parquet
        output_path = output_dir / f"{symbol}_{INTERVAL}_features.parquet"
        features_df.to_parquet(output_path, engine='pyarrow')
        
        print(f"[{symbol}] Успешно! Сохранено {len(features_df)} строк в {output_path.name}")
        return True
    except Exception as e:
        print(f"❌ [{symbol}] Ошибка при генерации фичей: {e}")
        return False

def main():
    print("="*50)
    print("🚀 СТАРТ ПОДГОТОВКИ MULTI-ASSET ДАННЫХ")
    print("="*50)
    print(f"Источники: {', '.join(SYMBOLS)}")
    print(f"Таймфрейм: {INTERVAL}, Старт: {START_DATE}")
    
    output_dir = Path(CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Скачиваем сырые данные (внутри функции работает параллельная загрузка)
    print("\n📥 ЭТАП 1: Загрузка сырых данных через SmartCacheLoader...")
    raw_data_dict = load_multi_crypto_data(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=None,
        interval=INTERVAL,
        source=SOURCE,
        cache_dir="data/cache",
        use_cache=True
    )
    
    if not raw_data_dict:
        print("❌ Не удалось скачать данные. Выход.")
        return
        
    # 2. Генерируем фичи (параллельно для разных монет)
    print("\n⚙️ ЭТАП 2: Генерация фичей (FeatureGenerator)...")
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=len(SYMBOLS)) as executor:
        futures = {
            executor.submit(process_symbol, sym, df, output_dir): sym 
            for sym, df in raw_data_dict.items()
        }
        
        for future in as_completed(futures):
            sym = futures[future]
            if future.result():
                success_count += 1
                
    print("\n" + "="*50)
    if success_count == len(SYMBOLS):
        print("✅ ВСЕ ДАННЫЕ УСПЕШНО ПОДГОТОВЛЕНЫ!")
    else:
        print(f"⚠️ ПОДГОТОВЛЕНО ЧАСТИЧНО: {success_count}/{len(SYMBOLS)}")
    print("="*50)

if __name__ == "__main__":
    main()
