#!/usr/bin/env python3
"""Скрипт для скачивания расширенного набора данных (6 месяцев)."""

import sys
sys.path.insert(0, '.')

from utils.binance_loader import MultiSymbolDataLoader
from datetime import datetime, timedelta

# Конфигурация
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"

# 6 месяцев
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

print(f"📥 ЗАГРУЗКА ДАННЫХ (6 месяцев):")
print(f"   Символы: {', '.join(SYMBOLS)}")
print(f"   Период: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
print(f"   Интервал: {INTERVAL}")
print(f"   Ожидаемый размер: ~{180*1440*2:,} свечей")
print(f"   Ожидаемое время: ~{(180*0.5)/60:.1f} минут")

# Загрузка
loader = MultiSymbolDataLoader(SYMBOLS)
data = loader.download_all(
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    interval=INTERVAL
)

# Статистика
print("\n" + "="*60)
print("📊 СТАТИСТИКА ЗАГРУЗКИ")
print("="*60)
stats = loader.get_combined_stats(data)
print(stats.to_string(index=False))
print("="*60)

print("\n✅ Данные успешно скачаны и закэшированы!")
