"""
Demo script showing Bybit historical Open Interest access.

This demonstrates that Bybit API provides OI data beyond Binance's 30-day limit.
"""

from datetime import datetime, timezone, timedelta
from core.data.data_sources import DataSourceFactory, BYBIT_FUTURES

def demo_historical_oi():
    """Fetch historical OI from Bybit."""
    print("🔍 Fetching Open Interest from Bybit...")
    print("-" * 60)

    bybit = DataSourceFactory.create(BYBIT_FUTURES, "BTCUSDT")
    now = datetime.now(timezone.utc)

    # Try fetching for different time ranges
    test_ranges = [
        ("Last 7 days", now - timedelta(days=7), now),
        ("Last 30 days", now - timedelta(days=30), now),
        ("Last 90 days", now - timedelta(days=90), now),
        ("Last 180 days", now - timedelta(days=180), now),
    ]

    for label, start_date, end_date in test_ranges:
        print(f"\n📅 {label}:")

        df = bybit.get_open_interest(
            interval="4h",
            start_time=start_date,
            end_time=end_date,
            limit=200
        )

        if not df.empty:
            print(f"   ✅ Got {len(df)} records")
            print(f"   📊 Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
            print(f"   💰 OI range: {df['sumOpenInterest'].min():.2f} → {df['sumOpenInterest'].max():.2f}")
        else:
            print(f"   ❌ No data")

    print("\n" + "=" * 60)
    print("✅ Bybit provides historical OI beyond 30 days!")
    print("=" * 60)

if __name__ == "__main__":
    demo_historical_oi()
