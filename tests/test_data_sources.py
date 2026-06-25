"""
Tests for data sources abstraction.

Tests:
- DataSourceFactory creates correct instances
- BinanceFuturesSource methods work
- BybitFuturesSource methods work
- Both sources return compatible DataFrame formats
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.data.data_sources import (
    DataSourceFactory,
    BinanceFuturesSource,
    BybitFuturesSource,
    BINANCE_FUTURES,
    BYBIT_FUTURES,
)


class TestDataSourceFactory:
    """Test factory pattern."""

    def test_create_binance(self):
        """Test creating Binance source."""
        src = DataSourceFactory.create("binance_futures", "BTCUSDT")
        assert isinstance(src, BinanceFuturesSource)
        assert src.symbol == "BTCUSDT"

    def test_create_bybit(self):
        """Test creating Bybit source."""
        src = DataSourceFactory.create("bybit_futures", "BTCUSDT")
        assert isinstance(src, BybitFuturesSource)
        assert src.symbol == "BTCUSDT"

    def test_case_insensitive(self):
        """Test factory is case-insensitive."""
        src1 = DataSourceFactory.create("BINANCE_FUTURES", "BTCUSDT")
        src2 = DataSourceFactory.create("binance_futures", "BTCUSDT")
        assert type(src1) == type(src2)

    def test_invalid_source(self):
        """Test invalid source raises error."""
        with pytest.raises(ValueError, match="Unknown source type"):
            DataSourceFactory.create("invalid_exchange", "BTCUSDT")

    def test_symbol_normalization(self):
        """Test symbols are uppercased."""
        src = DataSourceFactory.create("binance_futures", "btcusdt")
        assert src.symbol == "BTCUSDT"


class TestBinanceFuturesSource:
    """Test Binance data source."""

    def test_init(self):
        """Test initialization."""
        src = BinanceFuturesSource("BTCUSDT")
        assert src.symbol == "BTCUSDT"
        assert src.has_funding_data() is True
        assert src.has_oi_data() is True

    @pytest.mark.slow
    def test_get_klines_returns_df(self):
        """Test klines returns DataFrame."""
        src = BinanceFuturesSource("BTCUSDT")
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=10)

        df = src.get_klines(interval="1h", start_time=start, end_time=end, limit=10)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert "close" in df.columns

    @pytest.mark.slow
    def test_get_funding_rate(self):
        """Test funding rate fetch."""
        src = BinanceFuturesSource("BTCUSDT")

        df = src.get_funding_rate(limit=10)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "fundingRate" in df.columns

    @pytest.mark.slow
    def test_get_open_interest(self):
        """Test OI fetch."""
        src = BinanceFuturesSource("BTCUSDT")

        df = src.get_open_interest(interval="1h", limit=10)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "sumOpenInterest" in df.columns

    def test_invalid_interval_raises(self):
        """Test invalid interval raises ValueError."""
        src = BinanceFuturesSource("BTCUSDT")

        with pytest.raises(ValueError, match="Invalid interval"):
            src.get_klines(interval="invalid")

    def test_invalid_oi_interval_raises(self):
        """Test invalid OI interval raises ValueError."""
        src = BinanceFuturesSource("BTCUSDT")

        with pytest.raises(ValueError, match="Invalid OI interval"):
            src.get_open_interest(interval="3m")  # OI doesn't support 3m


class TestBybitFuturesSource:
    """Test Bybit data source."""

    def test_init(self):
        """Test initialization."""
        src = BybitFuturesSource("BTCUSDT")
        assert src.symbol == "BTCUSDT"
        assert src.has_funding_data() is True
        assert src.has_oi_data() is True

    @pytest.mark.slow
    def test_get_klines_returns_df(self):
        """Test klines returns DataFrame."""
        src = BybitFuturesSource("BTCUSDT")
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=10)

        df = src.get_klines(interval="4h", start_time=start, end_time=end, limit=10)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "close" in df.columns

    @pytest.mark.slow
    def test_get_funding_rate(self):
        """Test funding rate fetch."""
        src = BybitFuturesSource("BTCUSDT")

        df = src.get_funding_rate(limit=10)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "fundingRate" in df.columns

    @pytest.mark.slow
    def test_get_open_interest(self):
        """Test OI fetch."""
        src = BybitFuturesSource("BTCUSDT")

        df = src.get_open_interest(interval="4h", limit=10)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "timestamp" in df.columns
            assert "sumOpenInterest" in df.columns

    def test_interval_normalization(self):
        """Test interval normalization works."""
        src = BybitFuturesSource("BTCUSDT")

        # User-friendly intervals should be converted
        assert src._normalize_interval("1h") == "60"
        assert src._normalize_interval("4h") == "240"
        assert src._normalize_interval("1d") == "D"

    def test_invalid_interval_raises(self):
        """Test invalid interval raises ValueError."""
        src = BybitFuturesSource("BTCUSDT")

        with pytest.raises(ValueError, match="Invalid interval"):
            src._normalize_interval("invalid")


class TestDataSourceCompatibility:
    """Test that both sources return compatible formats."""

    @pytest.mark.slow
    def test_klines_column_compatibility(self):
        """Test both sources have compatible klines columns."""
        binance = BinanceFuturesSource("BTCUSDT")
        bybit = BybitFuturesSource("BTCUSDT")

        df_b = binance.get_klines(interval="1h", limit=5)
        df_by = bybit.get_klines(interval="1h", limit=5)

        # Both should have these core columns
        for df in [df_b, df_by]:
            if not df.empty:
                assert "timestamp" in df.columns
                assert "open" in df.columns
                assert "high" in df.columns
                assert "low" in df.columns
                assert "close" in df.columns
                assert "volume" in df.columns

    @pytest.mark.slow
    def test_funding_column_compatibility(self):
        """Test both sources have compatible funding columns."""
        binance = BinanceFuturesSource("BTCUSDT")
        bybit = BybitFuturesSource("BTCUSDT")

        df_b = binance.get_funding_rate(limit=5)
        df_by = bybit.get_funding_rate(limit=5)

        # Both should have these columns
        for df in [df_b, df_by]:
            if not df.empty:
                assert "timestamp" in df.columns
                assert "fundingRate" in df.columns

    @pytest.mark.slow
    def test_oi_column_compatibility(self):
        """Test both sources have compatible OI columns."""
        binance = BinanceFuturesSource("BTCUSDT")
        bybit = BybitFuturesSource("BTCUSDT")

        df_b = binance.get_open_interest(interval="4h", limit=5)
        df_by = bybit.get_open_interest(interval="4h", limit=5)

        # Both should have these columns
        for df in [df_b, df_by]:
            if not df.empty:
                assert "timestamp" in df.columns
                assert "sumOpenInterest" in df.columns
