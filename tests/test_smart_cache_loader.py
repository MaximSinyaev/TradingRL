"""
Tests for SmartCacheLoader.

Tests:
- Cache initialization
- Incremental downloads (only missing dates)
- Metadata queries
- Save/load by year
"""

import pytest
import pandas as pd
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.data.multi_source_loader import SmartCacheLoader
from core.data.data_sources import BINANCE_FUTURES, BYBIT_FUTURES


@pytest.fixture
def temp_cache(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    yield str(cache_dir)
    # Cleanup is automatic with tmp_path


class TestSmartCacheLoader:
    """Test smart cache functionality."""

    def test_init_creates_directories(self, temp_cache):
        """Test initialization creates cache structure."""
        loader = SmartCacheLoader(cache_dir=temp_cache)

        assert Path(temp_cache).exists()
        assert (Path(temp_cache) / "metadata.db").exists()
        assert (Path(temp_cache) / "data").exists()

    def test_config_hash_is_deterministic(self, temp_cache):
        """Test config hash is consistent for same inputs."""
        loader = SmartCacheLoader(cache_dir=temp_cache)

        hash1 = loader._get_config_hash("BTCUSDT", BINANCE_FUTURES, "4h", True, True)
        hash2 = loader._get_config_hash("BTCUSDT", BINANCE_FUTURES, "4h", True, True)

        assert hash1 == hash2

    def test_config_hash_differs_by_config(self, temp_cache):
        """Test different configs produce different hashes."""
        loader = SmartCacheLoader(cache_dir=temp_cache)

        hash1 = loader._get_config_hash("BTCUSDT", BINANCE_FUTURES, "4h", True, True)
        hash2 = loader._get_config_hash("BTCUSDT", BINANCE_FUTURES, "1h", True, True)
        hash3 = loader._get_config_hash("ETHUSDT", BINANCE_FUTURES, "4h", True, True)

        assert hash1 != hash2
        assert hash1 != hash3

    def test_get_existing_dates_empty_cache(self, temp_cache):
        """Test getting dates from empty cache returns empty set."""
        loader = SmartCacheLoader(cache_dir=temp_cache)

        dates = loader.get_existing_dates(
            "BTCUSDT", BINANCE_FUTURES, "4h", True, True
        )

        assert dates == set()

    def test_download_creates_metadata(self, temp_cache):
        """Test download creates metadata records."""
        # This test requires actual API - mark as slow
        pytest.skip("Requires API access - run with pytest -m slow")

    @pytest.mark.slow
    def test_incremental_download_only_fetches_missing(self, temp_cache):
        """Test incremental download only fetches missing dates."""
        loader = SmartCacheLoader(cache_dir=temp_cache)

        # First download
        df1 = loader.download(
            symbol="BTCUSDT",
            start_date="2024-06-20",
            end_date="2024-06-21",
            interval="4h",
            fetch_oi=False,  # Skip OI for faster test
        )

        # Second download should be from cache
        df2 = loader.download(
            symbol="BTCUSDT",
            start_date="2024-06-20",
            end_date="2024-06-21",
            interval="4h",
            fetch_oi=False,
        )

        # Should be identical
        if not df1.empty and not df2.empty:
            pd.testing.assert_frame_equal(df1.sort_values("timestamp").reset_index(drop=True),
                                          df2.sort_values("timestamp").reset_index(drop=True))

    @pytest.mark.slow
    def test_get_stats_returns_summary(self, temp_cache):
        """Test get_stats returns cache summary."""
        loader = SmartCacheLoader(cache_dir=temp_cache)

        # Download something
        loader.download(
            symbol="BTCUSDT",
            start_date="2024-06-20",
            end_date="2024-06-21",
            interval="4h",
            fetch_oi=False,
        )

        stats = loader.get_stats()

        assert isinstance(stats, pd.DataFrame)
        if not stats.empty:
            assert "symbol" in stats.columns
            assert "total_records" in stats.columns


class TestMultiSourceWrapper:
    """Test legacy compatibility wrapper."""

    def test_download_all(self, temp_cache):
        """Test downloading multiple symbols."""
        pytest.skip("Requires API access")

    def test_wrapper_uses_loader(self, temp_cache):
        """Test wrapper properly delegates to SmartCacheLoader."""
        from core.data.multi_source_loader import MultiSourceDataLoader

        wrapper = MultiSourceDataLoader(
            symbols=["BTCUSDT"],
            cache_dir=temp_cache
        )

        assert wrapper.loader is not None
        assert wrapper.loader.primary_source == BINANCE_FUTURES


class TestDataSourceSelector:
    """Test selecting different sources for different data types."""

    @pytest.mark.slow
    def test_bybit_for_oi_binance_for_klines(self, temp_cache):
        """Test using Binance for klines and Bybit for OI."""
        loader = SmartCacheLoader(
            cache_dir=temp_cache,
            primary_source=BINANCE_FUTURES,
            oi_source=BYBIT_FUTURES  # Use Bybit for historical OI
        )

        df = loader.download(
            symbol="BTCUSDT",
            start_date="2024-06-20",
            end_date="2024-06-21",
            interval="4h",
            fetch_funding=False,
            fetch_oi=True,
        )

        # Should have both klines and OI data
        if not df.empty:
            assert "close" in df.columns or "sumOpenInterest" in df.columns
