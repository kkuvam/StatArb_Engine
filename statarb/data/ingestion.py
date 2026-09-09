from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import yfinance as yf


class AbstractDataHandler(ABC):
    """Abstract interface for point-in-time financial data ingestion."""

    @abstractmethod
    def fetch_data(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns tuple of (adjusted_close_prices, daily_volumes)."""
        pass


class CachedYFinanceDataHandler(AbstractDataHandler):
    """Fetches total-return adjusted market data with local Parquet disk caching.

    Prevents rate-limiting (HTTP 429) by storing downloaded price/volume panels.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tickers = sorted(list(set(tickers)))
        cache_key = f"panel_{start_date}_{end_date}.parquet"
        price_cache_path = self.cache_dir / f"prices_{cache_key}"
        volume_cache_path = self.cache_dir / f"volumes_{cache_key}"

        # 1. Load from Parquet cache if available
        if price_cache_path.exists() and volume_cache_path.exists():
            prices = pd.read_parquet(price_cache_path)
            volumes = pd.read_parquet(volume_cache_path)

            # Ensure all requested tickers exist in cached panel
            if all(t in prices.columns for t in tickers):
                return prices[tickers], volumes[tickers]

        # 2. Batch download missing tickers from Yahoo Finance
        raw_data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,  # Adjusts for splits and dividends
            progress=False,
        )

        if raw_data.empty:
            raise ValueError(
                f"No data returned for tickers {tickers} between {start_date} and {end_date}."
            )

        if len(tickers) == 1:
            prices = raw_data["Close"].to_frame(name=tickers[0])
            volumes = raw_data["Volume"].to_frame(name=tickers[0])
        else:
            prices = raw_data["Close"]
            volumes = raw_data["Volume"]

        # 3. Persist to local Parquet cache
        prices.to_parquet(price_cache_path, compression="snappy")
        volumes.to_parquet(volume_cache_path, compression="snappy")

        return prices, volumes