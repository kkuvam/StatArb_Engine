from typing import Dict, Tuple
import numpy as np
import pandas as pd


class DataSanitizer:
    def __init__(self, config: Dict):
        self.min_adtv = float(config["universe"]["min_adtv_usd"])
        self.adtv_window = int(config["universe"]["adtv_window_days"])
        self.max_missing_pct = float(config["universe"]["max_missing_pct"])

    def process_panel(
        self, prices: pd.DataFrame, volumes: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Filter assets exceeding missing threshold
        missing_ratios = prices.isnull().sum() / len(prices)
        valid_assets = missing_ratios[missing_ratios <= self.max_missing_pct].index
        
        # Forward fill up to limit, drop remaining initial NaNs per-column
        clean_prices = prices[valid_assets].ffill(limit=3).bfill()
        clean_volumes = volumes[valid_assets].reindex(clean_prices.index).fillna(0.0)

        dollar_volume = clean_prices * clean_volumes
        rolling_adtv = dollar_volume.rolling(
            window=self.adtv_window, min_periods=1
        ).mean()

        liquidity_mask = rolling_adtv >= self.min_adtv
        log_prices = np.log(clean_prices)
        
        return log_prices, liquidity_mask