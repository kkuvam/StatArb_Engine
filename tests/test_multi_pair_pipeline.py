import numpy as np
import pandas as pd
import yaml

from config.universe import SECTOR_PAIRS, get_flat_pair_list, get_flat_ticker_list
from statarb.data.ingestion import CachedYFinanceDataHandler
from statarb.data.sanitization import DataSanitizer
from statarb.signals.kalman import KalmanHedgeEngine
from statarb.signals.ou_process import OUProcessAnalyzer


def test_multi_pair_expansion():
    # 1. Load Parameters & Initialize Handlers
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_handler = CachedYFinanceDataHandler(cache_dir="data/cache")
    sanitizer = DataSanitizer(config)
    kalman = KalmanHedgeEngine(
        delta=float(config["signals"]["kalman_delta"]),
        obs_noise=float(config["signals"]["kalman_obs_noise"]),
    )
    ou_analyzer = OUProcessAnalyzer()

    # 2. Ingest Universe Data
    tickers = get_flat_ticker_list()
    start_date = "2021-01-01"
    end_date = "2024-01-01"

    print(f"Fetching market data for {len(tickers)} tickers across sector pairs...")
    prices, volumes = data_handler.fetch_data(tickers, start_date, end_date)
    log_prices, liquidity_mask = sanitizer.process_panel(prices, volumes)

    # 3. Process Multi-Sector Pair Signals
    all_pairs = get_flat_pair_list()
    results = []

    for asset_A, asset_B in all_pairs:
        if asset_A not in log_prices.columns or asset_B not in log_prices.columns:
            continue

        yA = log_prices[asset_A]
        yB = log_prices[asset_B]

        # Run Kalman Filter
        kalman_df = kalman.filter_pair(yA, yB)
        latest_z = kalman_df["z_score"].iloc[-1]
        latest_beta = kalman_df["beta"].iloc[-1]
        spread = kalman_df["spread"]

        # Calculate OU Half-Life
        half_life = ou_analyzer.estimate_half_life(spread)

        results.append(
            {
                "Asset_A": asset_A,
                "Asset_B": asset_B,
                "Latest_Beta": round(latest_beta, 4),
                "Latest_ZScore": round(latest_z, 4),
                "OU_HalfLife_Days": round(half_life, 2),
                "Status": "ACTIVE" if 0.5 <= half_life <= 60.0 else "FILTERED",
            }
        )

    summary_df = pd.DataFrame(results)

    print("\n================ MULTI-PAIR SECTOR EXPANSION SUMMARY ================")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    test_multi_pair_expansion()