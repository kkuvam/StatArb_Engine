import sys
import yaml
import numpy as np
import pandas as pd

from config.universe import get_flat_ticker_list
from statarb.data.ingestion import CachedYFinanceDataHandler
from statarb.data.sanitization import DataSanitizer
from statarb.backtest.engine import RobustWalkForwardEngine


def run_full_system_test():
    print("================ STATARB ENGINE SYSTEM TEST SUITE ================")

    # 1. Configuration Loading
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("[PASS] Configuration loaded from config/params.yaml")

    # 2. Data Ingestion & Caching
    tickers = get_flat_ticker_list()
    start_date = "2021-01-01"
    end_date = "2024-01-01"

    data_handler = CachedYFinanceDataHandler(cache_dir="data/cache")
    prices, volumes = data_handler.fetch_data(tickers, start_date, end_date)

    assert not prices.empty, "Price DataFrame is empty!"
    assert not volumes.empty, "Volume DataFrame is empty!"
    print(
        f"[PASS] Data Ingestion & Parquet Caching ({len(tickers)} tickers, {prices.shape[0]} bars)"
    )

    # 3. Time Series Sanitization
    sanitizer = DataSanitizer(config)
    log_prices, liquidity_mask = sanitizer.process_panel(prices, volumes)

    assert not log_prices.isnull().values.any(), "Log prices contain NaN values!"
    print(f"[PASS] Data Sanitization & ADTV Masking (Shape: {log_prices.shape})")

    # 4. Multi-Sector Walk-Forward Simulation
    print("\nExecuting Walk-Forward Simulation Across Sector Universe...")
    engine = RobustWalkForwardEngine(
        config=config,
        lookback_window=126,
        transaction_cost_bps=float(config["optimization"]["transaction_cost_bps"]),
        z_entry=float(config["signals"]["z_entry"]),
        z_exit=float(config["signals"]["z_exit"]),
        tau_decay=15.0,
    )

    results_df, stats = engine.run_backtest(log_prices)

    # 5. Output Metric Verifications
    print("\n================ SYSTEM PERFORMANCE SUMMARY ================")
    for key, val in stats.items():
        if "Ratio" in key or "Turnover" in key or "Pairs" in key:
            print(f"  {key:<25}: {val:.4f}")
        else:
            print(f"  {key:<25}: {val * 100:.2f}%")

    print("\n================ TRAILING NAV TAIL ================")
    print(results_df.tail(5))

    # Assertions for valid execution
    assert (
        stats["Average Active Pairs"] > 0
    ), "Strategy generated zero active pair trades!"
    assert not np.isnan(stats["Sharpe Ratio"]), "Sharpe ratio evaluated to NaN!"
    assert results_df["NAV"].iloc[-1] > 0, "Portfolio NAV collapsed to 0!"

    print("\n[SUCCESS] ALL SYSTEM SUITE TESTS PASSED PERFECTLY.")


if __name__ == "__main__":
    run_full_system_test()