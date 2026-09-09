import yaml
from statarb.data.ingestion import YFinanceDataHandler
from statarb.data.sanitization import DataSanitizer


def run_pipeline_test():
    # Load configuration
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    handler = YFinanceDataHandler()
    sanitizer = DataSanitizer(config)

    # Energy Sector Pair Benchmark
    tickers = ["XOM", "CVX", "COP"]
    start_date = "2022-01-01"
    end_date = "2024-01-01"

    print("Fetching raw data...")
    prices, volumes = handler.fetch_data(tickers, start_date, end_date)

    print("Sanitizing time series and constructing masks...")
    log_prices, liquidity_mask = sanitizer.process_panel(prices, volumes)

    print("\n--- Pipeline Verification Results ---")
    print(f"Log-Price Matrix Dimensions (T x N): {log_prices.shape}")
    print("\nLog-Prices Head:")
    print(log_prices.head())
    print("\nLiquidity Mask Active Status Count:")
    print(liquidity_mask.sum(axis=0))


if __name__ == "__main__":
    run_pipeline_test()