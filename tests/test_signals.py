import yaml
from statarb.data.ingestion import YFinanceDataHandler
from statarb.data.sanitization import DataSanitizer
from statarb.signals.cointegration import CointegrationScanner
from statarb.signals.kalman import KalmanHedgeEngine


def run_signals_test():
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    handler = YFinanceDataHandler()
    sanitizer = DataSanitizer(config)
    scanner = CointegrationScanner(
        confidence_level=config["signals"]["johansen_confidence"]
    )
    kalman = KalmanHedgeEngine(
        delta=float(config["signals"]["kalman_delta"]),
        obs_noise=float(config["signals"]["kalman_obs_noise"]),
    )

    tickers = ["CVX", "XOM"]
    start_date = "2022-01-01"
    end_date = "2024-01-01"

    prices, volumes = handler.fetch_data(tickers, start_date, end_date)
    log_prices, _ = sanitizer.process_panel(prices, volumes)

    # 1. Johansen Cointegration Test
    coint_res = scanner.test_johansen(log_prices)
    print("\n--- Johansen Cointegration Test ---")
    print(f"Is Cointegrated: {coint_res['is_cointegrated']}")
    print(f"Trace Rank: {coint_res['rank_trace']}")
    print(f"Normalized Beta: {coint_res['beta']}")

    # 2. Dynamic Kalman Filter Signal Generation
    kalman_df = kalman.filter_pair(log_prices["CVX"], log_prices["XOM"])
    
    # 3. Spread Stationarity Check
    is_stat, p_val = scanner.verify_spread_stationarity(kalman_df["spread"])
    print("\n--- Kalman Dynamic Filter & Spread Validation ---")
    print(f"Spread ADF Stationarity Test (p-value): {p_val:.5f} (Is Stationary: {is_stat})")
    print("\nKalman Signals Tail:")
    print(kalman_df[["beta", "spread", "z_score"]].tail())


if __name__ == "__main__":
    run_signals_test()