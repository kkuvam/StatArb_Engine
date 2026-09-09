from typing import Dict, List, Tuple

# Multi-Sector Pair Definitions
SECTOR_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "Financials_MoneyCenter": [
        ("JPM", "BAC"),  # JPMorgan Chase vs. Bank of America
        ("C", "WFC"),    # Citigroup vs. Wells Fargo
    ],
    "Financials_Investment": [
        ("GS", "MS"),    # Goldman Sachs vs. Morgan Stanley
    ],
    "Payments": [
        ("V", "MA"),     # Visa vs. Mastercard
    ],
    "Energy_Integrated": [
        ("XOM", "CVX"),  # ExxonMobil vs. Chevron
    ],
    "Energy_E_and_P": [
        ("COP", "EOG"),  # ConocoPhillips vs. EOG Resources
    ],
    "Energy_Services": [
        ("SLB", "HAL"),  # Schlumberger vs. Halliburton
    ],
    "Consumer_Staples": [
        ("KO", "PEP"),   # Coca-Cola vs. PepsiCo
        ("PG", "CL"),    # Procter & Gamble vs. Colgate-Palmolive
    ],
    "Industrials": [
        ("CAT", "DE"),   # Caterpillar vs. John Deere
    ],
    "Semiconductors_Equip": [
        ("LRCX", "AMAT"),# Lam Research vs. Applied Materials
    ],
}

def get_flat_ticker_list() -> List[str]:
    """Extracts unique ticker set across all candidate pairs."""
    tickers = set()
    for pair_list in SECTOR_PAIRS.values():
        for t1, t2 in pair_list:
            tickers.add(t1)
            tickers.add(t2)
    return sorted(list(tickers))

def get_flat_pair_list() -> List[Tuple[str, str]]:
    """Extracts flat list of pair tuples."""
    pairs = []
    for pair_list in SECTOR_PAIRS.values():
        pairs.extend(pair_list)
    return pairs