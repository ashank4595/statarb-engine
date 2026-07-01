"""
sectors.py - Step 1 of pair discovery: group stocks by economic logic.

The discipline rule: we ONLY test pairs WITHIN the same sector. Two banks have
a real reason to move together (same interest rates, same regulation, same
economic drivers). A bank + a pharma stock that happen to correlate is almost
certainly a fluke - and trading flukes is how you lose money.

This pre-filter cuts the search from ~1,225 possible pairs down to a few dozen
economically-motivated candidates, which controls data-mining bias.

NOTE: these are standard NSE ticker symbols for the Nifty 50. Two things to check:
  1. Make sure each ticker matches YOUR CSV filename exactly (e.g. if your files
     are HDFCBANK_FUT.csv, either rename them or edit the tickers here).
  2. Nifty 50 membership changes over time - remove any stock you don't have data
     for, or add ones you do. The screen skips tickers with no data anyway.

The sectors with 3+ stocks are the richest for pairs (more combinations). Banks,
IT, and autos are usually the best hunting grounds for cointegration.
"""

SECTORS = {
    "banks": [
        "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK",
    ],
    "financials_nonbank": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "SHRIRAMFIN",
    ],
    "it": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    ],
    "energy_oil": [
        "RELIANCE", "ONGC", "BPCL", "COALINDIA",
    ],
    "power": [
        "NTPC", "POWERGRID", "TATAPOWER",
    ],
    "autos": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
    ],
    "metals": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO",
    ],
    "fmcg": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    ],
    "pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "APOLLOHOSP",
    ],
    "cement": [
        "ULTRACEMCO", "GRASIM", "SHREECEM",
    ],
    "infra_construction": [
        "LT", "ADANIPORTS", "ADANIENT",
    ],
    "consumer_other": [
        "ASIANPAINT", "TITAN", "TRENT",
    ],
    "telecom": [
        "BHARTIARTL",
    ],
}


def candidate_pairs(sectors: dict = SECTORS):
    """
    Turn the sector buckets into a flat list of within-sector candidate pairs.
    A bucket of 4 stocks yields 6 pairs (every combination of 2).
    Returns list of (sector, stock_a, stock_b).
    """
    from itertools import combinations
    pairs = []
    for sector, tickers in sectors.items():
        for a, b in combinations(sorted(tickers), 2):
            pairs.append((sector, a, b))
    return pairs


if __name__ == "__main__":
    pairs = candidate_pairs()
    print(f"{len(pairs)} candidate pairs from {len(SECTORS)} sectors:")
    for sector, a, b in pairs:
        print(f"  [{sector}] {a} - {b}")
