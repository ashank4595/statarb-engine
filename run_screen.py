"""
run_screen.py - the pair-discovery pipeline, end to end.

  1. load daily close panel from your near-month futures data
  2. generate economically-motivated candidate pairs (same-sector, from sectors.py)
  3. screen each pair: correlation, cointegration (Engle-Granger + ADF), half-life
  4. print a ranked shortlist of tradeable pairs

NO trading, NO P&L here - this is pure screening (Phase A). The output is the
list of pairs worth backtesting later.

Run:  python3 run_screen.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sectors import SECTORS, candidate_pairs
from data_layer import load_panel, health_check
from find_pairs import screen_all

# --- point this at your data ---
DATA_DIR = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

# only load the tickers we actually use in sectors.py (fast)
wanted = sorted({t for tickers in SECTORS.values() for t in tickers})

print(f"Loading {len(wanted)} tickers from near-month futures...")
close = load_panel(DATA_DIR, only_near_month=True, tickers=wanted)
print("\nData health check:")
health_check(close)

print("\nGenerating candidate pairs...")
pairs = candidate_pairs()
# keep only pairs where BOTH stocks actually loaded
have = set(close.columns)
pairs = [(s, a, b) for (s, a, b) in pairs if a in have and b in have]
print(f"  {len(pairs)} candidate pairs to screen")

print("\nScreening (this runs the cointegration test on each pair)...")
results = screen_all(close, pairs, coint_threshold=0.05)

if results.empty:
    print("No results - check that tickers in sectors.py match your filenames.")
else:
    passed = results[results["passes"]]
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(passed)} of {len(results)} pairs passed cointegration")
    print(f"{'='*70}\n")
    print("Top pairs (lowest cointegration p-value = strongest relationship):\n")
    print(results.head(20).to_string(index=False))
    print(f"\nFull ranked table has {len(results)} rows.")
    # save it
    out = os.path.join(os.path.dirname(__file__), "pair_screen_results.csv")
    results.to_csv(out, index=False)
    print(f"Saved full results to {out}")
