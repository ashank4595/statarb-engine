# Run from project root:
# (venv) statarb-engine % python3 run_screen.py

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from candidate_pairs.create_pairs import all_pairs
from candidate_pairs.cointegration import screen_all
from data_layer import load_panel
from backtest.config import FREQ

DATA_DIR = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

# get every unique ticker from all pairs
pairs = all_pairs()
tickers = list({t for _, a, b in pairs for t in [a, b]})
print(f"Loading {len(tickers)} tickers...")

# load daily close panel
close = load_panel(DATA_DIR, tickers=tickers, freq=FREQ)
print(f"Loaded {close.shape[1]} stocks x {close.shape[0]} days\n")

# run the cointegration screen
print("Screening all pairs...")
results = screen_all(close, pairs)

# print results
print(f"\n{'='*65}")
print(f"RESULTS: {results['passes'].sum()} of {len(results)} pairs passed")
print(f"{'='*65}\n")
print(results.to_string(index=False))

# save to csv
out = os.path.join(os.path.dirname(__file__), "pair_screen_results.csv")
results.to_csv(out, index=False)
print(f"\nSaved to pair_screen_results.csv")
