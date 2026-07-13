# make_mock_data.py
# Generate synthetic futures CSVs in data/ so the pipeline and the test suite can run
# end to end without any real market data.
#
# WHY THIS EXISTS
# Half of verify_pipeline.py's assertions need real prices, which means anyone cloning
# this repo cannot run the tests. That is backwards -- the tests are the part worth
# showing. This generator produces data in the exact input format, with a KNOWN answer
# planted in it, so the engine can be checked against ground truth:
#
#   COINTEGRATED pairs      built as  A = beta*B + OU(theta)
#                           the spread IS mean-reverting, by construction, with a
#                           beta and a half-life we chose. The screen must FIND these.
#
#   NON-COINTEGRATED pairs  two independent random walks.
#                           The spread is a random walk too. The screen must REJECT
#                           these -- and if it passes more than ~5% of them, its
#                           false-positive rate is not what it claims.
#
# The second group is the more important one. A screen that finds real cointegration is
# easy; a screen that correctly says "nothing here" when there is nothing here is the
# thing that is hard to get right, and the thing that silently breaks.
#
# Ticker names are S&P constituents purely so the sector groupings read naturally.
# None of the prices bear any relation to the real companies.
#
# Run from the repo root:  python3 make_mock_data.py

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

START = "2023-01-02"
YEARS = 3.5
BARS_PER_DAY = 7            # 09:30 -> 15:30 hourly-ish, keeps files small
SEED = 42

# --- what to plant ------------------------------------------------------------
#
# (ticker_a, ticker_b, beta, half_life_days)
# These pairs ARE cointegrated. The spread is an Ornstein-Uhlenbeck process, so it
# genuinely reverts to a mean, with the half-life given. A correct screen finds them.
COINTEGRATED = [
    ("XOM",  "CVX",  0.70,  12),    # oil majors
    ("JPM",  "BAC",  2.40,  20),    # money-center banks
    ("KO",   "PEP",  0.35,  15),    # beverages
    ("HD",   "LOW",  1.55,  25),    # home improvement
    ("V",    "MA",   0.60,  18),    # payment networks
]

# These are INDEPENDENT random walks. Nothing links them. The screen must reject them.
# There are deliberately many, so the false-positive rate is measurable: at p < 0.05
# you should see roughly 5% of these slip through, and no more.
INDEPENDENT = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "UNH", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR",
    "CAT", "DE", "BA", "GE", "HON", "MMM", "UPS", "RTX",
]

# starting price levels, deliberately spread across two orders of magnitude so the
# hedge ratio estimator's scale-invariance is actually exercised.
BASE_PRICES = {
    "XOM": 105.0, "CVX": 150.0, "JPM": 145.0, "BAC": 32.0,
    "KO": 60.0, "PEP": 170.0, "HD": 340.0, "LOW": 215.0,
    "V": 260.0, "MA": 430.0,
}


def _trading_index(rng: np.random.Generator) -> pd.DatetimeIndex:
    """Business days x intraday bars, so the CSVs look like real bar data."""
    days = pd.bdate_range(START, periods=int(YEARS * 252))
    times = pd.to_timedelta(np.arange(BARS_PER_DAY) + 9, unit="h") + pd.Timedelta("30min")
    stamps = [d + t for d in days for t in times]
    return pd.DatetimeIndex(stamps)


def random_walk(n: int, start: float, vol: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric-ish random walk. NOT mean-reverting -- it wanders and never comes back.
    This is the null hypothesis, made concrete."""
    steps = rng.normal(0, vol, n)
    return start * np.exp(np.cumsum(steps))


def ou_process(n: int, half_life_bars: float, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck: a mean-reverting process, which is exactly what a tradeable
    spread is supposed to be.

        dx = theta * x * dt + sigma * dW,   theta < 0

    theta is recovered from the half-life:  half_life = -ln(2) / theta
    so a 12-day half-life means the spread closes half its gap to zero in 12 days.
    This is the quantity half_life() in cointegration.py estimates."""
    theta = -np.log(2) / half_life_bars
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * x[i - 1] + rng.normal(0, sigma)
    return x


def write_csv(ticker: str, prices: np.ndarray, index: pd.DatetimeIndex,
              rng: np.random.Generator) -> None:
    """Write in the exact format data_layer expects: dd/mm/yyyy dates, HH:MM:SS times."""
    df = pd.DataFrame({
        "Date": index.strftime("%d/%m/%Y"),
        "Time": index.strftime("%H:%M:%S"),
        "Open": np.round(prices * (1 + rng.normal(0, 0.0004, len(prices))), 2),
        "High": np.round(prices * (1 + abs(rng.normal(0, 0.0008, len(prices)))), 2),
        "Low":  np.round(prices * (1 - abs(rng.normal(0, 0.0008, len(prices)))), 2),
        "Close": np.round(prices, 2),
        "Volume": rng.integers(10_000, 90_000, len(prices)),
    })
    path = os.path.join(OUT_DIR, f"{ticker}.csv")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    index = _trading_index(rng)
    n = len(index)
    bars_per_day = BARS_PER_DAY

    print(f"generating {n:,} bars ({n // bars_per_day:,} days) -> {OUT_DIR}\n")

    # --- cointegrated pairs ---------------------------------------------------
    print("COINTEGRATED (the screen must find these):")
    for a, b, beta, half_life_days in COINTEGRATED:
        price_b = random_walk(n, BASE_PRICES[b], 0.004, rng)

        # the spread is an OU process -> genuinely mean-reverting. Half-life is given
        # in DAYS, so convert to bars.
        spread = ou_process(n, half_life_days * bars_per_day,
                            sigma=BASE_PRICES[a] * 0.004, rng=rng)
        price_a = beta * price_b + spread

        write_csv(a, price_a, index, rng)
        write_csv(b, price_b, index, rng)
        print(f"  {a:<6} - {b:<6}  beta = {beta:<5}  half-life = {half_life_days} days")

    # --- independent random walks ---------------------------------------------
    print(f"\nINDEPENDENT random walks (the screen must REJECT these):")
    for t in INDEPENDENT:
        start = rng.uniform(40, 400)
        write_csv(t, random_walk(n, start, 0.004, rng), index, rng)
    print(f"  {len(INDEPENDENT)} tickers -> "
          f"{len(INDEPENDENT) * (len(INDEPENDENT) - 1) // 2} possible non-cointegrated pairs")

    total = 2 * len(COINTEGRATED) + len(INDEPENDENT)
    print(f"\nwrote {total} CSVs to {OUT_DIR}")
    print("\nWhat to check:")
    print("  1. The screen should find the 5 planted pairs and recover their betas.")
    print("  2. It should reject nearly all pairs built from the independent walks.")
    print("     At p < 0.05, expect ~5% of them to slip through -- that is the")
    print("     false-positive rate working as designed, not a bug. If MANY more")
    print("     pass, the test is miscalibrated.")
    print("  3. verification_scripts/adf_screen_dump.py plots the p-value histogram:")
    print("     the planted pairs make a spike near zero; the random walks should be")
    print("     spread UNIFORMLY across [0, 1]. That contrast is the whole point.")
