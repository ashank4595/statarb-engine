# verify_pipeline.py
# Property-based checks on the parts of the pipeline that fail SILENTLY.
#
# Every bug found in this repo so far was silent: MODE duplicated across three
# files so stale results were displayed without error; the z-score window nearly
# equal to the trading window so almost no days were tradeable; hedge_ratio
# returning -103 on an uncorrelated pair and ADF scoring the resulting nonsense
# spread without complaint. None of these threw. They all just produced numbers.
#
# So the tests below assert PROPERTIES that must hold, not just "it ran":
#   - known-answer   : a fit on data with a known beta must recover that beta
#   - symmetry       : beta(A,B) must equal 1/beta(B,A)
#   - scale          : rescaling a leg must scale beta predictably, not arbitrarily
#   - gating         : garbage pairs must be REJECTED, not merely scored badly
#   - no look-ahead  : the vectorized engine must equal the explicit loop
#   - consistency    : all three modes must apply the same gate
#
# This cannot prove the code is bug free -- nothing can. What it does is lock down
# the specific properties that have already broken once, so those failures cannot
# come back silently.
#
# Run from the repo root: python3 -m verification_scripts.verify_pipeline

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_layer import load_panel
from candidate_pairs.cointegration import (
    hedge_ratio, correlation, pair_is_valid, spread_with_beta,
    MIN_CORRELATION, MIN_BETA,
)
from backtest.zscore_signal import zscore, positions, ENTRY_THRESHOLD
from backtest.engine import backtest_pair, backtest_pair_loop
from backtest.validation_methods import run_full, run_split, run_rolling
from backtest import config

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

RNG = np.random.default_rng(42)   # fixed seed: these tests must be deterministic

_passed, _failed = 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion. Never raises -- we want the full report, not the
    first failure."""
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def _series(values) -> pd.Series:
    """Wrap an array in a Series with a real business-day index."""
    idx = pd.bdate_range("2022-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


# --- 1. hedge_ratio: known answer --------------------------------------------

def test_hedge_ratio_known_answer():
    print("\n[1] hedge_ratio recovers a beta it was given")

    b = _series(1000 + np.cumsum(RNG.normal(0, 10, 500)))
    a = _series(3.0 * b.to_numpy())          # A is exactly 3x B, zero noise
    beta = hedge_ratio(a, b)
    check("exact A = 3*B  ->  beta == 3", np.isclose(beta, 3.0, rtol=1e-6),
          f"got {beta:.6f}")

    # now the realistic case: a true beta of 3, plus independent noise on BOTH legs.
    # A sits at ~3000 and B at ~1000, so A's noise is naturally larger in rupees --
    # this is exactly the asymmetry that broke the unstandardized version.
    b2 = _series(1000 + np.cumsum(RNG.normal(0, 10, 800)))
    noise_a = RNG.normal(0, 30, 800)         # 1% of A's level
    noise_b = RNG.normal(0, 10, 800)         # 1% of B's level
    a2 = _series(3.0 * b2.to_numpy() + noise_a)
    b2 = _series(b2.to_numpy() + noise_b)
    beta2 = hedge_ratio(a2, b2)
    check("noisy A ~ 3*B  ->  beta within 10% of 3", abs(beta2 - 3.0) / 3.0 < 0.10,
          f"got {beta2:.4f}, error {abs(beta2 - 3.0) / 3.0:.1%}")


# --- 2. hedge_ratio: symmetry -------------------------------------------------

def test_hedge_ratio_symmetry():
    print("\n[2] hedge_ratio is symmetric (the reason TLS replaced OLS)")

    b = _series(500 + np.cumsum(RNG.normal(0, 5, 600)))
    a = _series(2.5 * b.to_numpy() + RNG.normal(0, 25, 600))

    fwd = hedge_ratio(a, b)
    rev = hedge_ratio(b, a)
    check("beta(A,B) == 1 / beta(B,A)", np.isclose(fwd, 1.0 / rev, rtol=1e-6),
          f"beta(A,B)={fwd:.6f}, 1/beta(B,A)={1.0 / rev:.6f}")


# --- 3. hedge_ratio: scale behaviour ------------------------------------------

def test_hedge_ratio_scale():
    print("\n[3] hedge_ratio scales predictably (the bug that was fixed)")

    b = _series(1000 + np.cumsum(RNG.normal(0, 10, 600)))
    a = _series(2.0 * b.to_numpy() + RNG.normal(0, 20, 600))
    beta = hedge_ratio(a, b)

    # spread = A - beta*B. Multiply A by 10 and beta must multiply by 10 for the
    # spread to keep its meaning. Multiply B by 10 and beta must divide by 10.
    beta_a10 = hedge_ratio(_series(a.to_numpy() * 10), b)
    check("A scaled 10x  ->  beta scales 10x", np.isclose(beta_a10, beta * 10, rtol=1e-6),
          f"expected {beta * 10:.6f}, got {beta_a10:.6f}")

    beta_b10 = hedge_ratio(a, _series(b.to_numpy() * 10))
    check("B scaled 10x  ->  beta scales 1/10", np.isclose(beta_b10, beta / 10, rtol=1e-6),
          f"expected {beta / 10:.6f}, got {beta_b10:.6f}")

    # the real test: a huge price-level gap must NOT inflate beta. True beta is
    # 50 here (A ~ 50,000, B ~ 1,000) -- the raw-TLS version blew up on exactly this.
    b3 = _series(1000 + np.cumsum(RNG.normal(0, 10, 800)))
    a3 = _series(50.0 * b3.to_numpy() + RNG.normal(0, 500, 800))
    beta3 = hedge_ratio(a3, b3)
    check("50x price gap  ->  beta within 10% of 50", abs(beta3 - 50.0) / 50.0 < 0.10,
          f"got {beta3:.3f}, error {abs(beta3 - 50.0) / 50.0:.1%}")


# --- 4. pair_is_valid: the gate -----------------------------------------------

def test_gate():
    print("\n[4] pair_is_valid rejects what ADF alone would have let through")

    # two independent random walks: no relationship whatsoever.
    a = _series(1000 + np.cumsum(RNG.normal(0, 10, 700)))
    b = _series(1000 + np.cumsum(RNG.normal(0, 10, 700)))
    beta = hedge_ratio(a, b)
    ok, reason = pair_is_valid(a, b, beta)
    corr = correlation(a, b)
    check("uncorrelated random walks are REJECTED", not ok,
          f"corr={corr:.3f}, beta={beta:.3f}, verdict={reason}")

    # inversely related legs -> negative beta -> would be long both legs.
    b2 = _series(1000 + np.cumsum(RNG.normal(0, 10, 700)))
    a2 = _series(5000 - 2.0 * b2.to_numpy() + RNG.normal(0, 20, 700))
    beta2 = hedge_ratio(a2, b2)
    ok2, reason2 = pair_is_valid(a2, b2, beta2)
    check("negative beta is REJECTED", not ok2,
          f"beta={beta2:.3f}, verdict={reason2}")

    # a genuinely cointegrated, positively related pair must still pass.
    b3 = _series(1000 + np.cumsum(RNG.normal(0, 10, 700)))
    a3 = _series(2.0 * b3.to_numpy() + RNG.normal(0, 15, 700))
    beta3 = hedge_ratio(a3, b3)
    ok3, reason3 = pair_is_valid(a3, b3, beta3)
    check("a good pair still PASSES", ok3,
          f"corr={correlation(a3, b3):.3f}, beta={beta3:.3f}, verdict={reason3}")

    check("nan beta is REJECTED", not pair_is_valid(a3, b3, np.nan)[0])


# --- 5. zscore: scale invariance ----------------------------------------------

def test_zscore_scale_invariant():
    print("\n[5] z-score is scale-free (so entry/exit thresholds never need retuning)")

    s = _series(np.cumsum(RNG.normal(0, 1, 400)))
    z1 = zscore(s).dropna()
    z2 = zscore(_series(s.to_numpy() * 1000)).dropna()
    check("zscore(1000 * spread) == zscore(spread)",
          np.allclose(z1.to_numpy(), z2.to_numpy(), rtol=1e-9),
          f"max diff {np.abs(z1.to_numpy() - z2.to_numpy()).max():.2e}")

    # a beta change rescales the spread; the SIGNAL must be unaffected by that
    # alone, which is why beta errors show up in P&L and margin, not in entries.
    p1 = positions(zscore(s))
    p2 = positions(zscore(_series(s.to_numpy() * 1000)))
    check("positions are identical after rescaling", p1.equals(p2))


# --- 6. positions: legal values and threshold logic ---------------------------

def test_positions():
    print("\n[6] positions() emits only legal values and respects its thresholds")

    s = _series(np.cumsum(RNG.normal(0, 1, 500)))
    z = zscore(s)
    p = positions(z)

    check("positions only ever in {-1, 0, +1}",
          set(p.unique()).issubset({-1.0, 0.0, 1.0}),
          f"found {sorted(p.unique())}")

    # a position may only OPEN on a day |z| exceeded the entry threshold.
    opened = p[(p != 0) & (p.shift(1) == 0)]
    bad = [d for d in opened.index if abs(z.loc[d]) <= ENTRY_THRESHOLD]
    check("every entry happened with |z| > ENTRY_THRESHOLD", len(bad) == 0,
          f"{len(bad)} entries fired below threshold")

    # sign convention: z above +entry means the spread is too WIDE -> short it.
    wrong = [d for d in opened.index if np.sign(opened.loc[d]) == np.sign(z.loc[d])]
    check("entry sign is opposite to the z-score sign", len(wrong) == 0,
          f"{len(wrong)} entries had the wrong sign")


# --- 7. engine: the t+1 rule --------------------------------------------------

def test_no_lookahead():
    print("\n[7] engine cannot see the future")

    s = _series(np.cumsum(RNG.normal(0, 1, 300)))
    p = positions(zscore(s))

    vec = backtest_pair(s, p)
    loop = backtest_pair_loop(s, p)
    check("vectorized engine == explicit loop",
          np.allclose(vec["net_pnl"].to_numpy(), loop["net_pnl"].to_numpy(), atol=1e-9),
          f"max diff {np.abs(vec['net_pnl'] - loop['net_pnl']).max():.2e}")

    # the t+1 rule stated directly: today's P&L must be YESTERDAY's position
    # times TODAY's spread move. If this ever equals today's position instead,
    # the backtest is earning the very move that triggered its own signal.
    expected = (p.shift(1) * s.diff()).fillna(0.0)
    check("gross_pnl[t] == positions[t-1] * spread_change[t]",
          np.allclose(vec["gross_pnl"].to_numpy(), expected.to_numpy(), atol=1e-9))

    lookahead = (p * s.diff()).fillna(0.0)
    check("gross_pnl[t] is NOT positions[t] * spread_change[t]",
          not np.allclose(vec["gross_pnl"].to_numpy(), lookahead.to_numpy(), atol=1e-9),
          "engine is using same-day positions -- look-ahead bias")

    # costs must be charged on every position change, never for free.
    changes = int((p.diff().abs() > 0).sum())
    check("a cost is charged on every position change",
          np.isclose(vec["costs"].sum(), changes * 0.05, rtol=1e-6),
          f"{changes} changes, costs {vec['costs'].sum():.4f}")


# --- 8. config: single source of truth ----------------------------------------

def test_config_single_source():
    print("\n[8] MODE is defined in exactly one place")

    from backtest import portfolio, run_all_pairs, report
    same = (portfolio.MODE == config.MODE
            and run_all_pairs.MODE == config.MODE
            and report.MODE == config.MODE)
    check("run_all_pairs, portfolio, report all read config.MODE", same,
          f"config={config.MODE} run_all_pairs={run_all_pairs.MODE} "
          f"portfolio={portfolio.MODE} report={report.MODE}")

    check("MODE is a legal value", config.MODE in ("full", "split", "rolling"),
          f"got {config.MODE!r}")


# --- 9. real data: the gate actually bites ------------------------------------

def test_real_data_gate():
    print("\n[9] real data: the pair that broke everything is now rejected")

    tickers = ["SHREECEM", "ACC", "COALINDIA", "ONGC", "HAVELLS", "CROMPTON"]
    close = load_panel(FOLDER, tickers=tickers)

    corr = correlation(close["SHREECEM"], close["ACC"])
    beta = hedge_ratio(close["SHREECEM"], close["ACC"])
    ok, reason = pair_is_valid(close["SHREECEM"], close["ACC"], beta)
    check("SHREECEM-ACC is rejected by the gate", not ok,
          f"corr={corr:.3f}, beta={beta:.3f}, verdict={reason}")

    # and it must be rejected in ALL THREE modes, not just the screening step.
    for name, fn in (("full", run_full), ("split", run_split), ("rolling", run_rolling)):
        check(f"SHREECEM-ACC not tradeable in {name} mode",
              fn(close, "SHREECEM", "ACC") is None)

    # a genuine pair must survive the gate in every mode.
    for name, fn in (("full", run_full), ("split", run_split), ("rolling", run_rolling)):
        check(f"COALINDIA-ONGC still tradeable in {name} mode",
              fn(close, "COALINDIA", "ONGC") is not None)

    # HAVELLS-CROMPTON: beta was 42% inflated under raw TLS (4.44 vs 3.12 OLS).
    beta_hc = hedge_ratio(close["HAVELLS"], close["CROMPTON"])
    check("HAVELLS-CROMPTON beta is no longer inflated toward 4.44",
          3.0 < beta_hc < 4.1, f"got {beta_hc:.3f}")


# --- 10. margin now accounts for the hedge ratio ------------------------------

def test_margin_uses_beta():
    print("\n[10] margin reflects both legs, not two copies of leg A")

    close = load_panel(FOLDER, tickers=["HAVELLS", "CROMPTON"])
    out = run_split(close, "HAVELLS", "CROMPTON")
    if out is None:
        check("HAVELLS-CROMPTON tradeable in split mode", False,
              "pair was rejected -- cannot check margin")
        return

    _sh, _pnl, _eq, margin = out
    beta = hedge_ratio(close[close.index < "2025-01-01"]["HAVELLS"],
                       close[close.index < "2025-01-01"]["CROMPTON"])

    trading = close[close.index >= "2025-01-01"]
    expected = (trading["HAVELLS"].mean() + abs(beta) * trading["CROMPTON"].mean()) * 0.20
    old_formula = trading["HAVELLS"].mean() * 2 * 0.20

    check("margin == (mean(A) + beta*mean(B)) * 0.20",
          np.isclose(margin, expected, rtol=1e-6),
          f"got {margin:.1f}, expected {expected:.1f}")
    check("margin differs from the old 2*mean(A) formula",
          not np.isclose(margin, old_formula, rtol=1e-3),
          f"new {margin:.1f} vs old {old_formula:.1f}")


# --- 11. rolling: the formation window is trailing, not expanding -------------

def test_rolling_window_is_trailing():
    print("\n[11] rolling formation window is trailing (was silently expanding)")

    close = load_panel(FOLDER, tickers=["COALINDIA", "ONGC"])

    out12 = run_rolling(close, "COALINDIA", "ONGC", formation_months=12, step_months=3)
    out24 = run_rolling(close, "COALINDIA", "ONGC", formation_months=24, step_months=3)

    if out12 is None or out24 is None:
        check("both rolling variants produced results", False,
              "one variant returned None -- cannot compare")
        return

    # if formation were still expanding, formation_months would only shift the
    # START of trading; the betas fitted in overlapping windows would be identical
    # and the late-period P&L would coincide. It must not.
    eq12, eq24 = out12[2], out24[2]
    overlap = eq12.index.intersection(eq24.index)
    check("12mo and 24mo formation give different results", len(overlap) > 0
          and not np.allclose(eq12.loc[overlap].to_numpy(),
                              eq24.loc[overlap].to_numpy(), atol=1e-6),
          "identical equity on the overlap -- formation window is not trailing")

    check("12mo formation starts trading earlier than 24mo",
          eq12.index.min() < eq24.index.min(),
          f"12mo starts {eq12.index.min().date()}, 24mo starts {eq24.index.min().date()}")


if __name__ == "__main__":
    print("=" * 68)
    print("pipeline verification")
    print(f"  MIN_CORRELATION = {MIN_CORRELATION}   MIN_BETA = {MIN_BETA}")
    print(f"  config.MODE = {config.MODE!r}   "
          f"FORMATION_MONTHS = {config.FORMATION_MONTHS}   "
          f"STEP_MONTHS = {config.STEP_MONTHS}")
    print("=" * 68)

    test_hedge_ratio_known_answer()
    test_hedge_ratio_symmetry()
    test_hedge_ratio_scale()
    test_gate()
    test_zscore_scale_invariant()
    test_positions()
    test_no_lookahead()
    test_config_single_source()
    test_real_data_gate()
    test_margin_uses_beta()
    test_rolling_window_is_trailing()

    print("\n" + "=" * 68)
    print(f"  {_passed} passed, {_failed} failed")
    print("=" * 68)
    if _failed:
        print("\nNOTE: a failure here does not necessarily mean the code is wrong --")
        print("it may mean an assumption in the test is wrong. Read the detail line")
        print("and decide which one is at fault before changing anything.")
    sys.exit(1 if _failed else 0)
