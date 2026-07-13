# config.py
# Single source of truth for constants used in multiple classes
#  MODE, WEIGHT_SCHEME, and COST_PER_UNIT, imported by
# run_all_pairs.py, portfolio.py, report.py, and validation_methods.py. 

# "full"    - fits and trades on ALL data; in-sample, not possible for real trading.
# "split"   - fits once on formation data, trades the rest untouched; first honest test.
# "rolling" - re-fits every STEP_MONTHS on trailing data; closest to how this runs live.
MODE = "rolling"

# Rolling-window shape, used ONLY by run_rolling. These two constants fully define
# the rolling variant -- there is no separate "three_month_rolling" mode, it is just
# FORMATION_MONTHS = 12 with STEP_MONTHS = 3.
#
#   FORMATION_MONTHS - trailing window used to test cointegration and fit beta.
#                      Shorter = beta stays current, but ADF has less power and
#                      rejects more pairs. 24 -> ~500 bars, 12 -> ~250 bars.
#   STEP_MONTHS      - how far forward each fitted beta is traded before re-fitting.
#
# NOTE: results are written to results_{MODE}.csv / daily_pnl_{MODE}.csv, so two
# rolling variants OVERWRITE each other. To compare 24mo vs 12mo, copy the outputs
# aside between runs.
FORMATION_MONTHS = 12
STEP_MONTHS = 3

# "equal_weight" - every pair gets the same nominal position size, ignoring volatility.
# "equal_risk"   - pairs are inverse-vol weighted so each contributes similar risk, not size.
WEIGHT_SCHEME = "equal_risk"

# Cost charged per unit of spread traded, on every position change (entry/exit/flip).
COST_PER_UNIT = 0.05
