# config.py
# Single source of truth for MODE, WEIGHT_SCHEME, and COST_PER_UNIT, imported by
# run_all_pairs.py, portfolio.py, report.py, and validation_methods.py. Change
# values here ONCE.

# "full"    - fits and trades on ALL data; in-sample, optimistic baseline only.
# "split"   - fits once on formation data, trades the rest untouched; first honest test.
# "rolling" - re-fits every STEP_MONTHS on trailing data; closest to how this runs live.
MODE = "split"

# "equal_weight" - every pair gets the same nominal position size, ignoring volatility.
# "equal_risk"   - pairs are inverse-vol weighted so each contributes similar risk, not size.
WEIGHT_SCHEME = "equal_risk"

# Cost charged per unit of spread traded, on every position change (entry/exit/flip).
COST_PER_UNIT = 0.05
