# statarb-engine

A market-neutral statistical-arbitrage (pairs-trading) research framework built
on equity OHLCV data (works on Nifty or S&P).

**Approach:** find same-sector stock pairs whose price *spread* is mean-reverting
(via Engle-Granger cointegration + ADF test, not just correlation), trade
deviations using a z-score signal, and validate out-of-sample with a
formation/trading split to control for data-mining bias.

## Pipeline
- `src/data_layer.py` - load one-CSV-per-stock OHLCV files into aligned panels.
- `src/signals.py` - cointegration screen, spread, z-score *(next)*.
- `src/backtest.py` - look-ahead-safe, cost-aware engine *(next)*.
- `src/evaluate.py` - Sharpe, drawdown, beta-to-index *(next)*.
- `tests/` - including a look-ahead "tripwire" test that proves the engine
  cannot peek at future data.

## Run
```bash
pip install -r requirements.txt
python make_fake_data.py   # generates synthetic CSVs + runs the loader
```

## Roadmap
- [x] Step 1: repo + data loader
- [ ] Step 2: backtest engine + look-ahead tripwire tests
- [ ] Step 3: economic pair selection (same-sector)
- [ ] Step 4: cointegration screen + half-life
- [ ] Step 5: z-score signal
- [ ] Step 6: formation/trading split (out-of-sample)
- [ ] Step 7: full backtest with costs
- [ ] Step 8: evaluation (market-neutrality check)
