Starting data:

date spread positions
Mon 100 0
Tue 90 +1
Wed 95 +1
Thu 100 0

Step 1: spread_returns = spread.diff()

date spread_returns
Mon NaN
Tue -10
Wed +5
Thu +5

Step 2: positions.shift(1)

date positions.shift(1)
Mon NaN
Tue 0
Wed +1
Thu +1

Step 3: gross_pnl = positions.shift(1) \* spread_returns

date gross_pnl
Mon NaN
Tue 0 (0 _ -10)
Wed +5 (+1 _ +5)
Thu +5 (+1 \* +5)

Step 4: trades = positions.diff().abs() (how much the position changed)

date positions diff abs = trades
Mon 0 NaN NaN
Tue +1 +1 1 (went 0 -> +1, traded 1 unit)
Wed +1 0 0 (no change, held)
Thu 0 -1 1 (went +1 -> 0, traded 1 unit)

Step 5: costs = trades \* cost_per_unit (say cost = 0.05)

date trades costs
Mon NaN NaN
Tue 1 0.05
Wed 0 0.00
Thu 1 0.05

Step 6: net_pnl = gross_pnl - costs

date gross_pnl costs net_pnl
Mon NaN NaN NaN
Tue 0 0.05 -0.05
Wed +5 0.00 +5.00
Thu +5 0.05 +4.95

Step 7: .fillna(0.0) replaces the NaN rows with 0

date gross_pnl costs net_pnl
Mon 0.0 0.0 0.00
Tue 0.0 0.05 -0.05
Wed 5.0 0.00 +5.00
Thu 5.0 0.05 +4.95

Step 8: equity = net_pnl.cumsum() (running total)

date net_pnl equity
Mon 0.00 0.00
Tue -0.05 -0.05
Wed +5.00 +4.95
Thu +4.95 +9.90

Final result DataFrame:

date gross_pnl costs net_pnl equity
Mon 0.0 0.00 0.00 0.00
Tue 0.0 0.05 -0.05 -0.05
Wed 5.0 0.00 5.00 4.95
Thu 5.0 0.05 4.95 9.90
