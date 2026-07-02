from itertools import combinations

# Produce a list of pairs of stocks which will be tested for cointegraiton

# 
# 1. Add potentially related pairs
#
MANUAL_PAIRS = [
    ("airline_oil", "INDIGO", "BPCL") # fuel is airlines' biggest cost
]

#
# 2. Sort stocks by sector and create pairs from stocks within that sector
#
SECTORS = {
    "banks": ["HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN"],
    "it":    ["TCS", "INFY", "WIPRO"],
    "airline": ["INDIGO", "SPICEJET"],
}

# Pair stocks in the same sector 
def sector_pairs(sectors):
    pairs = []
    for sector, tickers in sectors.items():   
        # Ex. sector = "bank", tickers = ["HDFCBANK", "ICICIBANK","AXISBANK"]
        for a, b in combinations(tickers, 2):  # every pair of 2 in that sector
            pairs.append((sector, a, b))
    return pairs
# pairs = [("banks","HDFCBANK","ICICIBANK"), ("banks","HDFCBANK","AXISBANK"), ...]

# Combine manual pairs + sector pairs into the full list
def all_pairs():
    return MANUAL_PAIRS + sector_pairs(SECTORS)

if __name__ == "__main__":
    print(all_pairs())