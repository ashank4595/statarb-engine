from itertools import combinations

# Produce a list of pairs of stocks which will be tested for cointegraiton
# Contains MANUAL_PAIRS, SECTORS and all_pairs function

# 
# 1. Add potentially related pairs (cross-sector hypotheses with named economic reasons)
#
MANUAL_PAIRS = [
    ("airline_oil", "INDIGO", "BPCL"),        # fuel is airlines' biggest cost
    ("airline_oil", "INDIGO", "IOC"),          # same reason, different oil marketer
    ("defence", "HAL", "BEL"),                 # both PSU defence manufacturers
    ("exchange", "BSE", "MCX"),                # both financial exchanges
    ("diagnostics", "LALPATHLAB", "METROPOLIS"), # both diagnostic lab chains
]

#
# 2. Sort stocks by sector and create pairs from stocks within that sector
#
SECTORS = {
    # large private + PSU banks — same rate cycle, same RBI regulation
    "banks_large": [
        "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN", "KOTAKBANK", "INDUSINDBK",
    ],
    # PSU banks — same government ownership, similar balance sheet structure
    "banks_psu": [
        "SBIN", "BANKBARODA", "CANBK", "PNB", "UNIONBANK", "INDIANB",
    ],
    # small/mid private banks — similar business models
    "banks_private_mid": [
        "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK", "AUBANK", "CUB",
    ],
    # non-bank financials — NBFCs and HFCs with similar credit exposure
    "financials_nbfc": [
        "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN", "MANAPPURAM",
    ],
    # insurance — life insurers with similar product mix
    "insurance_life": [
        "HDFCLIFE", "SBILIFE", "ICICIPRULI", "LICI",
    ],
    # capital markets — brokers, exchanges, asset managers
    "capital_markets": [
        "ANGELONE", "MOTILALOFS", "NUVAMA", "IIFL",
    ],
    # large IT — same USD revenue, same hiring cycle
    "it_large": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    ],
    # mid IT — similar size, similar verticals
    "it_mid": [
        "MPHASIS", "COFORGE", "PERSISTENT", "LTTS", "KPITTECH",
    ],
    # oil PSUs — same crude input, same govt pricing policy
    "oil_psu": [
        "ONGC", "OIL", "COALINDIA", "GAIL",
    ],
    # oil marketing companies — same downstream fuel retail business
    "oil_marketing": [
        "BPCL", "IOC", "HINDPETRO",
    ],
    # gas distribution — same piped gas business model
    "gas_distribution": [
        "IGL", "MGL", "GUJGASLTD", "ATGL",
    ],
    # power generation — same electricity generation business
    "power": [
        "NTPC", "TATAPOWER", "POWERGRID", "JSWENERGY", "TORNTPOWER",
    ],
    # renewable energy — same solar/wind exposure
    "renewable": [
        "ADANIGREEN", "ADANIENSOL", "SUZLON", "INOXWIND",
    ],
    # autos — same domestic demand cycle, same input costs
    "autos": [
        "MARUTI", "TATAMOTORS", "M&M", "HYUNDAI",
    ],
    # two-wheelers — same segment, same fuel/EV transition pressure
    "two_wheelers": [
        "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", "EICHERMOT",
    ],
    # auto ancillaries — same OEM customer base
    "auto_ancillaries": [
        "MOTHERSON", "APOLLOTYRE", "BALKRISIND", "EXIDEIND", "SONACOMS",
    ],
    # pharma large — same API/formulation business, same US FDA risk
    "pharma_large": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "DIVISLAB",
    ],
    # pharma mid — similar size, similar generic exposure
    "pharma_mid": [
        "TORNTPHARM", "ALKEM", "IPCALAB", "GLENMARK", "GRANULES",
    ],
    # hospitals — same healthcare delivery business
    "hospitals": [
        "APOLLOHOSP", "FORTIS", "MAXHEALTH",
    ],
    # cement — same input costs (coal, power), same infra demand
    "cement": [
        "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT", "JKCEMENT", "RAMCOCEM",
    ],
    # steel — same iron ore input, same infra demand cycle
    "steel": [
        "TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "NMDC",
    ],
    # metals non-ferrous — same commodity price exposure
    "metals_nonferrous": [
        "HINDALCO", "VEDL", "HINDCOPPER", "HINDZINC", "NATIONALUM",
    ],
    # FMCG — same rural/urban consumption demand
    "fmcg": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "COLPAL",
    ],
    # paints — same raw material (TiO2), same housing demand
    "paints": [
        "ASIANPAINT", "BERGEPAINT", "PIDILITIND",
    ],
    # real estate — same property cycle
    "realestate": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA",
    ],
    # infra / construction — same govt capex cycle
    "infra": [
        "LT", "NCC", "IRB", "NBCC",
    ],
    # defence PSU — same govt defence budget
    "defence": [
        "HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP",
    ],
    # telecom — same spectrum, same ARPU trends
    "telecom": [
        "BHARTIARTL", "IDEA",
    ],
    # consumer durables / electricals — same housing/infra demand
    "consumer_durables": [
        "HAVELLS", "CROMPTON", "VOLTAS", "BLUESTARCO", "POLYCAB", "KEI",
    ],
    # industrial / engineering — same capex cycle
    "industrials": [
        "SIEMENS", "ABB", "CUMMINSIND", "BOSCHLTD", "CGPOWER",
    ],
    # logistics — same freight/trade volume
    "logistics": [
        "CONCOR", "DELHIVERY",
    ],
    # hotels — same domestic travel demand
    "hotels": [
        "INDHOTEL", "ZEEL",
    ],
    # airlines — same jet fuel cost, same domestic pax demand
    "airlines": [
        "INDIGO", "GMRAIRPORT",
    ],
    # chemicals — same feedstock costs
    "chemicals": [
        "PIDILITIND", "DEEPAKNTR", "SRF", "NAVINFLUOR", "ATUL",
    ],
    # fertilisers — same gas/urea input costs
    "fertilisers": [
        "COROMANDEL", "CHAMBLFERT", "GNFC",
    ],
    # asset management / registrars — same AUM cycle
    "asset_management": [
        "HDFCAMC", "NAM-INDIA", "CAMS", "KFINTECH",
    ],
}


# Pair stocks in the same sector
def sector_pairs(sectors):
    pairs = []
    for sector, tickers in sectors.items():
        # Ex. sector = "banks_large", tickers = ["HDFCBANK", "ICICIBANK", ...]
        for a, b in combinations(tickers, 2):  # every pair of 2 in that sector
            pairs.append((sector, a, b))
    return pairs
# pairs = [("banks_large","HDFCBANK","ICICIBANK"), ("banks_large","HDFCBANK","AXISBANK"), ...]

# Set to a single (label, A, B) tuple to test just one pair; None uses the full universe.
ONLY_PAIR = ("index_arb", "NIFTY", "BANKNIFTY")

# Combine manual pairs + sector pairs into the full list
def all_pairs():
    if ONLY_PAIR is not None:
        return [ONLY_PAIR]
    return MANUAL_PAIRS + sector_pairs(SECTORS)

if __name__ == "__main__":
    pairs = all_pairs()
    print(f"{len(pairs)} total candidate pairs across {len(SECTORS)} sectors")
    for p in pairs[:10]:
        print(p)
    print("...")
