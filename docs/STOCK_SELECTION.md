# Agriculture Stock Selection: Objective Reference

## Summary

**Current state:** [yfinance_tickers.csv](../yfinance_tickers.csv) has 55 tickers with manual sector/exposure. It mixes ag-focused firms with broad conglomerates (WMT, COST, XOM, UL, NSRGY) and tangential sectors (Energy, Retail, Chemicals, Transportation).

**Objective approach:** Use **MOO (VanEck Agribusiness ETF)** constituents as the canonical agriculture universe. MOO tracks companies in agri-chemicals, animal health, fertilizers, farm equipment, aquaculture, livestock, and agricultural product trading.

---

## MOO Constituents (as of Feb 2026)

MOO holds **53 stocks**. Top 25 from [stockanalysis.com](https://stockanalysis.com/etf/moo/holdings/):

| # | Symbol | Name | Group (proposed) |
|---|--------|------|-----------------|
| 1 | DE | Deere & Company | Farm Machinery & Equipment |
| 2 | ZTS | Zoetis Inc. | Animal Health |
| 3 | BAYN.DE | Bayer AG | Seeds & Crop Protection |
| 4 | CTVA | Corteva, Inc. | Seeds & Crop Protection |
| 5 | NTR | Nutrien Ltd | Fertilizers |
| 6 | ADM | Archer-Daniels-Midland | Agricultural Trading & Processing |
| 7 | TSN | Tyson Foods | Food Processing |
| 8 | 6326.T | Kubota Corporation | Farm Machinery & Equipment |
| 9 | CF | CF Industries | Fertilizers |
| 10 | BG | Bunge Global | Agricultural Trading & Processing |
| 11 | MOWI.OL | Mowi ASA | Aquaculture |
| 12 | Wilmar (SGX) | Wilmar International | Agricultural Trading & Processing |
| 13 | YAR.OL | Yara International | Fertilizers |
| 14 | CNH | CNH Industrial | Farm Machinery & Equipment |
| 15 | DAR | Darling Ingredients | Food Processing / Rendering |
| 16 | TTC | The Toro Company | Farm Machinery & Equipment |
| 17 | ELAN | Elanco Animal Health | Animal Health |
| 18 | MOS | The Mosaic Company | Fertilizers |
| 19 | AGCO | AGCO Corporation | Farm Machinery & Equipment |
| 20 | SalMar (OSL) | SalMar ASA | Aquaculture |
| 21 | China Mengniu (HKG) | China Mengniu Dairy | Food Processing |
| 22 | NH Foods (TYO) | NH Foods Ltd | Food Processing |
| 23 | CPF-R.BK | Charoen Pokphand Foods | Food Processing |
| 24 | SDF (ETR) | K+S AG | Fertilizers |
| 25 | CALM | Cal-Maine Foods | Food Processing |

**US-listed tickers from MOO (yfinance-compatible):**  
DE, ZTS, CTVA, NTR, ADM, TSN, CF, BG, CNH, DAR, TTC, ELAN, MOS, AGCO, CALM

**International (require exchange suffix for yfinance):** BAYN.DE, 6326.T, MOWI.OL, YAR.OL

---

## Objective Grouping (GICS-based)

From `yf.Ticker(t).info`:

| Group | GICS Industry (yfinance) | Example tickers |
|-------|--------------------------|-----------------|
| **Fertilizers** | Agricultural Inputs (fertilizer focus) | NTR, CF, MOS, ICL, IPI, SMG, UAN |
| **Seeds & Crop Protection** | Agricultural Inputs (seeds/chem) | CTVA, FMC, AVD |
| **Farm Machinery & Equipment** | Farm & Heavy Construction Machinery | DE, AGCO, CNH, TTC, TITN, ALG |
| **Animal Health** | Drug Manufacturers - Specialty | ZTS, ELAN |
| **Agricultural Trading & Processing** | Farm Products, Agricultural Products | ADM, BG |
| **Food Processing** | Packaged Foods, Meat, Farm Products | TSN, CALM, LW, HRL, CPB, GIS, CAG |
| **Food Distribution** | Food Distribution | SYY, UNFI |
| **Farmland REIT** | REIT - Specialty | LAND, FPI |
| **Aquaculture** | Farm Products (seafood) | MOWI, SalMar |
| **Alternative Proteins** | Packaged Foods (plant-based) | BYND, OTLY |

---

## Current List vs MOO

### In current list AND in MOO (keep)
DE, CTVA, ADM, NTR, CF, BG, MOS, AGCO, TSN, TTC, CNH, CALM

### In current list, NOT in MOO (consider removing for focused ag)
- **Energy:** CVX, XOM, COP  
- **Retail:** WMT, COST, KR, TSCO  
- **Chemicals (broad):** DOW, LYB, BASFY  
- **Transportation:** UNP, CNI, NSC, CSX  
- **Conglomerates:** UL, NSRGY  

### In MOO, NOT in current list (consider adding)
- ZTS (Zoetis), ELAN (Elanco) — animal health  
- DAR (Darling Ingredients) — rendering/ingredients  
- BAYN (Bayer) — seeds/crop protection  

### Edge cases (ag-adjacent, judgment call)
- SYY (Sysco), UNFI — food distribution  
- TSCO (Tractor Supply) — ag retail  
- FMC, AVD, ICL — crop protection / fertilizers (ag inputs)  
- LW, HRL, CPB, GIS, CAG, HSY, SJM, MDLZ — packaged foods (downstream)  
- LAND, FPI — farmland REITs  
- BYND, OTLY — alternative proteins  

---

## Recommended Objective List

**Option A — MOO US-only (tightest):**  
DE, ZTS, CTVA, NTR, ADM, TSN, CF, BG, CNH, DAR, TTC, ELAN, MOS, AGCO, CALM  
(~15 tickers; add BAYN.DE, YAR.OL if including ADRs)

**Option B — MOO US + ag-adjacent from current:**  
Option A + FMC, AVD, ICL, LW, SYY, LAND, FPI, BYND (if data available)  
(~25 tickers)

**Option C — Current list minus clear non-ag:**  
Remove: CVX, XOM, COP, WMT, COST, KR, DOW, LYB, UNP, CNI, NSC, CSX, UL, NSRGY  
Keep: all Equipment, Fertilizers, Seeds/Crop Protection, Trading, Food Processing, Farmland REIT, Alternative Proteins, TSCO, SYY, UNFI, BASFY  
(~40 tickers)

---

## Data Sources

- **MOO holdings:** [stockanalysis.com/etf/moo/holdings](https://stockanalysis.com/etf/moo/holdings) (updated regularly)
- **yfinance:** `yf.Ticker("MOO").funds_data.top_holdings` — top 10 only
- **Full list:** Manual from VanEck/Morningstar, or subscription (e.g., stockanalysis Pro)

---

## Next Steps

1. **Decide scope:** Option A, B, or C (or custom).
2. **Build script:** `scripts/build_ticker_universe.py` to fetch MOO top N + optional filters.
3. **Snapshot:** Commit final `yfinance_tickers.csv` with date for paper reproducibility.
4. **Update ANALYSIS.md:** Document selection methodology in Section 1.1.
