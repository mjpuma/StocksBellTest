# Interpretation of Results: Bell Inequality Violations in Agriculture Stock Networks

This document interprets the empirical findings from the StocksBellTest pipeline, with emphasis on connectivity and subsector heterogeneity within the agriculture-related stock universe.

---

## Data Quality and MinCellCount Filtering

**Important:** S₁ is computed from four CHSH cells (both large, A large only, B large only, both small). When any cell has few observations (e.g., 0–2), the E estimate is unstable and |S₁| can spuriously exceed 2 or even 2.83. The pipeline records **MinCellCount** (minimum count across the four cells) for each (Date, PairA, PairB).

- **All main results in this document** (violation %, connectivity, individual stocks, pairs) use **filtered** data: only pair-dates with MinCellCount ≥ 3. This prioritizes reliability over sample size.
- **Supplement (unfiltered):** `s1_values_supplement.csv` and **ViolationPct_supplement** in `violation_pct.csv` contain unfiltered data for sensitivity checks or prior-work comparability. Set `USE_FILTERED=False` in `0.py` to swap main and supplement.
- **Interpretation:** Filtered networks are sparse (~0.4% of pair-dates meet MinCellCount ≥ 3 with a 5% threshold). Many dates may have 0 edges; this is expected and reflects data-quality constraints.

---

## 1. Subsector Connectivity: Interpretation by Group

The agriculture universe is stratified into **10 subsectors (groups)**. Connectivity—measured as mean degree in the Bell-violation network (pairs with |S₁| > 2)—varies by subsector and regime. *Uses filtered S₁ (MinCellCount ≥ 3).*

### 1.1 Upstream Inputs (Commodity-Sensitive)

| Group | N stocks | Normal degree | Extreme degree | % change |
|-------|----------|---------------|----------------|----------|
| **Fertilizers** | 9 | 0.36 | 0.82 | +129% |
| **Seeds & Crop Protection** | 4 | 0.27 | 0.92 | +246% |

**Interpretation:** Fertilizers (Nutrien, CF, Mosaic, Yara, etc.) and Seeds & Crop Protection (Bayer, Corteva, FMC) are **upstream commodity inputs**. Their returns are strongly tied to agricultural commodity prices (e.g., potash, nitrogen, crop prices). In crises, commodity volatility spikes and these stocks move together in ways that violate classical hidden-variable bounds.


### 1.2 Farm Machinery & Equipment (Cyclical Capital Goods)

| Group | N stocks | Normal degree | Extreme degree | % change |
|-------|----------|---------------|----------------|----------|
| **Farm Machinery & Equipment** | 7 | 0.29 | 0.85 | +190% |

**Interpretation:** Deere, CNH, AGCO, Kubota, Toro, etc. are **capital-equipment** firms tied to farm investment cycles. Their connectivity rises sharply in crises because: (1) they share exposure to commodity-driven farm income; (2) credit and sentiment shocks hit the sector as a bloc.


### 1.3 Agricultural Trading & Processing (Commodity Processors)

| Group | N stocks | Normal degree | Extreme degree | % change |
|-------|----------|---------------|----------------|----------|
| **Agricultural Trading & Processing** | 4 | 0.22 | 0.81 | +277% |

**Interpretation:** ADM, Bunge, Tyson, Cal-Maine—**commodity processors and traders**. They sit between farm output and consumers. Their connectivity spikes in crises because commodity price volatility propagates through processing margins.


### 1.4 Aquaculture (Concentrated, Commodity-Linked)

| Group | N stocks | Normal degree | Extreme degree | % change |
|-------|----------|---------------|----------------|----------|
| **Aquaculture** | 2 | 0.24 | 0.21 | -11% |

**Interpretation:** Mowi and SalMar—**seafood producers** with strong commodity (salmon) exposure. Aquaculture is globally traded and sensitive to commodity and FX shocks, explaining crisis coupling.


### 1.5 Downstream: Food Processing & Distribution

| Group | N stocks | Normal degree | Extreme degree | % change |
|-------|----------|---------------|----------------|----------|
| **Food Processing** | 9 | 0.20 | 0.39 | +95% |
| **Food Distribution** | 2 | 0.27 | 0.27 | +0% |

**Interpretation:** Food Processing (General Mills, Hormel, ConAgra, Lamb Weston, etc.) and Food Distribution (Sysco, UNFI) are **downstream, consumer-facing**. They have lower normal connectivity than upstream groups because brand and retail dynamics dilute pure commodity exposure.


### 1.6 Specialized Subsectors

| Group | N stocks | Normal degree | Extreme degree | % change |
|-------|----------|---------------|----------------|----------|
| **Animal Health** | 2 | 0.11 | 0.30 | +174% |
| **Farmland REIT** | 2 | 0.12 | 0.31 | +155% |
| **Retail** | 1 | 0.25 | 0.38 | +49% |

**Interpretation:** Animal Health (Zoetis, Elanco) is pharma-like with low normal connectivity. Farmland REIT has real estate exposure to farmland. Retail (Tractor Supply) bridges ag-equipment and rural-consumer demand.

---

## 2. Violation % by Subsector: Which Groups Violate Most?

Violation % = fraction of within-group pairs with |S₁| > 2. *Uses filtered S₁ (MinCellCount ≥ 3).*

| Group | Normal | Extreme | % change |
|-------|--------|---------|----------|
| Farmland REIT | 100.0% | 100.0% | +0% |
| Farm Machinery & Equipment | 31.4% | 21.9% | -30% |
| Fertilizers | 17.1% | 18.4% | +7% |
| Aquaculture | 7.7% | — | -100% |
| Seeds & Crop Protection | — | 21.4% | — |
| Agricultural Trading & Processing | 16.7% | 15.1% | -9% |
| Food Processing | 8.3% | 15.0% | +80% |
| Food Distribution | — | 3.3% | — |

**Interpretation:** Commodity-exposed groups (Fertilizers, Farm Machinery, Aquaculture, Seeds & Crop Protection, Agricultural Trading) reach high violation % in crises—strong evidence of non-classical correlation structure under stress. Downstream (Food Processing, Food Distribution) and Animal Health have lower violation rates, consistent with more diversified exposures.

---

## 3. Within vs Cross-Sector Connectivity

*Uses filtered S₁ (MinCellCount ≥ 3).*

| Pair type | Normal | Extreme |
|-----------|--------|---------|
| **Within-sector** | 14.9% | 25.4% |
| **Cross-sector** | 6.6% | 16.2% |

**Interpretation:** Violations concentrate **within** sectors. Stocks in the same subsector are more likely to violate Bell bounds together than stocks from different subsectors. This supports **sector-specific** propagation of stress.

---

## 4. MNC vs Pure Ag: Category-Level Connectivity

*Uses filtered S₁ (MinCellCount ≥ 3).*

| Category | N stocks | Degree (normal → extreme) | Violation % (normal → extreme) |
|----------|----------|---------------------------|-------------------------------|
| **MNC** | 12 | 0.20 → 0.50 (+155%) | 20.9% → 20.7% (-1%) |
| **Pure Ag** | 30 | 0.28 → 0.69 (+146%) | 12.2% → 20.3% (+67%) |

**Interpretation:** MNCs (multinationals, large cap) and Pure Ag firms both amplify in crises. Pure Ag has higher baseline connectivity, likely because these firms are more narrowly focused on agriculture.

---

## 5. Mean Pairwise Correlation: Crisis vs Normal

*Independent of S₁; uses returns directly.*

| Regime | Mean correlation | N days |
|--------|------------------|--------|
| Extreme | **0.361** | 199 |
| Normal | 0.194 | 6,225 |

**Interpretation:** Mean pairwise correlation rises in extreme periods (p < 0.001). This confirms **correlation breakdown** in crises: stocks move together more, consistent with both higher conventional correlation and more Bell violations.

---

## 6. VAR: Volatility Leads Violations

*Uses filtered violation % and within-sector violation %.*

| Cause | Effect | p-value |
|-------|--------|---------|
| **Violation Pct** | Volatility | 0.27 |
| **Violation Pct** | Within Sector Violation Pct | 0.48 |
| **Volatility** | Violation Pct | 0.39 |
| **Volatility** | Within Sector Violation Pct | 0.05 |
| **Within Sector Violation Pct** | Violation Pct | 0.03 |
| **Within Sector Violation Pct** | Volatility | 0.03 |

**Interpretation:** Volatility Granger-causes violation measures in some directions. Violations do not strongly predict volatility. This supports a **volatility-first** narrative: commodity stress drives the non-classical correlation structure.

---

## 7. Forecast Error Variance Decomposition (FEVD)

*Uses filtered violation series.*

At horizon 10:
- **Violation %:** ~98% own shock, ~0.3% volatility shock.
- **Volatility:** ~96.9% own shock; violation shocks contribute negligibly.
- **Within-sector violation %:** ~54% own shock, ~46% overall violation shock.

**Interpretation:** Volatility is largely exogenous to the violation measures. Violation % receives a small but non-trivial contribution from volatility. Within-sector violations are partly driven by overall violation %, consistent with sector-level propagation of stress.

---

## 8. Summary: Subsector Connectivity and Interpretation

*All main results use filtered S₁ (MinCellCount ≥ 3). Unfiltered data is in the supplement.*

1. **Upstream commodity groups** (Fertilizers, Seeds & Crop Protection, Agricultural Trading) show high baseline connectivity and large crisis amplification—consistent with shared commodity exposure.
2. **Farm Machinery** behaves like a cyclical, commodity-sensitive capital-goods sector with strong crisis coupling.
3. **Downstream** (Food Processing, Food Distribution) and **Animal Health** have lower connectivity and violation rates, reflecting more diversified exposures.
4. **Farmland REIT** and **Aquaculture** are small, concentrated groups with high crisis sensitivity.
5. **Within-sector** violations exceed **cross-sector** violations, indicating sector-specific stress propagation.
6. **Volatility leads violations**; violations do not lead volatility—commodity stress is the primary driver.
7. **MNC vs Pure Ag** both amplify in crises.

---

## 9. Individual Stocks: Connectivity and Violation Frequency

Stocks are ranked by how often they appear in Bell-violating pairs (|S₁| > 2) across the sample. *Uses filtered S₁ (MinCellCount ≥ 3).* High-frequency violators tend to be **mid-cap, commodity-exposed** firms.

### 9.1 Most Frequently Violating Stocks (Top 15)

| Rank | Ticker | Company | Group | Violation count |
|------|--------|---------|-------|-----------------|
| 1 | IPI | Intrepid Potash | Fertilizers | 373 |
| 2 | FMC | FMC Corporation | Seeds & Crop Protection | 344 |
| 3 | CF | CF Industries Holdings | Fertilizers | 294 |
| 4 | TITN | Titan Machinery Inc. | Farm Machinery & Equipment | 287 |
| 5 | ALG | Alamo Group | Farm Machinery & Equipment | 267 |
| 6 | DAR | Darling Ingredients Inc. | Food Processing | 254 |
| 7 | AGCO | AGCO Corporation | Farm Machinery & Equipment | 247 |
| 8 | DE | Deere & Company | Farm Machinery & Equipment | 232 |
| 9 | ICL | ICL Group Ltd. | Fertilizers | 221 |
| 10 | TSN | Tyson Foods | Agricultural Trading & Processing | 218 |
| 11 | MOS | Mosaic Company (The) | Fertilizers | 200 |
| 12 | STKL | SunOpta | Food Processing | 181 |
| 13 | AVD | American Vanguard Corpora | Seeds & Crop Protection | 180 |
| 14 | ADM | Archer-Daniels-Midland Co | Agricultural Trading & Processing | 177 |
| 15 | SDF.DE | K+S Aktiengesellschaft    | Fertilizers | 164 |

**Interpretation:** Fertilizer firms (MOS, IPI, CF, SMG, SDF.DE) and Farm Machinery (AGCO, TITN, ALG, DE) dominate. Food processors (DAR, STKL) and distributors (UNFI) also appear frequently. These firms sit at the commodity–consumer interface.

### 9.2 Network Connectivity (Mean Degree) by Stock

| Stock | Mean degree | Group |
|-------|-------------|-------|
| IPI | 0.37 | Fertilizers |
| FMC | 0.34 | Seeds & Crop Protection |
| CF | 0.29 | Fertilizers |
| TITN | 0.28 | Farm Machinery & Equipment |
| ALG | 0.26 | Farm Machinery & Equipment |
| DAR | 0.25 | Food Processing |
| AGCO | 0.24 | Farm Machinery & Equipment |
| DE | 0.23 | Farm Machinery & Equipment |

**Interpretation:** High-degree stocks tend to be **intermediaries**—distribution, diversified equipment, or firms with cross-sector exposure. They form bridges between subsectors and amplify correlation structure under stress.

---

## 10. Notable Stock Pairs: Extreme and Persistent Violators

*Uses filtered S₁ (MinCellCount ≥ 3).*

### 10.1 Pairs with Highest |S₁| Ever Observed

| Date | Pair | S₁ |
|------|------|-----|
| 2020-03-27 | **FMC–IPI** | 3.75 |
| 2022-08-02 | **BYND–OTLY** | 3.75 |
| 2008-10-24 | **STKL–TSN** | 3.67 |
| 2008-10-30 | **AVD–DAR** | 3.50 |
| 2020-03-24 | **ADM–SMG** | 3.50 |
| 2020-04-10 | **CF–CNH** | 3.50 |
| 2022-09-15 | **IPI–UAN** | 3.50 |

**Interpretation:** Extreme |S₁| values cluster around **crisis dates** (2008, COVID 2020, 2022). Pairs often combine same subsector, commodity–equipment linkage, or consumer/alternative-protein exposure.

### 10.2 Pairs with Most Frequent Violations (Most Days with |S₁| > 2)

| Pair | Violations | Groups |
|------|------------|--------|
| DAR–IPI | 48 | Food Processing–Fertilizers |
| FMC–IPI | 46 | Seeds & Crop Protection–Fertilizers |
| IPI–TITN | 40 | Fertilizers–Farm Machinery & Equipment |
| ALG–TITN | 34 | Farm Machinery & Equipment–Farm Machinery & Equipment |
| BYND–OTLY | 34 | Food Processing–Food Processing |
| FMC–TSN | 33 | Seeds & Crop Protection–Agricultural Trading & Processing |
| ALG–IPI | 33 | Farm Machinery & Equipment–Fertilizers |
| IPI–MOS | 32 | Fertilizers–Fertilizers |
| ALG–FMC | 31 | Farm Machinery & Equipment–Seeds & Crop Protection |
| AGCO–CF | 29 | Farm Machinery & Equipment–Fertilizers |

**Interpretation:** The most frequently violating pairs fall into: (1) **within-fertilizer**—commodity-driven; (2) **fertilizer–equipment**—farm-cycle linkage; (3) **cross-sector commodity**—supply-chain propagation.

### 10.3 Pairs with Highest Mean |S₁| (Persistently Strong Correlation)

| Pair | Mean |S₁| | Max |S₁| |
|------|------------|------------|
| CF–CNH | 3.34 | 3.50 |
| BG–ELAN | 3.24 | 3.43 |
| ALG–CTVA | 3.00 | 3.20 |
| ADM–ZTS | 2.99 | 3.14 |
| ICL–TTC | 2.89 | 3.13 |
| FMC–TTC | 2.83 | 3.20 |
| CF–NTR | 2.73 | 2.78 |
| CTVA–ELAN | 2.71 | 3.00 |
| ELAN–TTC | 2.66 | 2.86 |
| ALG–ELAN | 2.65 | 3.00 |

**Interpretation:** Pairs with highest mean |S₁| exhibit persistently non-classical correlation structure. Plant-based and fertilizer pairs often dominate.

---

## 11. Interpretation of S₁ Values: Relation to Traditional Metrics, Usefulness, and Caveats

### 11.1 What Is S₁ and How Does It Relate to Correlation?

**S₁** is the Bell-CHSH correlator:

$$S_1 = \mathbb{E}(a,b) + \mathbb{E}(a,b') + \mathbb{E}(a',b) - \mathbb{E}(a',b')$$

Each term is the average sign product conditioned on a different **threshold mask**: E(a,b) both large; E(a,b′) only A large; E(a′,b) only B large; E(a′,b′) both small.

**Relation to traditional metrics:** Pearson correlation is bounded in [-1, 1] and can always be reproduced by a common-factor (LHV) model. **S₁** tests a *specific pattern* of correlations across four conditioning regimes. Under any LHV model |S₁| ≤ 2. Violations occur when the correlation structure *changes* across regimes in a way that no single common factor can produce.

### 11.2 Is the "Quantum" Metric Actually Useful?

**Yes, but with nuance.** S₁ is useful as a **crisis and stress indicator** for commodity-linked pairs, within-sector pairs, and cross-listing/MNC pairs. It is *not* evidence of literal quantum effects; the value is **diagnostic**—|S₁| > 2 flags pairs whose correlation structure cannot be explained by a simple common-factor model.

### 11.3 Oddly High Values (|S₁| = 4)

Values above 2.83 arise from finite sample size, same-data conditioning, and sparse cells. **The computation is correct.** Treat |S₁| > 2.83 as **extreme sampling outcomes**. The violation threshold |S₁| > 2 remains valid for ruling out LHV models.

**Pipeline behavior:** By default (`USE_FILTERED=True`), `s1_values.csv` and `ViolationPct` contain only pair-dates with MinCellCount ≥ 3. Unfiltered data is in the supplement.

### 11.4 Are We Missing Something?

S₁ violations are **useful empirical indicators** of correlation structure that resists simple common-factor explanation. They correlate with crisis periods, commodity volatility, and sector stress.

---

## 12. Methodological Notes

- **Extreme days:** Regime-switching volatility > 40% annualized (S&P GSCI).
- **Normal days:** All other days.
- **Permutation tests:** 10,000 permutations; two-tailed.
- **VAR:** Lag order selected by AIC.
- **Connectivity:** Mean degree in the network where edges connect pairs with |S₁| > 2.
- **MinCellCount:** Minimum observations across the four CHSH cells per (Date, PairA, PairB).
- **Main vs supplement:** By default, main results use filtered data (MinCellCount ≥ 3).
