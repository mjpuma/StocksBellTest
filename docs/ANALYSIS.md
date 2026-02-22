# Methodology: Bell Inequality Violations in Agriculture Equity Networks

This document provides a complete methodology section suitable for an academic paper, including all mathematical definitions, data specifications, and procedural details.

---

## 1. Data and Stock Selection

### 1.1 Equity Universe

We use **55 agriculture-related tickers** from Yahoo Finance, defined in `yfinance_tickers.csv`. Tickers are classified by sector and exposure (Primary, Secondary, Tertiary). At download time, tickers with missing or delisted data are dropped; the analysis typically uses **53 tickers** (e.g., SAFM and K have historically failed to download).

**Date range:** 2000-01-01 through the most recent trading date at run time.

**Full ticker list:**

| Ticker | Name | Sector | Exposure |
|--------|------|--------|----------|
| DE | Deere & Company | Equipment | Primary |
| CTVA | Corteva Inc | Seeds/Crop Protection | Primary |
| ADM | Archer-Daniels-Midland | Trading/Processing | Primary |
| NTR | Nutrien Ltd | Fertilizers | Primary |
| CF | CF Industries | Fertilizers | Primary |
| BG | Bunge Limited | Trading/Processing | Primary |
| MOS | Mosaic Company | Fertilizers | Primary |
| AGCO | AGCO Corporation | Equipment | Primary |
| FMC | FMC Corporation | Crop Protection | Primary |
| TSN | Tyson Foods | Food Processing | Primary |
| LW | Lamb Weston Holdings | Food Processing | Primary |
| TTC | Toro Company | Equipment | Secondary |
| CNH | CNH Industrial | Equipment | Primary |
| SYY | Sysco Corporation | Food Distribution | Secondary |
| AVD | American Vanguard Corp | Crop Protection | Primary |
| IPI | Intrepid Potash | Fertilizers | Primary |
| CALM | Cal-Maine Foods | Food Processing | Primary |
| STKL | SunOpta Inc | Food Processing | Primary |
| ARTW | Arts-Way Manufacturing | Equipment | Primary |
| ALG | Alamo Group | Equipment | Primary |
| TITN | Titan Machinery | Equipment | Primary |
| TSCO | Tractor Supply Company | Retail | Secondary |
| ICL | ICL Group | Fertilizers | Primary |
| CMP | Compass Minerals | Fertilizers | Secondary |
| SMG | Scotts Miracle-Gro | Fertilizers | Primary |
| UAN | CVR Partners | Fertilizers | Primary |
| BASFY | BASF SE (ADR) | Chemicals | Secondary |
| UL | Unilever (ADR) | Food Processing | Tertiary |
| NSRGY | Nestle (ADR) | Food Processing | Tertiary |
| HRL | Hormel Foods | Food Processing | Primary |
| CPB | Campbell Soup | Food Processing | Primary |
| GIS | General Mills | Food Processing | Primary |
| CAG | ConAgra Brands | Food Processing | Primary |
| HSY | Hershey Company | Food Processing | Primary |
| SJM | J.M. Smucker | Food Processing | Primary |
| MDLZ | Mondelez International | Food Processing | Primary |
| UNFI | United Natural Foods | Food Distribution | Secondary |
| KR | Kroger Company | Retail | Tertiary |
| WMT | Walmart Inc | Retail | Tertiary |
| COST | Costco Wholesale | Retail | Tertiary |
| LAND | Gladstone Land Corporation | Farmland REIT | Primary |
| FPI | Farmland Partners Inc | Farmland REIT | Primary |
| CVX | Chevron Corporation | Energy | Tertiary |
| XOM | Exxon Mobil | Energy | Tertiary |
| COP | ConocoPhillips | Energy | Tertiary |
| UNP | Union Pacific | Transportation | Secondary |
| CNI | Canadian National Railway | Transportation | Secondary |
| NSC | Norfolk Southern | Transportation | Secondary |
| CSX | CSX Corporation | Transportation | Secondary |
| DOW | Dow Inc | Chemicals | Secondary |
| LYB | LyondellBasell | Chemicals | Secondary |
| BYND | Beyond Meat | Alternative Proteins | Primary |
| OTLY | Oatly Group | Alternative Proteins | Primary |

### 1.2 Returns

For each ticker, we compute daily **percent change in adjusted close price** (log returns are not used):

$$r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$

where \(P_{i,t}\) is the adjusted close on date \(t\). Rows with missing returns are dropped.

### 1.3 Commodity Volatility Index

We use the **S&P GSCI** (ticker ^SPGSCI) as the commodity volatility benchmark. The S&P GSCI is a world-production-weighted index of 24 physically delivered commodity futures, including Brent Crude, WTI Crude, Heating Oil, RBOB Gasoline, Gasoil, Natural Gas, Aluminum, Copper, Nickel, Lead, Zinc, Gold, Silver, Corn, Soybeans, Chicago Wheat, Kansas Wheat, Cotton, Sugar, Coffee, Cocoa, Live Cattle, Feeder Cattle, and Lean Hogs. WTI is weighted at approximately 25%, Brent at 18%, Corn at 5%, and Live Cattle at 4%.

**Date range for S&P GSCI:** 2000-01-01 to 2025-10-14 (hardcoded in 1.py).

---

## 2. Bell-CHSH S₁ and Violation Percentage

### 2.1 Sliding-Window Construction

For each unordered pair of tickers \((A, B)\), we form a bivariate return series and apply a **rolling window of 20 trading days**. Let \(x_t\) and \(y_t\) denote the returns of tickers A and B on day \(t\). For each window \(w\) ending on date \(t\), we have vectors \(\mathbf{x}_w = (x_{t-19}, \ldots, x_t)\) and \(\mathbf{y}_w = (y_{t-19}, \ldots, y_t)\).

### 2.2 Sign and Threshold Masks

Define the **sign** of each return as \(a_i = \mathrm{sign}(x_i)\) and \(b_i = \mathrm{sign}(y_i)\). We use a **fixed threshold** \(\tau = 0.05\) (5%) to define binary outcomes:

- **Above threshold:** \(|\cdot| \geq 0.05\)
- **Below threshold:** \(|\cdot| < 0.05\)

Four masks are defined:

- \(m_{ab}\): both \(|x_i| \geq \tau\) and \(|y_i| \geq \tau\)
- \(m_{ab'}\): \(|x_i| \geq \tau\) and \(|y_i| < \tau\)
- \(m_{a'b}\): \(|x_i| < \tau\) and \(|y_i| \geq \tau\)
- \(m_{a'b'}\): both \(|x_i| < \tau\) and \(|y_i| < \tau\)

### 2.3 Expectation Terms

For each mask \(m\), the expectation is the **average sign product** over days where the mask is true:

$$\mathbb{E}(a,b) = \frac{\sum_{i \in w} a_i \, b_i \, m_i}{\sum_{i \in w} m_i}$$

with the convention that the sum is zero (and the expectation undefined) when the denominator is zero. Here \(a_i b_i \in \{-1, +1\}\).

### 2.4 Bell-CHSH S₁

The CHSH-type correlator is:

$$S_1 = \mathbb{E}(a,b) + \mathbb{E}(a,b') + \mathbb{E}(a',b) - \mathbb{E}(a',b')$$

Under local hidden-variable theories, \(|S_1| \leq 2\). Values \(|S_1| > 2\) indicate **Bell inequality violations**.

### 2.5 Violation Percentage

For each trading date \(t\), we define:

$$\text{ViolationPct}_t = 100 \times \frac{\#\{(A,B) : |S_1^{(A,B)}(t)| > 2\}}{\text{TotalPairs}_t}$$

where \(\text{TotalPairs}_t\) is the number of ticker pairs with valid \(S_1\) on date \(t\). This yields one time series of violation percentage per day.

### 2.6 Outputs

- **s1_values.csv:** Date, PairA, PairB, S1
- **violation_pct.csv:** Date, ViolationPct, TotalPairs, ViolationCounts

---

## 3. Volatility Measures

### 3.1 Rolling Realized Volatility

$$\sigma_t^{\text{roll}} = \sqrt{252} \times \text{std}(r_{t-19}, \ldots, r_t)$$

with a 20-day window, annualized by \(\sqrt{252}\).

### 3.2 GARCH(1,1)

The conditional variance follows:

$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

Fitted via the `arch` package. The annualized volatility is \(\sqrt{252} \times \sigma_t\).

### 3.3 Regime-Switching Volatility

A two-regime Markov regression with switching variance is fitted to S&P GSCI returns. The **regime-switching volatility** is:

$$\sigma_t^{\text{regime}} = p_t \times \sqrt{252} \times \text{std}(r_{t-19}, \ldots, r_t)$$

where \(p_t\) is the smoothed probability of the high-variance regime. This series is used for event identification.

---

## 4. Event Identification

### 4.1 Extreme and Normal Periods

We classify each trading day as **extreme** or **normal** using the regime-switching volatility:

- **Extreme:** \(\sigma_t^{\text{regime}} > 0.40\) (40% annualized)
- **Normal:** \(\sigma_t^{\text{regime}} \leq 0.40\)

### 4.2 Block Detection

Consecutive days with the same classification form **blocks**. An extreme block is a maximal run of days with \(\sigma_t^{\text{regime}} > 0.40\). A normal block is a maximal run with \(\sigma_t^{\text{regime}} \leq 0.40\).

### 4.3 Event Selection

We retain only extreme blocks that are **immediately preceded** by a normal block (no gap). Among these, we select the **top 3** by:

1. Block length (number of trading days), descending
2. Mean regime volatility within the block, descending

These correspond to:

- **Event 1:** 2008 Financial Crisis
- **Event 2:** COVID-19
- **Event 3:** Ukraine War

Each event is a pair (extreme block, preceding normal block).

---

## 5. Network Construction

### 5.1 Daily Networks

For each trading date, we build an **undirected graph** \(G_t\):

- **Nodes:** All tickers that appear in the S1 dataset (typically 53)
- **Edges:** An edge between tickers A and B exists if and only if \(S_1^{(A,B)}(t) > 2\)

Edge weights are the corresponding \(S_1\) values. Isolated nodes are retained.

### 5.2 Global Network Metrics

| Metric | Symbol | Definition |
|--------|--------|------------|
| **Density** | \(d\) | \(d = \frac{2m}{n(n-1)}\) where \(m\) = edges, \(n\) = nodes |
| **Giant component size** | \(\mathcal{G}\) | \(\mathcal{G} = \frac{\max_i |C_i|}{|N|}\) (fraction of nodes in largest connected component) |
| **Average clustering** | \(C\) | \(C = \frac{1}{n} \sum_{v \in G} c_v\), \(c_v\) = local clustering of node \(v\) |
| **Global clustering** | — | Transitivity: fraction of possible triangles that exist |
| **Efficiency** | \(E\) | \(E(G) = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d(i,j)}\) over largest component |
| **Average shortest path** | \(L\) | \(L(G) = \frac{1}{n(n-1)} \sum_{i \neq j} d(i,j)\) over largest component |
| **Diameter** | — | \(\max_{i,j} d(i,j)\) over largest component |
| **Degree centralization** | \(C_D\) | \(C_D = \frac{\sum_i (k_{\max} - k_i)}{(n-1)(n-2)}\) |
| **Betweenness centralization** | \(C_B\) | \(C_B = \frac{\sum_i (b_{\max} - b_i)}{(n-1)(n-2)}\) |
| **Modularity** | \(Q\) | \(Q = \frac{1}{2m} \sum_{i,j} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)\) (greedy modularity) |
| **Community size entropy** | \(H\) | \(H = -\sum_i p_i \log p_i\), \(p_i\) = fraction of nodes in community \(i\) |
| **Number of communities** | — | Count from greedy modularity |
| **Assortativity** | \(r\) | Pearson correlation of degrees of connected nodes |
| **Scale-free exponent** | \(\alpha\) | \(P(k) \propto k^{-\alpha}\) fitted by method of moments |

### 5.3 Node-Level Metrics

- **Degree** \(k_i = \sum_j A_{ij}\)
- **Clustering coefficient** \(c_i = \frac{2e_i}{k_i(k_i-1)}\) where \(e_i\) = edges among neighbors of \(i\)
- **Betweenness centrality** \(b_i = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}\)
- **Closeness centrality** \(c_i^{\text{close}} = \frac{n-1}{\sum_j d(i,j)}\)
- **Eigenvector centrality** (largest eigenvector of adjacency matrix)

---

## 6. Bootstrap Confidence Intervals

For each event and each metric, we compute **95% bootstrap confidence intervals**:

1. Extract all non-NaN values of the metric over the period (extreme or normal).
2. Draw \(B = 5000\) bootstrap samples (resample with replacement, same size).
3. Compute the mean of each bootstrap sample.
4. Set:
   - \(\text{lo} = \text{percentile}(\text{boot\_means}, 2.5)\)
   - \(\text{med} = \text{percentile}(\text{boot\_means}, 50)\)
   - \(\text{hi} = \text{percentile}(\text{boot\_means}, 97.5)\)

Random seed: 12345.

---

## 7. Permutation Tests

### 7.1 Null Hypothesis

For each metric and each event (or aggregate), the null is: **extreme and normal samples come from the same distribution**.

### 7.2 Test Statistic

$$\Delta_{\text{obs}} = \bar{x} - \bar{y}$$

where \(\bar{x}\) = mean in extreme period, \(\bar{y}\) = mean in normal period.

### 7.3 Permutation Procedure

1. Pool all observations: \(n = n_x + n_y\).
2. For \(k = 1, \ldots, N_{\text{perm}}\) (with \(N_{\text{perm}} = 100{,}000\)):
   - Randomly permute the pooled sample.
   - Split into two groups of sizes \(n_x\) and \(n_y\).
   - Compute \(\Delta^{(k)} = \bar{x}^{(k)} - \bar{y}^{(k)}\).
3. Two-sided p-value:
   $$p = \frac{\#\{k : |\Delta^{(k)}| \geq |\Delta_{\text{obs}}|\} + 1}{N_{\text{perm}} + 1}$$

### 7.4 Effect Size: Cohen's d

$$d = \frac{\bar{x} - \bar{y}}{s_p}$$

where the pooled standard deviation is:

$$s_p = \sqrt{\frac{(n_x - 1) s_x^2 + (n_y - 1) s_y^2}{n_x + n_y - 2}}$$

### 7.5 Multiple Testing

Benjamini–Hochberg FDR correction is applied across metrics:

$$q_i = \min_{j \geq i} \frac{m}{j} p_{(j)}$$

where \(p_{(1)} \leq \cdots \leq p_{(m)}\). Metrics with \(p < 0.05\) or \(q < 0.05\) are flagged as significant.

---

## 8. Granger Causality

We test **Granger causality** between violation percentage and regime-switching volatility in both directions.

### 8.1 Data

- **Violation %:** Daily time series from `violation_pct.csv` (one value per date).
- **Regime volatility:** Daily time series from `regime_vol.csv`.

Both series are merged on date (inner join) and any missing values are dropped.

### 8.2 Test

For each direction (Violation → Volatility and Volatility → Violation) and for lags \(\ell = 1, \ldots, 20\), we run a Granger causality test using the F-test from the restricted vs. unrestricted regression (SSR-based). The `statsmodels` function `grangercausalitytests` is used with `verbose=False`.

### 8.3 Output

Results are saved with Direction, Lag, p-value, and q-value (here q-value equals p-value; no FDR correction across lags/directions in the current implementation).

---

## 9. Software and Parameters Summary

| Parameter | Value |
|-----------|-------|
| Rolling window (S1, volatility) | 20 trading days |
| Fixed threshold (S1) | 0.05 |
| Violation bound | \|S1\| > 2 |
| Edge threshold (network) | S1 > 2 |
| Volatility extreme threshold | 0.40 (40% annualized) |
| Bootstrap iterations | 5000 |
| Permutation iterations | 100,000 |
| Granger max lag | 20 |
| Random seed (bootstrap) | 12345 |
