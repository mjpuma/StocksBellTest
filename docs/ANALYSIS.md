# Analysis Methodology

## 0.py — S1 and Violation Percentage

Takes agriculture-related Yahoo Finance tickers and outputs:
- **s1_values.csv**: Date, PairA, PairB, S1
- **violation_pct.csv**: Date, ViolationPct, TotalPairs, ViolationCounts

### S1 Definition

$$S_1 = \mathbb{E}(a,b) + \mathbb{E}(a,b') + \mathbb{E}(a',b) - \mathbb{E}(a',b')$$

Expectations use the average sign product:
$$\mathbb{E}(a,b) = \frac{\sum_{i \in w} a_i b_i m_i}{\sum_{i \in w} m_i}$$

where \(m_i\) are masks. Masks are defined by a fixed threshold of 0.05:
- \(\mathbb{E}(a,b)\): both returns \(\geq\) 0.05
- \(\mathbb{E}(a',b')\): both returns \(<\) 0.05
- etc.

Violation % = 100 × (pairs with \(|S_1| > 2\)) / total pairs per day.

---

## 1.py — Volatility Methods

Computes rolling volatility, GARCH(1,1), and regime-switching volatility for S&P GSCI:

$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

Outputs: `Results/volatility_traces/*.csv`, `Figures/Established_Methods_&_S1.*`

---

## 2.py — Network Construction

Builds undirected networks per day. Edge if \(S_1 > 2\). All 53–54 tickers as nodes.

### Global Metrics

| Metric | Formula |
|--------|---------|
| Density | \(d = 2m / (n(n-1))\) |
| Giant component | \(\mathcal{G} = \max_i |C_i| / |N|\) |
| Avg clustering | \(C = \frac{1}{n} \sum_v c_v\) |
| Modularity | \(Q = \frac{1}{2m} \sum_{i,j} [A_{ij} - \frac{k_i k_j}{2m}] \delta(c_i, c_j)\) |
| Community entropy | \(H = -\sum_i p_i \log p_i\) |
| Assortativity | Pearson correlation of degrees of connected nodes |
| Scale-free α | \(P(k) \propto k^{-\alpha}\) (method of moments) |

### Node Metrics

Degree, clustering coefficient, betweenness, closeness, eigenvector centrality.

---

## 2Bootstrap.py — Bootstrap Confidence Intervals

Identifies extreme periods (regime vol > 40%, ≥40 days): 2008 Crisis, COVID-19, Ukraine War. Compares to preceding normal periods. Bootstrap (5000 iters) for 95% CI.

---

## 3.py — Permutation Tests

Permutation test (100,000 iterations) for difference in means between extreme vs normal periods. Cohen's d for effect size. Benjamini–Hochberg FDR for q-values.

---

## Granger_causality.py

Tests Granger causality between violation % and regime volatility (both directions, lags 1–20). Uses `violation_pct.csv` (one value per date).
