# Bell Inequality Violations in Agriculture Equity Networks

Analysis pipeline for detecting Bell inequality violations in agriculture-related equity returns and relating them to commodity volatility and network structure.

This repository is an updated and extended version of [56sarager/Final-Paper-Draft-](https://github.com/56sarager/Final-Paper-Draft-).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (0 → 1 → 2 → 2Bootstrap → 2Fig2 → 2Fig3 → 3 → Granger → timing)
python run_all.py

# Verify computations
python verify_computations.py
```

Use `MPLBACKEND=Agg` if running headless (e.g., on a server) to avoid blocking on interactive plots.

## Pipeline Overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `0.py` | `yfinance_tickers.csv` | `Results/s1_values.csv`, `Results/violation_pct.csv` |
| 2 | `1.py` | `Results/violation_pct.csv` | `Results/volatility_traces/*.csv`, `Fig01_volatility_and_violation.*` |
| 3 | `2.py` | `Results/s1_values.csv` | `Results/networks/*.csv`, `Figures/networks/*.html` |
| 4 | `2Bootstrap.py` | `Results/volatility_traces/`, `Results/networks/` | `Results/event_tables/*.csv` |
| 5 | `2Fig2.py` | `Results/networks/` | `Fig02_network_metrics.*` |
| 6 | `2Fig3_networks.py` | `Results/s1_values.csv`, `Results/volatility_traces/` | `Fig03a–d_network_*.png` |
| 7 | `3.py` | `Results/volatility_traces/`, `Results/networks/` | `Table01_permtest_*.*`, `Results/event_tables/*.csv` |
| 8 | `Granger_causality.py` | `Results/violation_pct.csv`, `Results/volatility_traces/regime_vol.csv` | `Results/granger_results.csv`, `Fig05_granger_causality.*` |
| 9 | `timing_analysis.py` | `Results/violation_pct.csv`, `Results/volatility_traces/regime_vol.csv` | `Figures/Supplement/FigS1_*`, `Fig04_timing_lead_lag.*`, `Table03_timing_lead_lag.*` |

## Data

- **Tickers**: `yfinance_tickers.csv` — agriculture-related Yahoo Finance tickers
- **S1 values**: Bell-CHSH correlator \(S_1 = E(a,b) + E(a,b') + E(a',b) - E(a',b')\)
- **Violation %**: Fraction of ticker pairs with \(|S_1| > 2\) per day
- **Volatility**: S&P GSCI (rolling, GARCH, regime-switching)

## Figures and Tables

### Figures

**Fig01. Volatility methods and S₁ violation percentage.** (A) Annualized volatility of the S&P GSCI: rolling realized (20-day), GARCH(1,1), and regime-switching. Dotted line indicates the 40% threshold used for extreme-period classification. (B) Daily fraction of ticker pairs with |S₁| > 2 (Bell inequality violations).

**Fig02. Network topology metrics as structural indicators of market fragility.** Panels show density, scale-free alpha, clustering coefficient, and community entropy over time, with vertical lines marking the 2008, COVID, and Ukraine crises.

**Fig03. Network structure during crisis periods.** Giant components of S₁-based correlation networks at peak volatility during the 2008, COVID-19, and Ukraine crises. Node size and color indicate degree.

**Fig04. Lead–lag by crisis.** Bar chart of Δ (days) between violation and volatility peaks for each crisis (positive = violation peaks after volatility).

**Fig05. Granger causality by crisis.** S₁↔volatility tests at lags 1–10 for 2008, COVID-19, and Ukraine. Crisis-specific patterns differ from aggregate.

**FigS1 (Supplement). Timing cross-correlation.** Aggregate cross-correlation of violation % vs regime volatility at lags −30 to +30 days.

### Tables

**Table01. Permutation test results.** Network metrics compared between extreme and normal periods for each event (2008, COVID-19, Ukraine War) and in aggregate. Reports observed means, percent change, Cohen's d, and p- and q-values (Benjamini–Hochberg FDR).

**Table02. Granger causality.** Crisis-specific tests for S₁↔volatility (both directions, lags 1–10).

**Table03. Timing lead–lag.** For each event, dates of violation and volatility peaks and which series leads.

## Documentation

- [docs/ANALYSIS.md](docs/ANALYSIS.md) — Mathematical definitions and methodology

## Requirements

See `requirements.txt`. Key: numpy, pandas, yfinance, arch, statsmodels, networkx, plotly, matplotlib.
