# Bell Inequality Violations in Agriculture Equity Networks

Analysis pipeline for detecting Bell inequality violations in agriculture-related equity returns and relating them to commodity volatility and network structure.

This repository is an updated and extended version of [56sarager/Final-Paper-Draft-](https://github.com/56sarager/Final-Paper-Draft-).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (0 → 1 → 2 → 2Bootstrap → 2Fig2 → 3 → Granger)
python run_all.py

# Verify computations
python verify_computations.py
```

Use `MPLBACKEND=Agg` if running headless (e.g., on a server) to avoid blocking on interactive plots.

## Pipeline Overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `0.py` | `yfinance_tickers.csv` | `Results/s1_values.csv`, `Results/violation_pct.csv` |
| 2 | `1.py` | `Results/violation_pct.csv` | `Results/volatility_traces/*.csv`, `Figures/Established_Methods_&_S1.*` |
| 3 | `2.py` | `Results/s1_values.csv` | `Results/networks/*.csv`, `Figures/networks/*.html` |
| 4 | `2Bootstrap.py` | `Results/volatility_traces/`, `Results/networks/` | `Results/event_tables/*.csv` |
| 5 | `2Fig2.py` | `Results/networks/` | `Figures/network_metrics_panel.png` |
| 6 | `3.py` | `Results/volatility_traces/`, `Results/networks/` | `Figures/event_tables/*.png`, `Results/event_tables/*.csv` |
| 7 | `Granger_causality.py` | `Results/violation_pct.csv`, `Results/volatility_traces/regime_vol.csv` | `Results/granger_results.csv`, `Figures/granger_results_table.png` |

## Data

- **Tickers**: `yfinance_tickers.csv` — agriculture-related Yahoo Finance tickers
- **S1 values**: Bell-CHSH correlator \(S_1 = E(a,b) + E(a,b') + E(a',b) - E(a',b')\)
- **Violation %**: Fraction of ticker pairs with \(|S_1| > 2\) per day
- **Volatility**: S&P GSCI (rolling, GARCH, regime-switching)

## Documentation

- [docs/ANALYSIS.md](docs/ANALYSIS.md) — Mathematical definitions and methodology

## Requirements

See `requirements.txt`. Key: numpy, pandas, yfinance, arch, statsmodels, networkx, plotly, matplotlib.
