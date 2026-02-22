# Figures Plan: Network Plots, Timing, and Naming

## 1. Systematic Naming (Implemented)

**Figures:** `Fig{N}_{short_description}.{png|svg}` — visual plots only  
**Tables:** `Table{N}_{short_description}.{png|svg}` — tabular data (not figures)

| Output | Description |
|--------|-------------|
| `Fig01_volatility_and_violation.*` | Volatility methods + S₁ violation % |
| `Fig02_network_metrics.*` | 4-panel network metrics over time |
| `Fig03a_network_2008.*` | Network snapshot at 2008 crisis peak |
| `Fig03b_network_covid.*` | Network snapshot at COVID peak |
| `Fig03c_network_ukraine.*` | Network snapshot at Ukraine peak |
| `Fig03d_network_calm.*` | Network snapshot during calm period |
| `Fig04_timing_crosscorr.*` | Cross-correlation of violation % vs volatility |
| `Fig05_timing_event_alignment.*` | Dual-axis event alignment plots |
| `Table01_permtest_event{N}.*` | Permutation test per event |
| `Table01_permtest_aggregate_*.png` | Aggregate permutation (p- and q-values) |
| `Table02_granger_causality.*` | Granger causality results |
| `Table03_timing_lead_lag.*` | Lead–lag table per event |

**Folder structure:**
```
Figures/
├── Fig01_volatility_and_violation.png
├── Fig02_network_metrics.png
├── Fig03a_network_2008.png
├── Fig03b_network_covid.png
├── Fig03c_network_ukraine.png
├── Fig03d_network_calm.png
├── Fig04_timing_crosscorr.png
├── Fig05_timing_event_alignment.png
├── Table01_permtest_event1.png
├── Table01_permtest_event2.png
├── Table01_permtest_event3.png
├── Table01_permtest_aggregate_pvals.png
├── Table01_permtest_aggregate_qvals.png
├── Table02_granger_causality.png
├── Table03_timing_lead_lag.png
└── networks/                 # daily HTML for exploration
    └── YYYY-MM-DD.html
```

---

## 2. Network Plots: Dyadic vs Alter-to-Alter

### Data structure
- **Dyadic edges:** Pairs (A, B) with S₁ > 2 — the direct Bell-violating relationships we measure.
- **Alter-to-alter:** For a focal node A, its alters are nodes B such that (A,B) violates. Alter-to-alter = pairs (B,C) where both B and C are alters of A — do they also violate with each other?

### Options for visualization

| Option | What to show | Pros | Cons |
|--------|--------------|------|------|
| **A. Dyads only** | Edges where S₁ > 2 (current) | Clean, directly interpretable | May miss clustering structure |
| **B. All pairs (weighted)** | Full graph, edge weight = \|S₁\| | Shows full dependence structure | 1378 edges → very dense, unreadable |
| **C. Dyads + alter-to-alter** | Violating dyads + highlight triangles (B,C) when (A,B) and (A,C) both violate | Reveals whether violations cluster (e.g., “if A–B and A–C violate, does B–C?”) | More complex; need clear visual encoding |
| **D. Dyads + alter-to-alter (focal)** | For each crisis, show ego network of top hub: its violating pairs + alter-to-alter among its neighbors | Focused, interpretable | Requires choosing focal node per event |

### Recommendation
- **Main paper:** **Option A (dyads only)** for comparative snapshots — clearest for readers.
- **Supplementary:** **Option C or D** — one figure showing alter-to-alter triangles (e.g., fraction of open triangles that close) as a metric, or ego networks for top-degree nodes during each crisis.

### Comparative snapshots (new)
- **Fig03a:** Network at peak of 2008 crisis (e.g., 2008-10-15)
- **Fig03b:** Network at COVID peak (e.g., 2020-03-23)
- **Fig03c:** Network at Ukraine peak (e.g., 2022-03-07)
- **Fig03d:** Network during a calm period (e.g., 2017-06-15)

Static PNG/SVG for the paper; optionally keep interactive HTML for exploration.

---

## 3. Timing Analyses (Critical)

### Questions to answer
1. **Does S₁ violation lead or lag volatility?** (Granger already tests this; we can add visual lead–lag plots.)
2. **Does violation % peak before, during, or after volatility peaks?** (Cross-correlation, peak alignment.)
3. **Event-by-event:** For each crisis, when does violation % spike relative to volatility spike?

### Proposed timing figures

| Figure | Content |
|--------|---------|
| **Fig06a_timing_crosscorr** | Cross-correlation of violation % vs regime volatility (lags -30 to +30 days). Shows if S₁ leads (negative lag) or lags (positive lag). |
| **Fig06b_timing_event_alignment** | For each event: dual-axis plot of violation % and volatility around the event window (e.g., ±60 days). Vertical line at volatility peak. |
| **Fig06c_timing_lead_lag_table** | Table: for each event, days from violation peak to volatility peak (or vice versa); which leads? |

### Metrics to compute
- **Cross-correlation:** `violation_pct` vs `regime_vol` at lags -30,…,+30.
- **Peak alignment:** For each event window, find argmax of violation % and argmax of volatility; report Δ (days).
- **Granger extension:** Already have Fig05; could add a small summary panel showing “S₁ leads at lag X” or “Volatility leads at lag Y” for each event window.

---

## 4. Implementation Status

- [x] **Rename figures and tables** — Fig01–05, Table01–03
- [x] **Timing analyses** — `timing_analysis.py` → Fig04, Fig05, Table03
- [x] **Comparative network snapshots** — `2Fig3_networks.py` → Fig03a–d (dyads only)
- [ ] **Alter-to-alter** (optional) — metric or supplementary ego-network figure

---

## 5. Summary

| Item | Status |
|------|--------|
| **Naming** | Figures vs Tables distinguished; `Fig{N}_`, `Table{N}_` |
| **Dyads** | Dyads only for main network snapshots |
| **Alter-to-alter** | Optional supplementary |
| **Timing** | `timing_analysis.py`: cross-correlation, event alignment, lead–lag table |
| **Network snapshots** | 4 comparative PNGs: 2008, COVID, Ukraine, calm |
