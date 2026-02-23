"""
Timing analysis: cross-correlation (supplement), lead-lag by event, and LaTeX table.

Requires Results/violation_pct.csv and Results/volatility_traces/regime_vol.csv.
Outputs Figures/Supplement/FigS1_timing_crosscorr.png (supplement),
         Figures/Fig04_timing_lead_lag.png, Results/timing_lead_lag.csv, Results/timing_lead_lag.tex
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("Figures", exist_ok=True)
os.makedirs("Figures/Supplement", exist_ok=True)
os.makedirs("Results", exist_ok=True)

VIOLATION_FILE = "Results/violation_pct.csv"
VOL_FILE = "Results/volatility_traces/regime_vol.csv"
MAX_LAG_DAYS = 30
N_BOOT = 500  # Bootstrap iterations for cross-correlation CI

# Option A: crisis-based windows (2008, COVID, Ukraine)
CRISIS_WINDOWS = [
    ("2008 Financial Crisis", "2008-09-01", "2009-03-31"),
    ("COVID-19", "2020-02-01", "2020-06-30"),
    ("Ukraine War", "2022-02-01", "2022-06-30"),
]


# Load data
vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"])
violation_df = pd.read_csv(VIOLATION_FILE, parse_dates=["Date"])

df = pd.merge(
    violation_df[["Date", "ViolationPct"]],
    vol_df[["Date", "Volatility"]],
    on="Date",
    how="inner"
).dropna().sort_values("Date").reset_index(drop=True)

violation = df["ViolationPct"].values
vol = df["Volatility"].values
dates = pd.to_datetime(df["Date"])

# Crisis-based event selection (Option A: 2008, COVID, Ukraine)
crisis_events = []
for name, start_str, end_str in CRISIS_WINDOWS:
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    crisis_events.append({"name": name, "start": start, "end": end})

# ---------------------------------------------------------------------------
# Fig04: Cross-correlation (improved: bootstrap CI, Science sizing)
# ---------------------------------------------------------------------------
def crosscorr(x, y, lag):
    """Cross-correlation at lag. Negative lag = violation leads volatility."""
    n = len(x)
    if lag >= 0:
        L = n - lag
        if L < 10:
            return np.nan
        x_trim, y_trim = x[:L], y[lag : lag + L]
    else:
        L = n + lag
        if L < 10:
            return np.nan
        x_trim, y_trim = x[-lag : -lag + L], y[:L]
    if np.std(x_trim) < 1e-10 or np.std(y_trim) < 1e-10:
        return np.nan
    return np.corrcoef(x_trim, y_trim)[0, 1]

lags = np.arange(-MAX_LAG_DAYS, MAX_LAG_DAYS + 1)
cc = np.array([crosscorr(violation, vol, int(l)) for l in lags])

# Bootstrap CI for cross-correlation
rng = np.random.default_rng(42)
cc_boot = np.full((N_BOOT, len(lags)), np.nan)
n = len(violation)
for i in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    v_b, vol_b = violation[idx], vol[idx]
    for j, lag in enumerate(lags):
        cc_boot[i, j] = crosscorr(v_b, vol_b, int(lag))
cc_lo = np.nanpercentile(cc_boot, 2.5, axis=0)
cc_hi = np.nanpercentile(cc_boot, 97.5, axis=0)

# Science single-column width ~8.3 cm = 3.27 in
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(lags, cc, "o-", color="steelblue", linewidth=1.5, markersize=3)
ax.fill_between(lags, cc_lo, cc_hi, color="steelblue", alpha=0.2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
ax.set_xlabel("Lag (days); negative = violation leads")
ax.set_ylabel("Cross-correlation")
ax.grid(True, alpha=0.3)
best_idx = np.nanargmax(np.abs(cc))
best_lag = int(lags[best_idx])
ax.annotate(f"Max |r| at lag = {best_lag}", xy=(best_lag, cc[best_idx]),
            xytext=(8, 8), textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
plt.tight_layout()
plt.savefig("Figures/Supplement/FigS1_timing_crosscorr.png", dpi=300)
plt.savefig("Figures/Supplement/FigS1_timing_crosscorr.svg")
plt.close()
print("Saved Figures/Supplement/FigS1_timing_crosscorr.png (supplement)")

# ---------------------------------------------------------------------------
# Fig04: Lead-lag bar chart (crisis-based) [main text]
# ---------------------------------------------------------------------------
rows = []
for evt in crisis_events:
    name = evt["name"]
    window_start = evt["start"]
    window_end = evt["end"]
    mask = (df["Date"] >= window_start) & (df["Date"] <= window_end)
    sub = df.loc[mask]

    if len(sub) < 2:
        rows.append({"Event": name, "Δ (days)": np.nan, "Leads": "N/A"})
        continue

    vol_peak_date = sub.loc[sub["Volatility"].idxmax(), "Date"]
    viol_peak_date = sub.loc[sub["ViolationPct"].idxmax(), "Date"]
    delta = (viol_peak_date - vol_peak_date).days

    if delta > 0:
        leads = "Violation"
    elif delta < 0:
        leads = "Volatility"
    else:
        leads = "Same day"

    rows.append({"Event": name, "Δ (days)": delta, "Leads": leads})

table_df = pd.DataFrame(rows)

# Lead-lag horizontal bar chart (exclude rows with missing delta)
plot_df = table_df.dropna(subset=["Δ (days)"])
if len(plot_df) > 0:
    fig, ax = plt.subplots(figsize=(3.3, 2))
    events = plot_df["Event"].tolist()
    deltas = plot_df["Δ (days)"].astype(int).tolist()
    colors = ["mediumpurple" if d > 0 else "steelblue" for d in deltas]
    ax.barh(events, deltas, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ (days); positive = violation peaks after volatility")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("Figures/Fig04_timing_lead_lag.png", dpi=300)
    plt.savefig("Figures/Fig04_timing_lead_lag.svg")
    plt.close()
    print("Saved Figures/Fig04_timing_lead_lag.png")

# ---------------------------------------------------------------------------
# Table03: Timing lead-lag (LaTeX + CSV)
# ---------------------------------------------------------------------------
full_rows = []
for evt in crisis_events:
    name = evt["name"]
    window_start = evt["start"]
    window_end = evt["end"]
    mask = (df["Date"] >= window_start) & (df["Date"] <= window_end)
    sub = df.loc[mask]

    if len(sub) < 2:
        full_rows.append({"Event": name, "Violation Peak": "N/A", "Volatility Peak": "N/A", "Δ (days)": "—", "Leads": "N/A"})
        continue

    vol_peak_date = sub.loc[sub["Volatility"].idxmax(), "Date"]
    viol_peak_date = sub.loc[sub["ViolationPct"].idxmax(), "Date"]
    delta = (viol_peak_date - vol_peak_date).days
    leads = "Violation" if delta > 0 else ("Volatility" if delta < 0 else "Same day")

    full_rows.append({
        "Event": name,
        "Violation Peak": pd.Timestamp(viol_peak_date).strftime("%Y-%m-%d"),
        "Volatility Peak": pd.Timestamp(vol_peak_date).strftime("%Y-%m-%d"),
        "Δ (days)": delta,
        "Leads": leads
    })

table_full = pd.DataFrame(full_rows)
table_full.to_csv("Results/timing_lead_lag.csv", index=False)
print("Saved Results/timing_lead_lag.csv")

# LaTeX
tex_df = table_full.copy()
tex_df = tex_df.rename(columns={"Violation Peak": "Violation", "Volatility Peak": "Volatility", "Δ (days)": "Δ"})
tex_str = tex_df.to_latex(index=False, column_format="lcccr")
with open("Results/timing_lead_lag.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Lead-lag between S$_1$ violation and volatility peaks by crisis event.}\n")
    f.write("\\label{tab:timing_leadlag}\n")
    f.write(tex_str)
    f.write("\\end{table}\n")
print("Saved Results/timing_lead_lag.tex")
