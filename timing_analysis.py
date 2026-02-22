"""
Timing analysis: cross-correlation, event alignment, and lead-lag table.

Requires Results/violation_pct.csv, Results/volatility_traces/regime_vol.csv.
Outputs Figures/Fig04_timing_crosscorr, Fig05_timing_event_alignment, Table03_timing_lead_lag.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.makedirs("Figures", exist_ok=True)
os.makedirs("Results", exist_ok=True)

VIOLATION_FILE = "Results/violation_pct.csv"
VOL_FILE = "Results/volatility_traces/regime_vol.csv"
VOL_THRESHOLD = 0.4
MAX_LAG_DAYS = 30
EVENT_WINDOW_DAYS = 60
N_EVENTS = 3


def find_blocks(dates, mask):
    """Find contiguous blocks where mask is True."""
    blocks = []
    in_block = False
    n = len(mask)
    for i in range(n):
        if mask[i] and not in_block:
            in_block = True
            i0 = i
        if not mask[i] and in_block:
            i1 = i - 1
            blocks.append({"idx_start": i0, "idx_end": i1})
            in_block = False
    if in_block:
        blocks.append({"idx_start": i0, "idx_end": n - 1})
    for b in blocks:
        b["start"] = dates[b["idx_start"]]
        b["end"] = dates[b["idx_end"]]
        b["length"] = b["idx_end"] - b["idx_start"] + 1
    return blocks


# Load data
violation_df = pd.read_csv(VIOLATION_FILE, parse_dates=["Date"])
vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"])

df = pd.merge(
    violation_df[["Date", "ViolationPct"]],
    vol_df[["Date", "Volatility"]],
    on="Date",
    how="inner"
).dropna().sort_values("Date").reset_index(drop=True)

violation = df["ViolationPct"].values
vol = df["Volatility"].values
dates = pd.to_datetime(df["Date"])

# Event blocks (same logic as 2Bootstrap)
mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
vol_dates = pd.to_datetime(vol_df["Date"])
ext_blocks = find_blocks(vol_dates.values, mask_extreme)
norm_blocks = find_blocks(vol_dates.values, ~mask_extreme)
for b in ext_blocks:
    i0, i1 = b["idx_start"], b["idx_end"]
    b["max_vol"] = vol_df.loc[i0:i1, "Volatility"].max()
    b["mean_vol"] = vol_df.loc[i0:i1, "Volatility"].mean()

matched_pairs = []
for e in ext_blocks:
    e_start = e["idx_start"]
    preceding = None
    for nb in norm_blocks:
        if nb["idx_end"] == e_start - 1:
            preceding = nb
            break
    if preceding is not None:
        matched_pairs.append({"ext": e, "norm": preceding})

matched_pairs = sorted(matched_pairs, key=lambda x: (-x["ext"]["length"], -x["ext"]["mean_vol"]))[:N_EVENTS]

event_names = ["2008 Financial Crisis", "COVID-19", "Ukraine War"]

# ---------------------------------------------------------------------------
# Fig04: Cross-correlation
# ---------------------------------------------------------------------------
def crosscorr(x, y, lag):
    """Cross-correlation at lag. Negative lag = violation leads volatility."""
    n = len(x)
    if lag >= 0:
        # y[t+lag] vs x[t]: trim x from end, y from start
        L = n - lag
        if L < 10:
            return np.nan
        x_trim, y_trim = x[:L], y[lag : lag + L]
    else:
        # x[t-lag] vs y[t]: trim x from start, y from end
        L = n + lag  # lag is negative
        if L < 10:
            return np.nan
        x_trim, y_trim = x[-lag : -lag + L], y[:L]
    if np.std(x_trim) < 1e-10 or np.std(y_trim) < 1e-10:
        return np.nan
    return np.corrcoef(x_trim, y_trim)[0, 1]

lags = np.arange(-MAX_LAG_DAYS, MAX_LAG_DAYS + 1)
cc = [crosscorr(violation, vol, int(l)) for l in lags]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lags, cc, "o-", color="steelblue", linewidth=2, markersize=4)
ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
ax.set_xlabel("Lag (days); negative = violation leads volatility")
ax.set_ylabel("Cross-correlation")
ax.set_title("Violation % vs Regime Volatility")
ax.grid(True, alpha=0.3)
best_lag = lags[np.nanargmax(np.abs(cc))]
ax.annotate(f"Max |r| at lag = {int(best_lag)}", xy=(best_lag, cc[int(best_lag + MAX_LAG_DAYS)]),
            xytext=(10, 10), textcoords="offset points", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
plt.tight_layout()
plt.savefig("Figures/Fig04_timing_crosscorr.png", dpi=300)
plt.savefig("Figures/Fig04_timing_crosscorr.svg")
plt.close()
print("Saved Figures/Fig04_timing_crosscorr.png")

# ---------------------------------------------------------------------------
# Fig05: Event alignment (dual-axis per event)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
plt.rcParams.update({"font.size": 16})

for idx, (pair, name) in enumerate(zip(matched_pairs[:N_EVENTS], event_names)):
    ext = pair["ext"]
    center = pd.Timestamp(ext["start"]) + (pd.Timestamp(ext["end"]) - pd.Timestamp(ext["start"])) / 2
    window_start = center - pd.Timedelta(days=EVENT_WINDOW_DAYS)
    window_end = center + pd.Timedelta(days=EVENT_WINDOW_DAYS)

    mask = (df["Date"] >= window_start) & (df["Date"] <= window_end)
    sub = df.loc[mask].copy()

    if len(sub) < 5:
        continue

    ax1 = axes[idx]
    ax2 = ax1.twinx()

    ax1.plot(sub["Date"], sub["ViolationPct"], color="mediumvioletred", linewidth=2, label="Violation %")
    ax2.plot(sub["Date"], sub["Volatility"], color="darkturquoise", linewidth=2, label="Regime Volatility")

    vol_peak_idx = sub["Volatility"].idxmax()
    vol_peak_date = sub.loc[vol_peak_idx, "Date"]
    viol_peak_idx = sub["ViolationPct"].idxmax()
    viol_peak_date = sub.loc[viol_peak_idx, "Date"]

    ax1.axvline(vol_peak_date, color="darkturquoise", linestyle="--", alpha=0.7)
    ax1.axvline(viol_peak_date, color="mediumvioletred", linestyle=":", alpha=0.7)

    ax1.set_ylabel("Violation %", color="mediumvioletred")
    ax2.set_ylabel("Volatility", color="darkturquoise")
    ax1.set_title(name)
    ax1.tick_params(axis="y", labelcolor="mediumvioletred")
    ax2.tick_params(axis="y", labelcolor="darkturquoise")
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("Figures/Fig05_timing_event_alignment.png", dpi=300)
plt.savefig("Figures/Fig05_timing_event_alignment.svg")
plt.close()
print("Saved Figures/Fig05_timing_event_alignment.png")

# ---------------------------------------------------------------------------
# Table03: Lead-lag
# ---------------------------------------------------------------------------
rows = []
for pair, name in zip(matched_pairs[:N_EVENTS], event_names):
    ext = pair["ext"]
    center = pd.Timestamp(ext["start"]) + (pd.Timestamp(ext["end"]) - pd.Timestamp(ext["start"])) / 2
    window_start = center - pd.Timedelta(days=EVENT_WINDOW_DAYS)
    window_end = center + pd.Timedelta(days=EVENT_WINDOW_DAYS)

    mask = (df["Date"] >= window_start) & (df["Date"] <= window_end)
    sub = df.loc[mask]

    if len(sub) < 2:
        rows.append({"Event": name, "Violation Peak": "N/A", "Volatility Peak": "N/A", "Δ (days)": np.nan, "Leads": "N/A"})
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

    rows.append({
        "Event": name,
        "Violation Peak": pd.Timestamp(viol_peak_date).strftime("%Y-%m-%d"),
        "Volatility Peak": pd.Timestamp(vol_peak_date).strftime("%Y-%m-%d"),
        "Δ (days)": delta,
        "Leads": leads
    })

table_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 1.2 * len(table_df) + 1))
ax.axis("off")
cols = list(table_df.columns)
cell_text = [[str(row[c]) for c in cols] for _, row in table_df.iterrows()]
tbl = ax.table(
    cellText=cell_text,
    colLabels=cols,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(16)
tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#e8e8e8")
    cell.set_edgecolor("black")
plt.tight_layout()
plt.savefig("Figures/Table03_timing_lead_lag.png", dpi=300, bbox_inches="tight")
plt.savefig("Figures/Table03_timing_lead_lag.svg", bbox_inches="tight")
plt.close()
print("Saved Figures/Table03_timing_lead_lag.png")

# Also save CSV
table_df.to_csv("Results/timing_lead_lag.csv", index=False)
print("Saved Results/timing_lead_lag.csv")
