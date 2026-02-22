"""
Generate Fig 2: network metrics panel (density, giant component, clustering, etc.).

Requires Results/networks from 2.py.
Outputs Figures/network_metrics_panel.png
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update({"font.size": 16})
METRICS_DIR = "Results/networks"
OUTPUT_FIG = "Figures/Fig02_network_metrics.png"
os.makedirs("Figures", exist_ok=True)

metric_files = sorted(glob.glob(os.path.join(METRICS_DIR, "*_metrics.csv")))
if not metric_files:
    raise FileNotFoundError("No *_metrics.csv files found in Results/networks")

dfs = [pd.read_csv(f) for f in metric_files]
metrics_df = pd.concat(dfs, ignore_index=True)
metrics_df["Date"] = pd.to_datetime(metrics_df["Date"])
metrics_df = metrics_df.sort_values("Date").reset_index(drop=True)

crises = {
" 2008 Financial Crisis": pd.to_datetime("2008-09-15"),
" COVID-19": pd.to_datetime("2020-03-01"),
" Ukraine War": pd.to_datetime("2022-02-24")
}

top_metrics = ["Density", "GiantComponentSizePct", "AvgClusteringCoeff"]
other_metrics = ["CommunitySizeEntropy", "NumCommunities", "ScaleFreeAlpha"]

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

if "GiantComponentSizePct" in metrics_df.columns:
    axes[0].plot(metrics_df["Date"], metrics_df["GiantComponentSizePct"]/100, label="Giant Component Size", color='darksalmon', linewidth=2)
    axes[0].plot(metrics_df["Date"], metrics_df["AvgClusteringCoeff"], label="Average Clustering Coefficient", color='silver', linewidth=2)

axes[0].plot(metrics_df["Date"], metrics_df["Density"], label="Density", color='royalblue', linewidth=2)
axes[0].text(0.02, 0.98, "(A)", transform=axes[0].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, frameon=True)
if "ScaleFreeAlpha" in metrics_df.columns:
    axes[1].plot(metrics_df["Date"], metrics_df["ScaleFreeAlpha"], linewidth=2, color='slateblue')
    axes[1].text(0.02, 0.98, "(B)", transform=axes[1].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")

if "CommunitySizeEntropy" in metrics_df.columns:
    axes[2].plot(metrics_df["Date"], metrics_df["CommunitySizeEntropy"], color="darkorange", linewidth=2)
    axes[2].text(0.02, 0.98, "(C)", transform=axes[2].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")

if "NumCommunities" in metrics_df.columns:
    axes[3].plot(metrics_df["Date"], metrics_df["NumCommunities"], color="gainsboro", linewidth=2)
    axes[3].text(0.02, 0.98, "(D)", transform=axes[3].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")

# Stagger y-positions for crisis labels to avoid overlap (2008, COVID, Ukraine)
# Offset labels left of vertical lines by ~45 days
label_offset_days = pd.Timedelta(days=45)
crisis_y_fracs = [0.95, 0.85, 0.75]  # top, middle, lower for axes[:2]
crisis_y_fracs_lower = [0.35, 0.25, 0.15]  # for axes[2:]
for i, (label, date) in enumerate(crises.items()):
    y_frac = crisis_y_fracs[min(i, len(crisis_y_fracs) - 1)]
    y_frac_lo = crisis_y_fracs_lower[min(i, len(crisis_y_fracs_lower) - 1)]
    text_date = date - label_offset_days
    for ax in axes[:2]:
        ax.axvline(date, color="black", linestyle="--", alpha=0.7)
        y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac
        ax.text(text_date, y_pos, label, rotation=0, color="black", verticalalignment="top", fontsize=14, ha="right")
    for ax in axes[2:]:
        ax.axvline(date, color="black", linestyle="--", alpha=0.7)
        y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac_lo
        ax.text(text_date, y_pos, label, rotation=0, color="black", verticalalignment="top", fontsize=14, ha="right")

# X-axis: year only (match Established_Methods); y-axis tick count
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(OUTPUT_FIG, dpi=300)
plt.savefig(OUTPUT_FIG.replace(".png", ".svg"))
plt.close()
print(f"Saved panel figure to {OUTPUT_FIG}")
