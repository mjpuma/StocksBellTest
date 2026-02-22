"""
Generate Fig 2: network metrics panel (density, giant component, clustering, etc.).

Requires Results/networks from 2.py.
Outputs Figures/network_metrics_panel.png
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

METRICS_DIR = "Results/networks"
OUTPUT_FIG = "Figures/network_metrics_panel.png"
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
    axes[0].plot(metrics_df["Date"], metrics_df["AvgClusteringCoeff"], label="Avgerage Clustering Coefficient", color='silver', linewidth=2)
    axes[0].set_title("Density, Giant Component, Clustering")

axes[0].plot(metrics_df["Date"], metrics_df["Density"], label="Density", color='royalblue', linewidth=2)
axes[0].legend()
if "ScaleFreeAlpha" in metrics_df.columns:
    axes[1].plot(metrics_df["Date"], metrics_df["ScaleFreeAlpha"], linewidth=2, color='slateblue')
    axes[1].set_title("Scale Free Alpha")

if "CommunitySizeEntropy" in metrics_df.columns:
    axes[2].plot(metrics_df["Date"], metrics_df["CommunitySizeEntropy"], color="darkorange", linewidth=2)
    axes[2].set_title("Community Size Entropy")

if "NumCommunities" in metrics_df.columns:
    axes[3].plot(metrics_df["Date"], metrics_df["NumCommunities"], color="gainsboro", linewidth=2)
    axes[3].set_title("Number of Communities")
    axes[3].set_xlabel("Date")

for label, date in crises.items():
    for ax in axes[:2]:
        ax.axvline(date, color="black", linestyle="--", alpha=0.7)
        ax.text(date, ax.get_ylim()[1]*0.95, label, rotation=0, color="black", verticalalignment="top", fontsize=9)
    for ax in axes[2:]:
        ax.axvline(date, color="black", linestyle="--", alpha=0.7)
        ax.text(date, ax.get_ylim()[1]*0.3, label, rotation=0, color="black", verticalalignment="top", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()
print(f"Saved panel figure to {OUTPUT_FIG}")
