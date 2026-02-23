"""
Generate Fig 2: network metrics panel (structural indicators of market fragility).

Option 2 layout: Density, Scale-Free Alpha, Avg Clustering, Community Size Entropy.
Requires Results/networks from 2.py.
Outputs Figures/Fig02_network_metrics.png
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Monochromatic blue shades (one metric per panel)
BLUE_A = "#1E3A8A"   # dark blue
BLUE_B = "#4169E1"   # royal blue
BLUE_C = "#5B9BD5"   # medium blue
BLUE_D = "#87CEEB"   # sky blue

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Panel A: Density (strongest connectivity spike)
axes[0].plot(metrics_df["Date"], metrics_df["Density"], linewidth=2, color=BLUE_A)
axes[0].text(0.02, 0.98, "(A)", transform=axes[0].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
axes[0].set_ylabel("Density")

# Panel B: Scale-Free Alpha (degree distribution spike)
axes[1].plot(metrics_df["Date"], metrics_df["ScaleFreeAlpha"], linewidth=2, color=BLUE_B)
axes[1].text(0.02, 0.98, "(B)", transform=axes[1].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
axes[1].set_ylabel("Scale-Free\nAlpha")

# Panel C: Avg Clustering Coefficient (local clustering spike)
axes[2].plot(metrics_df["Date"], metrics_df["AvgClusteringCoeff"], linewidth=2, color=BLUE_C)
axes[2].text(0.02, 0.98, "(C)", transform=axes[2].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
axes[2].set_ylabel("Average Clustering\nCoefficient")

# Panel D: Community Size Entropy (drops in crisis)
axes[3].plot(metrics_df["Date"], metrics_df["CommunitySizeEntropy"], linewidth=2, color=BLUE_D)
axes[3].text(0.02, 0.98, "(D)", transform=axes[3].transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
axes[3].set_ylabel("Community Size\nEntropy")

# Crisis labels: 2008/COVID left of line; Ukraine right of line
# Panel C: COVID and Ukraine near top; Panel D: lower positions
label_offset_days = pd.Timedelta(days=45)
crisis_y_fracs = [0.95, 0.85, 0.75]  # axes 0,1: 2008, COVID, Ukraine
crisis_y_fracs_c = [0.75, 0.95, 0.85]  # Panel C: 2008 lower, COVID/Ukraine near top
crisis_y_fracs_d = [0.35, 0.25, 0.15]  # Panel D: lower positions
crisis_list = list(crises.items())
for i, (label, date) in enumerate(crisis_list):
    is_ukraine = "Ukraine" in label
    text_date = date + label_offset_days if is_ukraine else date - label_offset_days
    ha = "left" if is_ukraine else "right"
    for ax in axes:
        ax.axvline(date, color="black", linestyle="--", alpha=0.7)
    for j, ax in enumerate(axes):
        if j <= 1:
            y_frac = crisis_y_fracs[i]
        elif j == 2:
            y_frac = crisis_y_fracs_c[i]
        else:
            y_frac = crisis_y_fracs_d[i]
        y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac
        ax.text(text_date, y_pos, label, rotation=0, color="black", verticalalignment="top", fontsize=14, ha=ha)

# X-axis: year only (match Established_Methods); y-axis tick count
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_FIG, dpi=300)
plt.savefig(OUTPUT_FIG.replace(".png", ".svg"))
plt.close()
print(f"Saved panel figure to {OUTPUT_FIG}")
