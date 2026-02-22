"""
Generate Fig03: comparative network snapshots at crisis and calm periods.

Requires Results/s1_values.csv from 0.py, Results/volatility_traces/regime_vol.csv.
Outputs Figures/Fig03a_network_2008.png, Fig03b_network_covid.png,
         Fig03c_network_ukraine.png, Fig03d_network_calm.png
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

S1_FILE = "Results/s1_values.csv"
VOL_FILE = "Results/volatility_traces/regime_vol.csv"
VOL_THRESHOLD = 0.4
N_EVENTS = 3

os.makedirs("Figures", exist_ok=True)

df = pd.read_csv(S1_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
ALL_NODES = sorted(set(df["PairA"]).union(set(df["PairB"])))


def find_blocks(dates, mask):
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


def create_network(df_day, threshold=2.0):
    edges = df_day[df_day["S1"] > threshold][["PairA", "PairB", "S1"]]
    G = nx.Graph()
    G.add_nodes_from(ALL_NODES)
    for _, row in edges.iterrows():
        G.add_edge(row["PairA"], row["PairB"], weight=row["S1"])
    return G


def plot_network_static(G, title, outpath):
    """Static matplotlib network plot."""
    if G.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No edges (no violations)", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        ax.axis("off")
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5)
        fig, ax = plt.subplots(figsize=(10, 10))
        degrees = dict(G.degree())
        node_sizes = [30 + 20 * degrees.get(n, 0) for n in G.nodes()]
        node_colors = [degrees.get(n, 0) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                               cmap=plt.cm.Blues, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.savefig(outpath.replace(".png", ".svg"))
    plt.close()


# Get event dates (peak volatility in each crisis)
vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
ext_blocks = find_blocks(vol_df["Date"].values, mask_extreme)
norm_blocks = find_blocks(vol_df["Date"].values, ~mask_extreme)

for b in ext_blocks:
    i0, i1 = b["idx_start"], b["idx_end"]
    b["mean_vol"] = vol_df.loc[i0:i1, "Volatility"].mean()
    b["max_vol"] = vol_df.loc[i0:i1, "Volatility"].max()

matched = []
for e in ext_blocks:
    preceding = next((nb for nb in norm_blocks if nb["idx_end"] == e["idx_start"] - 1), None)
    if preceding:
        matched.append({"ext": e, "norm": preceding})
matched = sorted(matched, key=lambda x: (-x["ext"]["length"], -x["ext"]["mean_vol"]))[:N_EVENTS]

# Pick peak date in each event (max volatility day)
crisis_dates = []
for pair in matched:
    ext = pair["ext"]
    sub = vol_df.loc[ext["idx_start"] : ext["idx_end"]]
    peak_idx = sub["Volatility"].idxmax()
    crisis_dates.append(vol_df.loc[peak_idx, "Date"])

# Calm period: pick a date in a long normal block
norm_blocks_sorted = sorted(norm_blocks, key=lambda b: b["idx_end"] - b["idx_start"] + 1, reverse=True)
calm_block = None
for nb in norm_blocks_sorted:
    if nb["idx_end"] - nb["idx_start"] >= 60:  # at least 60 trading days
        calm_block = nb
        break
if calm_block:
    mid = calm_block["idx_start"] + (calm_block["idx_end"] - calm_block["idx_start"]) // 2
    calm_date = vol_df.loc[mid, "Date"]
else:
    calm_date = pd.Timestamp("2017-06-15")  # fallback

# Ensure dates exist in s1 data
available_dates = sorted(df["Date"].unique())

def nearest_date(target):
    target = pd.Timestamp(target)
    idx = np.argmin(np.abs(pd.to_datetime(available_dates) - target))
    return available_dates[idx]

crisis_dates = [nearest_date(d) for d in crisis_dates]
calm_date = nearest_date(calm_date)

labels = ["2008 Financial Crisis", "COVID-19", "Ukraine War"]
outputs = [
    ("Figures/Fig03a_network_2008.png", labels[0]),
    ("Figures/Fig03b_network_covid.png", labels[1]),
    ("Figures/Fig03c_network_ukraine.png", labels[2]),
    ("Figures/Fig03d_network_calm.png", "Calm period"),
]
dates_to_plot = crisis_dates + [calm_date]

for (outpath, label), d in zip(outputs, dates_to_plot):
    df_day = df[df["Date"] == d]
    G = create_network(df_day)
    date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    plot_network_static(G, f"{label} ({date_str})", outpath)
    print(f"Saved {outpath}")
