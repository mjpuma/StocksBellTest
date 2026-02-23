"""
Generate Fig03: network snapshots at crisis periods (giant component only).

Requires Results/s1_values.csv from 0.py, Results/volatility_traces/regime_vol.csv.
Outputs Figures/Fig03_networks.png (1×3 multipanel, full-width).
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
S1_FILE = "Results/s1_values.csv"
VOL_FILE = "Results/volatility_traces/regime_vol.csv"
VOL_THRESHOLD = 0.4
N_EVENTS = 3
LABEL_DEGREE_MIN = 2  # Only label nodes with degree >= this
NODE_SIZE_BASE = 80
NODE_SIZE_SCALE = 25

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


def extract_giant_component(G):
    """Return subgraph of largest connected component, or None if no edges."""
    if G.number_of_edges() == 0:
        return None
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


# Get event dates (peak volatility in each crisis)
vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
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

# Ensure dates exist in s1 data
available_dates = sorted(df["Date"].unique())

def nearest_date(target):
    target = pd.Timestamp(target)
    idx = np.argmin(np.abs(pd.to_datetime(available_dates) - target))
    return available_dates[idx]

crisis_dates = [nearest_date(d) for d in crisis_dates]
labels = ["2008 Financial Crisis", "COVID-19", "Ukraine War"]

# Build networks and extract giant components
graphs = []
for d in crisis_dates:
    df_day = df[df["Date"] == d]
    G = create_network(df_day)
    G_gc = extract_giant_component(G)
    graphs.append(G_gc)

# Shared color/size scales: global min/max degree across all giant components
degree_values = []
for G_gc in graphs:
    if G_gc is not None:
        degree_values.extend(dict(G_gc.degree()).values())
vmin = 0
vmax = max(degree_values) if degree_values else 1

# Full-width 1×3 multipanel figure (reduced space between panels)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"wspace": 0.12})
for ax, G_gc, label, d in zip(axes, graphs, labels, crisis_dates):
    date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    ax.set_title(f"{label}\n({date_str})", fontsize=14)
    ax.axis("off")
    if G_gc is None or G_gc.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center", fontsize=14)
        continue
    pos = nx.spring_layout(G_gc, seed=42, k=1.5)
    degrees = dict(G_gc.degree())
    node_sizes = [NODE_SIZE_BASE + NODE_SIZE_SCALE * degrees[n] for n in G_gc.nodes()]
    node_colors = [degrees[n] for n in G_gc.nodes()]
    nx.draw_networkx_nodes(G_gc, pos, node_size=node_sizes, node_color=node_colors,
                          cmap=plt.cm.Blues, alpha=0.8, ax=ax, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G_gc, pos, alpha=0.4, width=0.8, ax=ax)
    labels_to_show = {n: n for n in G_gc.nodes() if degrees[n] >= LABEL_DEGREE_MIN}
    # White text for dark blue nodes (high degree), black for light blue
    norm = (vmax - vmin) or 1
    white_nodes = {n: n for n in labels_to_show if (degrees[n] - vmin) / norm > 0.5}
    black_nodes = {n: n for n in labels_to_show if (degrees[n] - vmin) / norm <= 0.5}
    if white_nodes:
        nx.draw_networkx_labels(G_gc, pos, labels=white_nodes, font_size=7, ax=ax, font_color="white")
    if black_nodes:
        nx.draw_networkx_labels(G_gc, pos, labels=black_nodes, font_size=7, ax=ax, font_color="black")

# Horizontal colorbar at bottom (use subplots_adjust to avoid tight_layout warning)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.8, aspect=40,
                    pad=0.08, label="Degree")

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.18, top=0.92, wspace=0.12)
outpath = "Figures/Fig03_networks.png"
plt.savefig(outpath, dpi=300)
plt.savefig(outpath.replace(".png", ".svg"))
plt.close()
print(f"Saved {outpath}")
