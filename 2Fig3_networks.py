"""
Generate Fig03: network snapshots at crisis periods (giant component only).

Requires Results/s1_values.csv from 0.py, Results/volatility_traces/regime_vol.csv,
yfinance_tickers.csv for group/category coloring.
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
TICKER_FILE = "yfinance_tickers.csv"
VOL_THRESHOLD = 0.4
N_EVENTS = 3
LABEL_DEGREE_MIN = 2  # Only label nodes with degree >= this
NODE_SIZE_BASE = 80
NODE_SIZE_SCALE = 25

os.makedirs("Figures", exist_ok=True)

# Load ticker metadata for group/category coloring
ticker_to_group = {}
ticker_to_category = {}
if os.path.exists(TICKER_FILE):
    tickers_df = pd.read_csv(TICKER_FILE)
    for _, row in tickers_df.iterrows():
        t = str(row.get("ticker", "")).strip()
        if t:
            ticker_to_group[t] = str(row.get("group", "Other")).strip()
            ticker_to_category[t] = str(row.get("category", "Other")).strip()

# Discrete colors per group (tab10 + tab20 for many groups)
GROUP_COLORS = {
    "Fertilizers": "#2E86AB",
    "Seeds & Crop Protection": "#A23B72",
    "Farm Machinery & Equipment": "#F18F01",
    "Animal Health": "#C73E1D",
    "Agricultural Trading & Processing": "#3B1F2B",
    "Food Processing": "#95C623",
    "Food Distribution": "#6A994E",
    "Farmland REIT": "#BC6C25",
    "Aquaculture": "#0077B6",
    "Retail": "#9B5DE5",
    "Other": "#888888",
}

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
    node_colors = [GROUP_COLORS.get(ticker_to_group.get(n, "Other"), "#888888") for n in G_gc.nodes()]
    nx.draw_networkx_nodes(G_gc, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G_gc, pos, alpha=0.4, width=0.8, ax=ax)
    labels_to_show = {n: n for n in G_gc.nodes() if degrees[n] >= LABEL_DEGREE_MIN}
    nx.draw_networkx_labels(G_gc, pos, labels=labels_to_show, font_size=7, ax=ax, font_color="black")

# Legend for groups (unique groups across all panels)
all_groups = set()
for G_gc in graphs:
    if G_gc is not None:
        for n in G_gc.nodes():
            all_groups.add(ticker_to_group.get(n, "Other"))
legend_handles = [plt.matplotlib.patches.Patch(facecolor=GROUP_COLORS.get(g, "#888888"),
                edgecolor="none", label=g) for g in sorted(all_groups)]
fig.legend(handles=legend_handles, loc="lower center", ncol=min(4, len(legend_handles)),
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.04))

fig.text(0.5, 0.01, "Edges: pairs with |S₁| > 2 (Bell inequality violation)",
         ha="center", fontsize=9, style="italic")
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.26, top=0.92, wspace=0.12)
outpath = "Figures/Fig03_networks.png"
plt.savefig(outpath, dpi=300)
plt.savefig(outpath.replace(".png", ".svg"))
plt.close()
print(f"Saved {outpath}")
