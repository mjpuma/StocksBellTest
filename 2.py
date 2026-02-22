"""
Build daily networks from S1 values (edge if S1 > 2).

Requires Results/s1_values.csv from 0.py.
Outputs Results/networks/*.csv and Figures/networks/*.html
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms import community
from scipy.stats import powerlaw, entropy


S1_FILE = "Results/s1_values.csv"
NETWORK_HTML_DIR = "Figures/networks"
METRICS_DIR = "Results/networks"

os.makedirs(NETWORK_HTML_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

df = pd.read_csv(S1_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

ALL_NODES = sorted(set(df["PairA"]).union(set(df["PairB"])))

def create_network(df_day, threshold=2.0):
    """Create undirected network where edges satisfy S1 > threshold, but include all nodes."""
    edges = df_day[df_day["S1"] > threshold][["PairA", "PairB", "S1"]]
    G = nx.Graph()
    G.add_nodes_from(ALL_NODES)  
    for _, row in edges.iterrows():
        G.add_edge(row["PairA"], row["PairB"], weight=row["S1"])
    return G

def calculate_global_metrics(G):
    if G.number_of_nodes() == 0:
        return {k: np.nan for k in [
            "NumNodes","NumEdges","Density","GiantComponentSizePct","AvgClusteringCoeff",
            "GlobalClustering","Efficiency","AvgShortestPathLength","Diameter",
            "DegreeCentralization","BetweennessCentralization","Modularity",
            "CommunitySizeEntropy","NumCommunities","Assortativity","ScaleFreeAlpha"
        ]}
    
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if m == 0:
        return {
            "NumNodes": n, "NumEdges": 0, "Density": 0,
            "GiantComponentSizePct": 0, "AvgClusteringCoeff": 0,
            "GlobalClustering": 0, "Efficiency": 0,
            "AvgShortestPathLength": np.nan, "Diameter": np.nan,
            "DegreeCentralization": 0, "BetweennessCentralization": 0,
            "Modularity": np.nan, "CommunitySizeEntropy": np.nan,
            "NumCommunities": 0, "Assortativity": np.nan,
            "ScaleFreeAlpha": np.nan
        }

    #Density
    density = nx.density(G)
    
    #Giant Component Size
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_gc = G.subgraph(largest_cc)
    gc_size_pct = len(largest_cc) / n * 100

    #Average Clustering Coefficient
    avg_clustering = nx.average_clustering(G)

    #Global Clustering Coefficient/ Transitivity
    global_clustering = nx.transitivity(G)

    try:
        eff = nx.global_efficiency(G_gc)
        asp = nx.average_shortest_path_length(G_gc)
        diam = nx.diameter(G_gc)
    except Exception:
        eff, asp, diam = np.nan, np.nan, np.nan

    degrees = np.array([v for _, v in G.degree()])
    if len(degrees) > 1:
        degree_centralization = ((degrees.max() * len(degrees) - degrees.sum()) /
                                 ((len(degrees)-1)**2))
    else:
        degree_centralization = np.nan

    bet = nx.betweenness_centrality(G)
    if len(bet) > 1:
        max_b = max(bet.values())
        bet_centralization = sum(max_b - np.array(list(bet.values()))) / (len(G)**2 - 3*len(G) + 2)
    else:
        bet_centralization = np.nan

    try:
        comms = list(community.greedy_modularity_communities(G))
        modularity = community.modularity(G, comms)
        comm_sizes = np.array([len(c) for c in comms])
        comm_entropy = entropy(comm_sizes / comm_sizes.sum())
        num_comms = len(comms)
    except Exception:
        modularity, comm_entropy, num_comms = np.nan, np.nan, np.nan

    try:
        assort = nx.degree_pearson_correlation_coefficient(G)
    except Exception:
        assort = np.nan

    degs = [d for _, d in G.degree()]
    try:
        a, _, _ = powerlaw.fit(degs, floc=0, method="MM")
        scale_free_alpha = a
    except Exception:
        scale_free_alpha = np.nan

    return {
        "NumNodes": n,
        "NumEdges": m,
        "Density": density,
        "GiantComponentSizePct": gc_size_pct,
        "AvgClusteringCoeff": avg_clustering,
        "GlobalClustering": global_clustering,
        "Efficiency": eff,
        "AvgShortestPathLength": asp,
        "Diameter": diam,
        "DegreeCentralization": degree_centralization,
        "BetweennessCentralization": bet_centralization,
        "Modularity": modularity,
        "CommunitySizeEntropy": comm_entropy,
        "NumCommunities": num_comms,
        "Assortativity": assort,
        "ScaleFreeAlpha": scale_free_alpha
    }

def calculate_node_metrics(G):
    """Compute node-level metrics for all 54 nodes."""
    deg = dict(G.degree())
    cluster = nx.clustering(G)
    between = nx.betweenness_centrality(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    close = nx.closeness_centrality(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    except Exception:
        eig = {n: np.nan for n in G.nodes()}
    
    df_nodes = pd.DataFrame({
        "Node": list(G.nodes()),
        "Degree": [deg.get(n, 0) for n in G.nodes()],
        "ClusteringCoeff": [cluster.get(n, 0) for n in G.nodes()],
        "Betweenness": [between.get(n, 0) for n in G.nodes()],
        "Closeness": [close.get(n, 0) for n in G.nodes()],
        "EigenvectorCentrality": [eig.get(n, np.nan) for n in G.nodes()],
    })
    return df_nodes

def plot_network(G, title, out_html):
    """Save interactive Plotly network visualization."""
    pos = nx.spring_layout(G, seed=42)
    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=0.5, color="#888"),
            hoverinfo="text",
            text=f"{u} – {v}<br>|S₁| = {d.get('weight', np.nan):.2f}",
            mode="lines"
        ))

    degrees = dict(G.degree())
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = [f"{n}<br>Degree = {degrees[n]}" for n in G.nodes()]
    node_size = [4 + 2 * degrees[n] for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color=list(degrees.values()),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Degree"),
            line_width=0.5
        )
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            font=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    fig.write_html(out_html)

#========================================================================================================
#Main Loop
#========================================================================================================

for date, df_day in df.groupby("Date"):
    G = create_network(df_day)
    date_str = date.strftime("%Y-%m-%d")

    html_path = os.path.join(NETWORK_HTML_DIR, f"{date_str}.html")
    plot_network(G, f"Network {date_str}", html_path)

    metrics = calculate_global_metrics(G)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "Date", date_str)
    metrics_df.to_csv(os.path.join(METRICS_DIR, f"{date_str}_metrics.csv"), index=False)

    node_df = calculate_node_metrics(G)
    node_df.insert(0, "Date", date_str)
    node_df.to_csv(os.path.join(METRICS_DIR, f"{date_str}_properties.csv"), index=False)

    print(f"Processed {date_str}: {len(G)} nodes, {G.number_of_edges()} edges")

