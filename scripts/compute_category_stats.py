#!/usr/bin/env python3
"""
Compute network metrics stratified by category (MNC vs Pure Ag) and by group.

Reads Results/networks/*_properties.csv (per-node metrics) and yfinance_tickers.csv,
aggregates by Date + Category and Date + Group, outputs Results/category_stats.csv
and Results/group_stats.csv for stratified analysis.
"""

import os
import glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(PROJECT_ROOT, "Results", "networks")
TICKER_FILE = os.path.join(PROJECT_ROOT, "yfinance_tickers.csv")
OUT_CATEGORY = os.path.join(PROJECT_ROOT, "Results", "category_stats.csv")
OUT_GROUP = os.path.join(PROJECT_ROOT, "Results", "group_stats.csv")


def main():
    os.chdir(PROJECT_ROOT)

    prop_files = sorted(glob.glob(os.path.join(METRICS_DIR, "*_properties.csv")))
    if not prop_files:
        print("No *_properties.csv found in Results/networks. Run 2.py first.")
        return

    node_dfs = []
    for f in prop_files:
        df = pd.read_csv(f)
        node_dfs.append(df)
    nodes_df = pd.concat(node_dfs, ignore_index=True)
    nodes_df["Date"] = pd.to_datetime(nodes_df["Date"])

    tickers_df = pd.read_csv(TICKER_FILE)
    ticker_to_category = dict(zip(tickers_df["ticker"].astype(str).str.strip(),
                                  tickers_df["category"].astype(str).str.strip()))
    ticker_to_group = dict(zip(tickers_df["ticker"].astype(str).str.strip(),
                               tickers_df["group"].astype(str).str.strip()))

    nodes_df["category"] = nodes_df["Node"].map(lambda n: ticker_to_category.get(str(n), "Other"))
    nodes_df["group"] = nodes_df["Node"].map(lambda n: ticker_to_group.get(str(n), "Other"))

    # Aggregate by Date, Category
    cat_stats = nodes_df.groupby(["Date", "category"]).agg(
        NodeCount=("Node", "count"),
        MeanDegree=("Degree", "mean"),
        MeanClustering=("ClusteringCoeff", "mean"),
        MeanBetweenness=("Betweenness", "mean"),
    ).reset_index()

    # Aggregate by Date, Group
    grp_stats = nodes_df.groupby(["Date", "group"]).agg(
        NodeCount=("Node", "count"),
        MeanDegree=("Degree", "mean"),
        MeanClustering=("ClusteringCoeff", "mean"),
        MeanBetweenness=("Betweenness", "mean"),
    ).reset_index()

    os.makedirs(os.path.dirname(OUT_CATEGORY), exist_ok=True)
    cat_stats.to_csv(OUT_CATEGORY, index=False)
    grp_stats.to_csv(OUT_GROUP, index=False)
    print(f"Saved {OUT_CATEGORY} ({len(cat_stats)} rows)")
    print(f"Saved {OUT_GROUP} ({len(grp_stats)} rows)")


if __name__ == "__main__":
    main()
