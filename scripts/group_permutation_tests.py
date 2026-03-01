#!/usr/bin/env python3
"""
Group-level and category-level permutation tests: network metrics by sector and MNC vs Pure Ag.

Uses group_stats.csv, category_stats.csv, and regime_vol.csv for extreme/normal classification.
Outputs multi-panel publication figures: Fig06_group_permtest.png (3 panels), Fig06_category_permtest.png (3 panels).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP_STATS = os.path.join(PROJECT_ROOT, "Results", "group_stats.csv")
CATEGORY_STATS = os.path.join(PROJECT_ROOT, "Results", "category_stats.csv")
VOL_FILE = os.path.join(PROJECT_ROOT, "Results", "volatility_traces", "regime_vol.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results")
FIG_DIR = os.path.join(PROJECT_ROOT, "Figures")
VOL_THRESHOLD = 0.4
N_PERM = 10000  # fewer for speed; increase for paper

METRICS = [
    ("MeanDegree", "Mean Degree", "% Change in Mean Degree"),
    ("MeanClustering", "Mean Clustering", "% Change in Mean Clustering"),
    ("MeanBetweenness", "Mean Betweenness", "% Change in Mean Betweenness"),
]

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
})


def permutation_test(x, y, n_perm, seed=42):
    x, y = np.asarray(x), np.asarray(y)
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    obs = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n_x = x.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        d = perm[:n_x].mean() - perm[n_x:].mean()
        if abs(d) >= abs(obs):
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return obs, p_val


def main():
    os.chdir(PROJECT_ROOT)
    os.makedirs(FIG_DIR, exist_ok=True)

    vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"]).sort_values("Date")
    vol_df = vol_df[vol_df["Date"].notna()].reset_index(drop=True)
    mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
    ext_dates = set(vol_df.loc[mask_extreme, "Date"].dt.normalize())
    norm_dates = set(vol_df.loc[~mask_extreme, "Date"].dt.normalize())

    gs = pd.read_csv(GROUP_STATS)
    gs["Date"] = pd.to_datetime(gs["Date"]).dt.normalize()
    gs["extreme"] = gs["Date"].isin(ext_dates)

    group_dfs = {}
    for col, title_label, xlabel in METRICS:
        results = []
        for group in gs["group"].unique():
            sub = gs[gs["group"] == group]
            x = sub[sub["extreme"]][col].dropna().values
            y = sub[~sub["extreme"]][col].dropna().values
            if len(x) < 5 or len(y) < 5:
                continue
            node_count = int(sub["NodeCount"].iloc[0])
            obs_ext, obs_norm = x.mean(), y.mean()
            denom = abs(obs_norm) if obs_norm != 0 else 1e-10
            pct = 100 * (obs_ext - obs_norm) / denom
            delta, p_val = permutation_test(x, y, N_PERM)
            results.append({
                "Group": group,
                "NodeCount": node_count,
                "Extreme": obs_ext,
                "Normal": obs_norm,
                "Percent_change": pct,
                "p_value": p_val,
            })
        res_df = pd.DataFrame(results)
        group_dfs[col] = res_df
        out_csv = os.path.join(OUT_DIR, f"group_permtest_{col}.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

    # Multi-panel Fig06: group-level (A, B, C)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.5), sharey=False)
    for idx, (col, title_label, xlabel) in enumerate(METRICS):
        ax = axes[idx]
        plot_df = group_dfs[col].sort_values("Percent_change", ascending=True)
        colors = ["#1a5276" if p < 0.05 else "#95a5a6" for p in plot_df["p_value"]]
        bars = ax.barh(plot_df["Group"], plot_df["Percent_change"], color=colors, height=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8, zorder=0)
        ax.set_xlabel(f"% change (extreme vs normal)")
        ax.set_ylabel("") if idx > 0 else ax.set_ylabel("Sector")
        ax.set_title(title_label)
        ax.text(-0.12, 1.02, "(" + "ABC"[idx] + ")", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "Fig06_group_permtest.png")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_fig.replace(".png", ".svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved {out_fig}")

    # Category-level (MNC vs Pure Ag)
    category_dfs = {}
    if os.path.exists(CATEGORY_STATS):
        cs = pd.read_csv(CATEGORY_STATS)
        cs["Date"] = pd.to_datetime(cs["Date"]).dt.normalize()
        cs["extreme"] = cs["Date"].isin(ext_dates)
        cs = cs[cs["category"].isin(["MNC", "Pure Ag"])]

        for col, title_label, xlabel in METRICS:
            results = []
            for cat in cs["category"].unique():
                sub = cs[cs["category"] == cat]
                x = sub[sub["extreme"]][col].dropna().values
                y = sub[~sub["extreme"]][col].dropna().values
                if len(x) < 5 or len(y) < 5:
                    continue
                node_count = int(sub["NodeCount"].iloc[0])
                obs_ext, obs_norm = x.mean(), y.mean()
                denom = abs(obs_norm) if obs_norm != 0 else 1e-10
                pct = 100 * (obs_ext - obs_norm) / denom
                delta, p_val = permutation_test(x, y, N_PERM)
                results.append({
                    "Category": cat,
                    "NodeCount": node_count,
                    "Extreme": obs_ext,
                    "Normal": obs_norm,
                    "Percent_change": pct,
                    "p_value": p_val,
                })
            res_df = pd.DataFrame(results)
            category_dfs[col] = res_df
            out_csv = os.path.join(OUT_DIR, f"category_permtest_{col}.csv")
            res_df.to_csv(out_csv, index=False)
            print(f"Saved {out_csv}")

        # Multi-panel Fig06_category: MNC vs Pure Ag (D, E, F)
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=True)
        for idx, (col, title_label, xlabel) in enumerate(METRICS):
            ax = axes[idx]
            plot_df = category_dfs[col].sort_values("Percent_change", ascending=True)
            colors = ["#1a5276" if p < 0.05 else "#95a5a6" for p in plot_df["p_value"]]
            ax.barh(plot_df["Category"], plot_df["Percent_change"], color=colors, height=0.6, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.8, zorder=0)
            ax.set_xlabel(f"% change (extreme vs normal)")
            ax.set_ylabel("") if idx > 0 else ax.set_ylabel("Category")
            ax.set_title(title_label)
            ax.text(-0.12, 1.02, "(" + "DEF"[idx] + ")", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out_fig = os.path.join(FIG_DIR, "Fig06_category_permtest.png")
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        fig.savefig(out_fig.replace(".png", ".svg"), bbox_inches="tight")
        plt.close()
        print(f"Saved {out_fig}")

    # Backward compatibility: group_permtest.csv
    gs_deg = pd.read_csv(os.path.join(OUT_DIR, "group_permtest_MeanDegree.csv"))
    gs_deg.to_csv(os.path.join(OUT_DIR, "group_permtest.csv"), index=False)


if __name__ == "__main__":
    main()
