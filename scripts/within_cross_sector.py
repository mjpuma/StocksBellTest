#!/usr/bin/env python3
"""
Within vs cross-sector violations: violation % by pair type, by group, and by category.

Uses s1_values.csv and yfinance_tickers.csv for group/category mapping.
Outputs multi-panel Fig07: (A) within vs cross, (B) by sector, (C) by category (MNC vs Pure Ag).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1_FILE = os.path.join(PROJECT_ROOT, "Results", "s1_values.csv")
TICKERS_FILE = os.path.join(PROJECT_ROOT, "yfinance_tickers.csv")
VOL_FILE = os.path.join(PROJECT_ROOT, "Results", "volatility_traces", "regime_vol.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "Results", "within_cross_sector.csv")
VIOLATION_BY_GROUP_CSV = os.path.join(PROJECT_ROOT, "Results", "violation_by_group.csv")
VIOLATION_BY_CATEGORY_CSV = os.path.join(PROJECT_ROOT, "Results", "violation_by_category.csv")
FIG07_MULTIPANEL = os.path.join(PROJECT_ROOT, "Figures", "Fig07_violations_multipanel.png")
VOL_THRESHOLD = 0.4
VIOLATION_THRESHOLD = 2.0  # |S1| > 2
N_PERM = 10000


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
        if abs(perm[:n_x].mean() - perm[n_x:].mean()) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    os.chdir(PROJECT_ROOT)
    os.makedirs("Figures", exist_ok=True)

    tickers = pd.read_csv(TICKERS_FILE)
    ticker_to_group = dict(zip(tickers["ticker"], tickers["group"]))
    ticker_to_category = dict(zip(tickers["ticker"], tickers["category"]))

    s1 = pd.read_csv(S1_FILE)
    s1["Date"] = pd.to_datetime(s1["Date"])
    s1["violation"] = s1["S1"].abs() > VIOLATION_THRESHOLD

    def pair_type(row):
        ga = ticker_to_group.get(row["PairA"], None)
        gb = ticker_to_group.get(row["PairB"], None)
        if ga is None or gb is None:
            return "unknown"
        return "within" if ga == gb else "cross"

    s1["pair_type"] = s1.apply(pair_type, axis=1)
    s1 = s1[s1["pair_type"] != "unknown"]

    vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"])
    vol_df["Date"] = pd.to_datetime(vol_df["Date"]).dt.normalize()
    mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
    ext_dates = set(vol_df.loc[mask_extreme, "Date"])
    norm_dates = set(vol_df.loc[~mask_extreme, "Date"])

    s1["extreme"] = s1["Date"].dt.normalize().isin(ext_dates)

    agg = s1.groupby(["Date", "pair_type", "extreme"]).agg(
        total=("S1", "count"),
        violations=("violation", "sum"),
    ).reset_index()
    agg["violation_pct"] = 100 * agg["violations"] / agg["total"]

    # Aggregate by extreme vs normal
    summary = agg.groupby(["pair_type", "extreme"]).agg(
        mean_violation_pct=("violation_pct", "mean"),
        std_violation_pct=("violation_pct", "std"),
        n_days=("Date", "nunique"),
    ).reset_index()

    summary.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # Violation % by group
    s1["group"] = s1.apply(
        lambda r: ticker_to_group.get(r["PairA"]) if ticker_to_group.get(r["PairA"]) == ticker_to_group.get(r["PairB"]) else None,
        axis=1,
    )
    within_only = s1[s1["group"].notna()].copy()
    grp_agg = within_only.groupby(["Date", "group"]).agg(
        total=("S1", "count"),
        violations=("violation", "sum"),
    ).reset_index()
    grp_agg["violation_pct"] = 100 * grp_agg["violations"] / grp_agg["total"]
    grp_agg["Date"] = pd.to_datetime(grp_agg["Date"]).dt.normalize()
    vol_lookup = vol_df.drop_duplicates("Date").set_index("Date")
    vol_lookup["extreme"] = vol_lookup["Volatility"] > VOL_THRESHOLD
    grp_agg = grp_agg.merge(vol_lookup[["extreme"]], left_on="Date", right_index=True, how="inner")
    grp_results = []
    for group in grp_agg["group"].unique():
        sub = grp_agg[grp_agg["group"] == group]
        x = sub.loc[sub["extreme"], "violation_pct"].values
        y = sub.loc[~sub["extreme"], "violation_pct"].values
        if len(x) < 5 or len(y) < 5:
            continue
        obs_ext, obs_norm = x.mean(), y.mean()
        pct = 100 * (obs_ext - obs_norm) / (abs(obs_norm) + 1e-10)
        delta, p_val = permutation_test(x, y, N_PERM)
        grp_results.append({
            "Group": group,
            "Extreme": obs_ext,
            "Normal": obs_norm,
            "Percent_change": pct,
            "p_value": p_val,
        })
    grp_df = pd.DataFrame(grp_results)
    grp_df.to_csv(VIOLATION_BY_GROUP_CSV, index=False)
    print(f"Saved {VIOLATION_BY_GROUP_CSV}")

    # Violation % by category (MNC vs Pure Ag)
    s1["category"] = s1.apply(
        lambda r: ticker_to_category.get(r["PairA"]) if ticker_to_category.get(r["PairA"]) == ticker_to_category.get(r["PairB"]) else None,
        axis=1,
    )
    cat_only = s1[s1["category"].notna() & s1["category"].isin(["MNC", "Pure Ag"])].copy()
    cat_agg = cat_only.groupby(["Date", "category"]).agg(
        total=("S1", "count"),
        violations=("violation", "sum"),
    ).reset_index()
    cat_agg["violation_pct"] = 100 * cat_agg["violations"] / cat_agg["total"]
    cat_agg["Date"] = pd.to_datetime(cat_agg["Date"]).dt.normalize()
    cat_agg = cat_agg.merge(vol_lookup[["extreme"]], left_on="Date", right_index=True, how="inner")
    cat_results = []
    for cat in cat_agg["category"].unique():
        sub = cat_agg[cat_agg["category"] == cat]
        x = sub.loc[sub["extreme"], "violation_pct"].values
        y = sub.loc[~sub["extreme"], "violation_pct"].values
        if len(x) < 5 or len(y) < 5:
            continue
        obs_ext, obs_norm = x.mean(), y.mean()
        pct = 100 * (obs_ext - obs_norm) / (abs(obs_norm) + 1e-10)
        delta, p_val = permutation_test(x, y, N_PERM)
        cat_results.append({
            "Category": cat,
            "Extreme": obs_ext,
            "Normal": obs_norm,
            "Percent_change": pct,
            "p_value": p_val,
        })
    cat_df = pd.DataFrame(cat_results)
    cat_df.to_csv(VIOLATION_BY_CATEGORY_CSV, index=False)
    print(f"Saved {VIOLATION_BY_CATEGORY_CSV}")

    # Multi-panel Fig07: (A) within vs cross, (B) by sector, (C) by category
    fig = plt.figure(figsize=(10.5, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.8, 0.8], wspace=0.35)

    # (A) Within vs cross
    ax_a = fig.add_subplot(gs[0])
    labels = ["Within\n(extreme)", "Within\n(normal)", "Cross\n(extreme)", "Cross\n(normal)"]
    vals = []
    for pt in ["within", "cross"]:
        for ext in [True, False]:
            row = summary[(summary["pair_type"] == pt) & (summary["extreme"] == ext)]
            vals.append(row["mean_violation_pct"].values[0] if len(row) else 0)
    colors = ["#c0392b", "#e8b4b8", "#2980b9", "#aed6f1"]
    ax_a.bar(range(4), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Violation % (|S₁| > 2)")
    ax_a.set_title("Pair type")
    ax_a.text(-0.15, 1.02, "(A)", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="y", linestyle="--", alpha=0.4)

    # (B) By sector
    ax_b = fig.add_subplot(gs[1])
    plot_df = grp_df.sort_values("Percent_change", ascending=True)
    colors_b = ["#1a5276" if p < 0.05 else "#95a5a6" for p in plot_df["p_value"]]
    ax_b.barh(plot_df["Group"], plot_df["Percent_change"], color=colors_b, height=0.7, edgecolor="white", linewidth=0.5)
    ax_b.axvline(0, color="black", linewidth=0.8, zorder=0)
    ax_b.set_xlabel("% change (extreme vs normal)")
    ax_b.set_title("Violation % by sector")
    ax_b.text(-0.08, 1.02, "(B)", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="x", linestyle="--", alpha=0.4)

    # (C) By category
    ax_c = fig.add_subplot(gs[2])
    plot_cat = cat_df.sort_values("Percent_change", ascending=True)
    colors_c = ["#1a5276" if p < 0.05 else "#95a5a6" for p in plot_cat["p_value"]]
    ax_c.barh(plot_cat["Category"], plot_cat["Percent_change"], color=colors_c, height=0.6, edgecolor="white", linewidth=0.5)
    ax_c.axvline(0, color="black", linewidth=0.8, zorder=0)
    ax_c.set_xlabel("% change (extreme vs normal)")
    ax_c.set_title("MNC vs Pure Ag")
    ax_c.text(-0.2, 1.02, "(C)", transform=ax_c.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="x", linestyle="--", alpha=0.4)

    plt.savefig(FIG07_MULTIPANEL, dpi=300, bbox_inches="tight")
    plt.savefig(FIG07_MULTIPANEL.replace(".png", ".svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG07_MULTIPANEL}")


if __name__ == "__main__":
    main()
