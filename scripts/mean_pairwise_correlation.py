#!/usr/bin/env python3
"""
Mean pairwise correlation: crisis vs normal days.

Uses Results/returns.csv and regime_vol.csv for extreme/normal classification.
Outputs Results/mean_pairwise_corr.csv, Figures/Fig08_mean_pairwise_corr.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETURNS_FILE = os.path.join(PROJECT_ROOT, "Results", "returns.csv")
VOL_FILE = os.path.join(PROJECT_ROOT, "Results", "volatility_traces", "regime_vol.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "Results", "mean_pairwise_corr.csv")
OUT_FIG = os.path.join(PROJECT_ROOT, "Figures", "Fig08_mean_pairwise_corr.png")
VOL_THRESHOLD = 0.4
N_PERM = 10000


WINDOW = 20


def daily_mean_pairwise_corr(returns_df):
    """Compute daily mean pairwise Pearson correlation using rolling window."""
    ret = returns_df.dropna(axis=1, how="all")
    n_tickers = ret.shape[1]
    if n_tickers < 2:
        return pd.Series(dtype=float)
    dates = ret.index[WINDOW:]
    corrs = []
    for i in range(WINDOW, len(ret)):
        window = ret.iloc[i - WINDOW : i]
        valid = window.dropna(axis=1, how="any")
        if valid.shape[1] < 2:
            corrs.append(np.nan)
            continue
        cmat = valid.corr()
        triu = np.triu_indices(valid.shape[1], k=1)
        mean_corr = cmat.values[triu].mean()
        corrs.append(mean_corr)
    return pd.Series(corrs, index=dates)


def permutation_test(x, y, n_perm, seed=42):
    x, y = np.asarray(x), np.asarray(y)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
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
    os.makedirs("Figures", exist_ok=True)

    if not os.path.exists(RETURNS_FILE):
        raise FileNotFoundError(
            f"{RETURNS_FILE} not found. Run 0.py first to generate returns."
        )

    ret = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    daily_corr = daily_mean_pairwise_corr(ret)
    daily_corr = daily_corr.dropna()

    vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"])
    vol_df["Date"] = pd.to_datetime(vol_df["Date"]).dt.normalize()
    mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
    ext_dates = set(vol_df.loc[mask_extreme, "Date"])
    norm_dates = set(vol_df.loc[~mask_extreme, "Date"])

    corr_dates = daily_corr.index.normalize()
    ext_corr = daily_corr[corr_dates.isin(ext_dates)].values
    norm_corr = daily_corr[corr_dates.isin(norm_dates)].values

    delta, p_val = permutation_test(ext_corr, norm_corr, N_PERM)

    summary = pd.DataFrame([{
        "Regime": "Extreme",
        "Mean_corr": float(np.nanmean(ext_corr)),
        "N_days": len(ext_corr),
    }, {
        "Regime": "Normal",
        "Mean_corr": float(np.nanmean(norm_corr)),
        "N_days": len(norm_corr),
    }])
    summary.to_csv(OUT_CSV, index=False)
    with open(OUT_CSV.replace(".csv", "_permtest.txt"), "w") as f:
        f.write(f"Delta (extreme - normal): {delta:.6f}\n")
        f.write(f"p-value (permutation): {p_val:.6f}\n")

    print(f"Saved {OUT_CSV}")
    print(f"Delta: {delta:.6f}, p-value: {p_val:.6f}")

    # Figure: bar chart extreme vs normal (publication style)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(4, 3.5))
    vals = [float(np.nanmean(ext_corr)), float(np.nanmean(norm_corr))]
    colors = ["#1a5276", "#3498db"]
    ax.bar(["Extreme", "Normal"], vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean pairwise correlation")
    ax.set_title("Crisis vs normal")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".png", ".svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
