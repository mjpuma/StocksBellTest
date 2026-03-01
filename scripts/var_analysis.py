#!/usr/bin/env python3
"""
VAR: Violation %, volatility, and optionally sector-level violation %.

Fits a Vector Autoregression on:
  - violation_pct (overall daily violation %)
  - volatility (regime_vol)
  - within_sector_violation_pct (optional: violation % for within-group pairs only)

Outputs Results/var_*.csv, Figures/Fig09_var_*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIOLATION_PCT = os.path.join(PROJECT_ROOT, "Results", "violation_pct.csv")
VOL_FILE = os.path.join(PROJECT_ROOT, "Results", "volatility_traces", "regime_vol.csv")
S1_FILE = os.path.join(PROJECT_ROOT, "Results", "s1_values.csv")
TICKERS_FILE = os.path.join(PROJECT_ROOT, "yfinance_tickers.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results")
FIG_DIR = os.path.join(PROJECT_ROOT, "Figures")
VIOLATION_THRESHOLD = 2.0
MAX_LAGS = 10
INCLUDE_SECTOR = True


def build_within_sector_violation_pct(s1_df, ticker_to_group):
    """Daily violation % for within-group pairs only."""
    s1 = s1_df.copy()
    s1["violation"] = s1["S1"].abs() > VIOLATION_THRESHOLD

    def pair_type(row):
        ga = ticker_to_group.get(row["PairA"], None)
        gb = ticker_to_group.get(row["PairB"], None)
        if ga is None or gb is None:
            return None
        return "within" if ga == gb else "cross"

    s1["pair_type"] = s1.apply(pair_type, axis=1)
    within = s1[s1["pair_type"] == "within"]
    agg = within.groupby("Date").agg(
        total=("S1", "count"),
        violations=("violation", "sum"),
    ).reset_index()
    agg["within_sector_violation_pct"] = 100 * agg["violations"] / agg["total"]
    return agg[["Date", "within_sector_violation_pct"]]


def main():
    os.chdir(PROJECT_ROOT)
    os.makedirs(FIG_DIR, exist_ok=True)

    vp = pd.read_csv(VIOLATION_PCT)
    vp["Date"] = pd.to_datetime(vp["Date"])
    vp = vp[["Date", "ViolationPct"]].rename(columns={"ViolationPct": "violation_pct"})

    vol = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"])
    vol = vol[vol["Date"].notna()].copy()
    vol = vol[["Date", "Volatility"]].rename(columns={"Volatility": "volatility"})

    # Merge on date
    df = vp.merge(vol, on="Date", how="inner").dropna().sort_values("Date").reset_index(drop=True)

    if INCLUDE_SECTOR:
        tickers = pd.read_csv(TICKERS_FILE)
        ticker_to_group = dict(zip(tickers["ticker"], tickers["group"]))
        s1 = pd.read_csv(S1_FILE, parse_dates=["Date"])
        within_df = build_within_sector_violation_pct(s1, ticker_to_group)
        within_df["Date"] = pd.to_datetime(within_df["Date"])
        df = df.merge(within_df, on="Date", how="left")
        df["within_sector_violation_pct"] = df["within_sector_violation_pct"].fillna(0)

    # Use levels; VAR with constant can handle mildly non-stationary series
    var_cols = ["violation_pct", "volatility"]
    if INCLUDE_SECTOR and "within_sector_violation_pct" in df.columns:
        var_cols.append("within_sector_violation_pct")

    var_df = df[["Date"] + var_cols].dropna().set_index("Date")

    # Fit VAR
    model = VAR(var_df)
    result = model.fit(maxlags=MAX_LAGS, ic="aic")
    lag_order = result.k_ar

    # Summary
    summary_path = os.path.join(OUT_DIR, "var_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"VAR lag order (AIC): {lag_order}\n")
        f.write(str(result.summary()))
    print(f"Saved {summary_path}")

    # Granger causality
    gc = result.test_causality(var_cols[0], var_cols[1:], kind="f")
    gc_path = os.path.join(OUT_DIR, "var_granger.csv")
    granger_rows = []
    for cause in var_cols:
        for effect in var_cols:
            if cause == effect:
                continue
            try:
                t = result.test_causality(effect, [cause], kind="f")
                granger_rows.append({
                    "cause": cause,
                    "effect": effect,
                    "p_value": t.pvalue,
                    "significant": t.pvalue < 0.05,
                })
            except Exception:
                pass
    pd.DataFrame(granger_rows).to_csv(gc_path, index=False)
    print(f"Saved {gc_path}")

    # Impulse response figure (publication style)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    irf = result.irf(10)
    nv = len(var_cols)
    fig, axes = plt.subplots(nv, nv, figsize=(4.5 * nv, 3.5 * nv))
    if nv == 1:
        axes = np.array([[axes]])
    panel_letters = "ABCDEFGHI"
    idx = 0
    for i, resp_var in enumerate(var_cols):
        for j, shock_var in enumerate(var_cols):
            ax = axes[i, j]
            irf_vals = irf.irfs[:, i, j]
            ax.plot(range(len(irf_vals)), irf_vals, color="#1a5276", linewidth=1.5)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_title(f"{shock_var} → {resp_var}")
            ax.set_xlabel("Horizon")
            ax.text(-0.15, 1.02, f"({panel_letters[idx]})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            idx += 1
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig09_var_irf.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Fig09_var_irf.svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved Figures/Fig09_var_irf.*")

    # Variance decomposition (decomp shape: n_vars x n_horizons x n_vars)
    vd = result.fevd(10)
    fevd_path = os.path.join(OUT_DIR, "var_fevd.csv")
    fevd_rows = []
    n_h = vd.decomp.shape[1]
    for step in range(n_h):
        for i, var_name in enumerate(var_cols):
            for j, shock_name in enumerate(var_cols):
                fevd_rows.append({
                    "horizon": step,
                    "variable": var_name,
                    "shock_from": shock_name,
                    "fevd": float(vd.decomp[i, step, j]),
                })
    pd.DataFrame(fevd_rows).to_csv(fevd_path, index=False)
    print(f"Saved {fevd_path}")


if __name__ == "__main__":
    main()
