"""
Granger causality tests: violation % vs regime volatility (both directions), by crisis window.

Crisis windows (Option A): 2008-09→2009-03, 2020-02→2020-06, 2022-02→2022-06.
Max lag 10 per crisis (narrower windows).

Requires Results/violation_pct.csv and Results/volatility_traces/regime_vol.csv.
Outputs Results/granger_results.csv, Results/granger_results.tex, Figures/Fig05_granger_causality.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

os.makedirs("Figures", exist_ok=True)
os.makedirs("Results", exist_ok=True)

VIOLATION_FILE = "Results/violation_pct.csv"
VOL_FILE = "Results/volatility_traces/regime_vol.csv"
OUT_CSV = "Results/granger_results.csv"
OUT_TEX = "Results/granger_results.tex"
OUT_FIG = "Figures/Fig05_granger_causality.png"

MAX_LAG = 10
SIG = 0.05

# Option A: narrower crisis windows
CRISIS_WINDOWS = [
    ("2008 Financial Crisis", "2008-09-01", "2009-03-31"),
    ("COVID-19", "2020-02-01", "2020-06-30"),
    ("Ukraine War", "2022-02-01", "2022-06-30"),
]

s1 = pd.read_csv(VIOLATION_FILE, parse_dates=["Date"])
vol = pd.read_csv(VOL_FILE, parse_dates=["Date"])
s1 = s1.rename(columns={"ViolationPct": "S1_value"})
vol = vol.rename(columns={"Volatility": "vol"})

df = pd.merge(
    s1[["Date", "S1_value"]],
    vol[["Date", "vol"]],
    on="Date",
    how="inner"
).dropna().reset_index(drop=True)

results = []

for crisis_name, start_str, end_str in CRISIS_WINDOWS:
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    sub = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if len(sub) < 30:
        print(f"Warning: {crisis_name} has only {len(sub)} observations, skipping")
        continue

    data1 = sub[["vol", "S1_value"]]
    gc1 = grangercausalitytests(data1, maxlag=MAX_LAG, verbose=False)

    for lag in range(1, MAX_LAG + 1):
        pval = gc1[lag][0]["ssr_ftest"][1]
        results.append({
            "Crisis": crisis_name,
            "Direction": "S1 → Volatility",
            "Lag": lag,
            "p Value": pval,
            "q Value": pval
        })

    data2 = sub[["S1_value", "vol"]]
    gc2 = grangercausalitytests(data2, maxlag=MAX_LAG, verbose=False)

    for lag in range(1, MAX_LAG + 1):
        pval = gc2[lag][0]["ssr_ftest"][1]
        results.append({
            "Crisis": crisis_name,
            "Direction": "Volatility → S1",
            "Lag": lag,
            "p Value": pval,
            "q Value": pval
        })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV}")

# LaTeX export (supplement)
def fmt_pval(p):
    if p < 1e-10:
        return r"$<10^{-10}$"
    s = f"{p:.2e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")

tex_df = res_df.copy()
tex_df["p Value"] = tex_df["p Value"].apply(fmt_pval)
tex_df["q Value"] = tex_df["q Value"].apply(fmt_pval)
tex_df = tex_df.rename(columns={"p Value": "p", "q Value": "q"})

tex_str = tex_df.to_latex(index=False, escape=False, column_format="lllcc")
with open(OUT_TEX, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Granger causality tests by crisis: S$_1$ violation \\% vs regime volatility (lags 1--10).}\n")
    f.write("\\label{tab:granger}\n")
    f.write(tex_str)
    f.write("\\end{table}\n")
print(f"Saved {OUT_TEX}")

# Bar chart: grouped bars per crisis, two panels (Vol→S1, S1→Vol)
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

crises = [c[0] for c in CRISIS_WINDOWS]
colors = ["steelblue", "coral", "mediumseagreen"]
x = np.arange(MAX_LAG) + 1
width = 0.25

for i, crisis in enumerate(crises):
    vol_to_s1 = res_df[(res_df["Crisis"] == crisis) & (res_df["Direction"] == "Volatility → S1")]
    s1_to_vol = res_df[(res_df["Crisis"] == crisis) & (res_df["Direction"] == "S1 → Volatility")]
    if len(vol_to_s1) == 0:
        continue
    p_clip = lambda v: max(v, 1e-20)
    neglog10 = lambda v: -np.log10(p_clip(v))
    offset = (i - 1) * width
    c = colors[i % len(colors)]
    axes[0].bar(x + offset, vol_to_s1["p Value"].apply(neglog10), width=width, label=crisis, color=c, alpha=0.8)
    axes[1].bar(x + offset, s1_to_vol["p Value"].apply(neglog10), width=width, label=crisis, color=c, alpha=0.8)

sig_line = -np.log10(SIG)
axes[0].axhline(sig_line, color="red", linestyle="--", linewidth=1, label="α = 0.05")
axes[0].set_ylabel(r"$-\log_{10}$(p)")
axes[0].set_title("(A) Volatility → S₁")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].axhline(sig_line, color="red", linestyle="--", linewidth=1, label="α = 0.05")
axes[1].set_xlabel("Lag (days)")
axes[1].set_ylabel(r"$-\log_{10}$(p)")
axes[1].set_title("(B) S₁ → Volatility")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.savefig(OUT_FIG.replace(".png", ".svg"))
plt.close()
print(f"Saved {OUT_FIG}")
