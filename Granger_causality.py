import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

S1_FILE = "Results/s1_values.csv"                 
VOL_FILE = "Results/volatility_traces/regime_vol.csv"
OUT_CSV = "Results/granger_results.csv"

MAX_LAG = 20
SIG = 0.05

s1 = pd.read_csv(S1_FILE, parse_dates=["Date"])
vol = pd.read_csv(VOL_FILE, parse_dates=["Date"])

s1 = s1.rename(columns={"S1": "S1_value"})
vol = vol.rename(columns={"Volatility": "vol"})

df = pd.merge(
    s1[["Date", "S1_value"]],
    vol[["Date", "vol"]],
    on="Date",
    how="inner"
).dropna().reset_index(drop=True)

results = []

data1 = df[["vol", "S1_value"]]
gc1 = grangercausalitytests(data1, maxlag=MAX_LAG, verbose=False)

for lag in range(1, MAX_LAG + 1):
    pval = gc1[lag][0]["ssr_ftest"][1]
    results.append({
        "Direction": "S1 → Volatility",
        "Lag": lag,
        "p Value": pval,
        "q Value": pval  
    })

data2 = df[["S1_value", "vol"]]
gc2 = grangercausalitytests(data2, maxlag=MAX_LAG, verbose=False)

for lag in range(1, MAX_LAG + 1):
    pval = gc2[lag][0]["ssr_ftest"][1]
    results.append({
        "Direction": "Volatility → S1",
        "Lag": lag,
        "p Value": pval,
        "q Value": pval
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_CSV, index=False)

cols = ["Direction", "Lag", "p Value", "q Value"]
cell_text = res_df[cols].round(5).values.tolist()

row_colors = []
for _, row in res_df.iterrows():
    if not pd.isna(row["q Value"]) and row["q Value"] < 0.05:
        row_colors.append(["#b3d9ff"] * len(cols))
    else:
        row_colors.append(["white"] * len(cols))

fig, ax = plt.subplots(figsize=(12, 0.6 * len(res_df) + 1.5))
ax.axis("off")

tbl = ax.table(
    cellText=cell_text,
    colLabels=cols,
    cellColours=row_colors,
    cellLoc="center",
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.12)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#e8e8e8")
    cell.set_edgecolor("black")
    cell.set_height(0.03)

plt.tight_layout()
plt.show()


