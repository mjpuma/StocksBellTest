"""
Volatility methods (rolling, GARCH, regime-switching) for S&P GSCI.

Requires Results/violation_pct.csv from 0.py.
Outputs Results/volatility_traces/*.csv and Figures/Established_Methods_&_S1.*
"""

import numpy as np
import pandas as pd
import os
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.io as pio

pio.renderers.default = "browser"

#========================================================================================================
#Data
#========================================================================================================

ticker = "^SPGSCI"
data = yf.download(ticker, start="2000-01-01", end="2025-10-14", progress=False)
data = data[["Close"]].rename(columns={"Close": "price"}).dropna()
data["returns"] = data["price"].pct_change()
data = data.dropna().copy()

#========================================================================================================
#Rolling Volatility
#========================================================================================================

window = 20
data["rolling_vol"] = data["returns"].rolling(window).std() * np.sqrt(252)

#========================================================================================================
#Garch
#========================================================================================================

returns = (data["returns"].dropna() * 100)
am = arch_model(returns, vol="Garch", p=1, q=1)
res = am.fit(disp="off")
data.loc[returns.index, "garch_vol"] = np.sqrt(res.conditional_volatility ** 2) / 100 * np.sqrt(252)

#========================================================================================================
#Markov Regression
#========================================================================================================

try:
    mod = MarkovRegression(data["returns"], k_regimes=2, trend="c", switching_variance=True)
    res_mr = mod.fit(disp=False)
    data["regime_prob"] = res_mr.smoothed_marginal_probabilities[1]
    data["regime_vol"] = data["regime_prob"] * data["returns"].rolling(20).std() * np.sqrt(252)
except ValueError as e:
    print("Markov model failed — using fallback method:", e)
    data["regime_prob"] = np.nan
    data["regime_vol"] = np.nan

#========================================================================================================
#Save Results
#========================================================================================================

os.makedirs("Results/volatility_traces", exist_ok=True)

export_cols = ["rolling_vol", "garch_vol", "regime_vol"]
for col in export_cols:
    output_path = os.path.join("Results/volatility_traces", f"{col}.csv")
    out_df = data[[col]].dropna().reset_index()
    out_df.rename(columns={"index": "Date", col: "Volatility"}, inplace=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {col} → {output_path}")

#========================================================================================================
#Plot
#========================================================================================================
violation_path = "Results/violation_pct.csv"
violation_df = pd.read_csv(violation_path)
violation_df["Date"] = pd.to_datetime(violation_df["Date"])

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    row_heights=[0.6, 0.4],
    subplot_titles=("(A)", "(B)")
)

fig.add_trace(go.Scatter(x=data.index, y=data["rolling_vol"], mode="lines",
                         name="Rolling Realized Vol (20d)", line=dict(color="darkturquoise")), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["garch_vol"], mode="lines",
                         name="GARCH(1,1) Volatility", line=dict(color="lightskyblue")), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["regime_vol"], mode="lines",
                         name="Regime-Switching Volatility", line=dict(color="lightsalmon")), row=1, col=1)
fig.add_trace(go.Scatter(
    x=violation_df["Date"], y=violation_df["ViolationPct"],
    mode="lines", name="Violation % (|S₁|>2)",
    line=dict(color="mediumvioletred")
), row=2, col=1)

vol_series = data["rolling_vol"].dropna()
p95, p80, p20 = np.percentile(vol_series, [95, 80, 20])
mu, sigma = vol_series.mean(), vol_series.std()
std_extreme, std_high = mu + 2*sigma, mu + sigma
std_low, std_extreme_low = mu - sigma, mu - 2*sigma
abs_extreme, abs_high, abs_normal = 0.4, 0.25, 0.15

thresholds = {
    "Percentiles": [
        (p95, "slategrey", "dot", "P95", 1.05),
        (p80, "slategrey", "dash", "P80", 1.05),
        (p20, "slategrey", "dash", "P20", 1.05)
    ],
    "StdDev": [
        (std_extreme, "dimgray", "dot", "μ+2σ", 1.06),
        (std_high, "dimgray", "dash", "μ+σ", 1.06),
        (std_low, "dimgray", "dash", "μ-σ", 1.06),
        (std_extreme_low, "dimgray", "dot", "μ-2σ", 1.06)
    ],
    "Absolute": [
        (abs_extreme, "darkslategray", "dot", ">40%", 1.14),
        (abs_high, "darkslategray", "dash", "25–40%", 1.14),
        (abs_normal, "darkslategray", "dash", "15–25%", 1.14)
    ]
}

buttons = []
for set_name, thresh_list in thresholds.items():
    shapes, annotations = [], []
    for y, color, dash, text, x_side in thresh_list:
        shapes.append(dict(type="line", xref="x", yref="y", x0=data.index.min(), x1=data.index.max(),
                           y0=y, y1=y, line=dict(color=color, dash=dash)))
        annotations.append(dict(x=x_side, y=y, xref="paper", yref="y",
                                text=text, showarrow=False, align="left"))
    buttons.append(dict(label=set_name, method="relayout", args=[{"shapes": shapes, "annotations": annotations}]))

all_shapes, all_annotations = [], []
for thresh_list in thresholds.values():
    for y, color, dash, text, x_side in thresh_list:
        all_shapes.append(dict(type="line", xref="x", yref="y", x0=data.index.min(), x1=data.index.max(),
                               y0=y, y1=y, line=dict(color=color, dash=dash)))
        all_annotations.append(dict(x=x_side, y=y, xref="paper", yref="y",
                                    text=text, showarrow=False, align="left"))
buttons.append(dict(label="All", method="relayout",
                    args=[{"shapes": all_shapes, "annotations": all_annotations}]))

fig.update_layout(
    height=1200,
    width=1800,
    font=dict(size=16),
    updatemenus=[dict(
        type="dropdown", x=0.99, y=1.05, xanchor="right", yanchor="top",
        direction="down", buttons=buttons
    )],
    yaxis_title="Annualized Volatility",
    yaxis2_title="Violation % (|S₁| > 2)",
    xaxis2_title="Date",
    showlegend=True
)

fig.show()
fig.write_html("Figures/Established_Methods_&_S1.html")

plt.rcParams.update({"font.size": 16})
fig_m, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={"height_ratios": [0.6, 0.4]})

ax1.text(0.02, 0.98, "(A)", transform=ax1.transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
ax1.plot(data.index, data["rolling_vol"], color="darkturquoise", label="Rolling Realized Vol (20d)")
ax1.plot(data.index, data["garch_vol"], color="lightskyblue", label="GARCH(1,1) Volatility")
ax1.plot(data.index, data["regime_vol"], color="lightsalmon", label="Regime-Switching Volatility")
ax1.set_ylabel("Annualized Volatility")
label_offset = {
    "Percentiles": 0.002,
    "StdDev": 0.002,
    "Absolute": 0.002,
}

g40_entry = None
for group, thresh_list in thresholds.items():
    for y, color, style, label, _ in thresh_list:
        if label == ">40%":
            g40_entry = (y, color, style, label)
            break
    if g40_entry:
        break

if g40_entry:
    y, color, style, label = g40_entry
    mstyle = {"dot": ":", "dash": "--", "solid": "-"}.get(style, "--")
    ax1.axhline(y, color=color, linestyle=mstyle, alpha=0.9, linewidth=1.2, zorder=3)

    ax1.annotate(
        text=label,
        xy=(data.index.max(), y),
        xycoords="data",
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=16,
        color=color,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
        zorder=4
    )


ax1.legend(loc="upper right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.4)

# X-axis: year only; y-axis tick count
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax2.text(0.02, 0.98, "(B)", transform=ax2.transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
ax2.plot(violation_df["Date"], violation_df["ViolationPct"],
         color="mediumvioletred", label="Violation Percent (|S1|>2)")
ax2.set_ylabel("Violation Percent (|S1| > 2)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("Figures/Fig01_volatility_and_violation.png", dpi=300)
plt.savefig("Figures/Fig01_volatility_and_violation.svg")
plt.close()