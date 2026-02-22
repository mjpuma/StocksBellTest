"""
Permutation tests for network metrics: extreme vs normal periods.

Requires Results/volatility_traces/regime_vol.csv and Results/networks.
Outputs Figures/event_tables/*.png and Results/event_tables/*.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

os.makedirs("Results/event_tables", exist_ok=True)
os.makedirs("Figures/event_tables", exist_ok=True)

VOL_FILE = "Results/volatility_traces/regime_vol.csv"
METRICS_DIR = "Results/networks"
OUT_TABLE_DIR = "Figures/event_tables"
OUT_DATA_DIR = "Results/event_tables"
SIG_THRESHOLD = 0.05
VOL_THRESHOLD = 0.4
N_EVENTS = 3  

N_PERM = 100000              

def permutation_test(x, y, n_perm=None, seed=None, batch=2000, show_progress=False):

    n_perm=N_PERM

    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan

    obs = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n_x = x.size

    rng = np.random.default_rng(seed)

    count = 0
    done = 0

    pbar = tqdm(total=n_perm, disable=not show_progress, desc="perms", leave=False)
    try:
        while done < n_perm:
            this_batch = min(batch, n_perm - done)
            for _ in range(this_batch):
                perm = rng.permutation(pooled)   
                d = perm[:n_x].mean() - perm[n_x:].mean()
                if abs(d) >= abs(obs):
                    count += 1
            done += this_batch
            pbar.update(this_batch)
        pbar.close()
    except KeyboardInterrupt:
        pbar.close()
        print("Permutation loop interrupted by user. Returning current p estimate.")
        if done == 0:
            return obs, np.nan
        return obs, float((count + 1) / (done + 1))

    p_val = float((count + 1) / (n_perm + 1))
    return obs, p_val

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else np.nan

def safe_fmt_scientific(x, precision=2):
    if pd.isna(x):
        return "N/A"
    try:
        return np.format_float_scientific(float(x), precision=precision)
    except Exception:
        return str(x)

def plot_table(df_numeric, output_file, title, color_col="p_value", include_q=False):
    df_num = df_numeric.copy()
    df_disp = df_num.copy()

    for col in ["Delta_obs", "Obs_ext", "Obs_norm", "Cohens_d", "Absolute_change"]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    if "Percent_change" in df_disp.columns:
        df_disp["Percent_change"] = df_disp["Percent_change"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

    if "p_value" in df_disp.columns:
        df_disp["p_value"] = df_disp["p_value"].map(lambda x: safe_fmt_scientific(x, precision=2))
    if "q_value" in df_disp.columns:
        df_disp["q_value"] = df_disp["q_value"].map(lambda x: safe_fmt_scientific(x, precision=2))

    if not include_q and "q_value" in df_disp.columns:
        df_disp = df_disp.drop(columns=["q_value"])

    cell_colors = []
    for _, row in df_num.iterrows():
        val = row.get(color_col, np.nan)
        if pd.isna(val):
            color = "#ffffff"
        elif float(val) < SIG_THRESHOLD:
            color = "#add8e6" 
        else:
            color = "#f9c4b0"
        cell_colors.append([color] * len(df_disp.columns))

    n_rows, n_cols = df_disp.shape
    fig_w = max(10, 1.4 * n_cols)
    fig_h = max(2, 0.4 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    fontsize = 10 if n_cols <= 8 else max(6, int(10 - (n_cols - 8) * 0.5))
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", fontsize=fontsize)
            cell.set_facecolor("#f0f0f0")
        else:
            if df_disp.columns[c] == "Delta_obs":
                cell.get_text().set_weight("bold")
        cell.set_edgecolor("black")

    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_file}")

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

vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
ext_blocks = find_blocks(vol_df["Date"].values, mask_extreme)
norm_blocks = find_blocks(vol_df["Date"].values, ~mask_extreme)

for b in ext_blocks:
    i0, i1 = b["idx_start"], b["idx_end"]
    b["max_vol"] = vol_df.loc[i0:i1, "Volatility"].max()
    b["mean_vol"] = vol_df.loc[i0:i1, "Volatility"].mean()

matched_pairs = []
for e in ext_blocks:
    e_start = e["idx_start"]
    preceding = None
    for nb in norm_blocks:
        if nb["idx_end"] == e_start - 1:
            preceding = nb
            break
    if preceding is not None:
        matched_pairs.append({"ext": e, "norm": preceding})

if len(matched_pairs) == 0:
    raise SystemExit("No extreme blocks have an immediately preceding normal block.")

matched_pairs = sorted(matched_pairs, key=lambda x: (-x["ext"]["length"], -x["ext"]["mean_vol"]))

def load_all_metrics():
    files = sorted(glob.glob(os.path.join(METRICS_DIR, "*_metrics.csv")))
    dfs = [pd.read_csv(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    return all_df.sort_values("Date")

METRIC_NAME_MAP = {
    "NumNodes": "Number of Nodes",
    "NumEdges": "Number of Edges",
    "Density": "Density",
    "GiantComponentSizePct": "Giant Component Size",
    "AvgClusteringCoeff": "Average Clustering \n Coefficient",
    "GlobalClustering": "Global Clustering",
    "Efficiency": "Efficiency",
    "AvgShortestPathLength": "Average Shortest \n Path Length",
    "Diameter": "Diameter",
    "DegreeCentralization": "Degree Centralization",
    "BetweennessCentralization": "Betweenness \n Centralization",
    "Modularity": "Modularity",
    "CommunitySizeEntropy": "Community Size Entropy",
    "NumCommunities": "Number of Communities",
    "Assortativity": "Assortativity",
    "ScaleFreeAlpha": "Scale-Free Alpha"
}

def fmt(x):
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (int, float, np.number)):
        return f"{x:.6f}"
    return str(x)

def plot_event_table_from_csv(event_id):
    FILE = f"Results/event_tables/permtest_event{event_id}.csv"
    OUTPUT = f"Figures/event_tables/permtest_event{event_id}_table.png"

    df = pd.read_csv(FILE)
    df = df.drop(df.index[0])

    df["Metric"] = df["Metric"].map(METRIC_NAME_MAP)

    df = df.rename(columns={
        "Obs_norm": "Normal",
        "Obs_ext": "Extreme",
        "Percent_change": "Percent Change (%)",
        "Cohens_d": "Cohen's d",
        "p_value": "p Value",
        "q_value": "q Value"
    })

    df = df[[
        "Metric",
        "Normal",
        "Extreme",
        "Percent Change (%)",
        "Cohen's d",
        "p Value",
        "q Value"
    ]]

    cols = list(df.columns)
    cell_text = []
    row_colors = []

    for _, row in df.iterrows():
        cell_text.append([
            row["Metric"],
            fmt(row["Normal"]),
            fmt(row["Extreme"]),
            fmt(row["Percent Change (%)"]),
            fmt(row["Cohen's d"]),
            fmt(row["p Value"]),
            fmt(row["q Value"])
        ])

        if not pd.isna(row["q Value"]) and row["q Value"] < SIG_THRESHOLD:
            row_colors.append(["#b3d9ff"] * len(cols))
        else:
            row_colors.append(["white"] * len(cols))

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(df) + 1.5))
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
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT}")

#========================================================================================================
#Main Loop
#========================================================================================================

if __name__ == "__main__":
    all_metrics = load_all_metrics()
    print(f"Loaded {len(all_metrics)} metric rows with columns: {list(all_metrics.columns)}")

    for i, pair in enumerate(matched_pairs[:N_EVENTS], start=1):
        ext, norm = pair["ext"], pair["norm"]
        ext_dates = all_metrics["Date"].between(ext["start"], ext["end"])
        norm_dates = all_metrics["Date"].between(norm["start"], norm["end"])

        extreme_df = all_metrics[ext_dates]
        normal_df = all_metrics[norm_dates]

        metrics = [c for c in all_metrics.columns if c not in ["Date"]]
        results = []

        for metric in tqdm(metrics, desc=f"Event {i} permutation test"):
            x = extreme_df[metric].dropna().values
            y = normal_df[metric].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            delta_obs, p_val = permutation_test(x, y, N_PERM)
            d_val = cohens_d(x, y)
            obs_ext, obs_norm = np.mean(x), np.mean(y)
            pct_change = 100 * (obs_ext - obs_norm) / abs(obs_norm) if obs_norm != 0 else np.nan
            abs_change = obs_ext - obs_norm
            results.append({
                "Metric": metric,
                "Obs_ext": obs_ext,
                "Obs_norm": obs_norm,
                "Delta_obs": delta_obs,
                "p_value": p_val,
                "Cohens_d": d_val,
                "Percent_change": pct_change,
                "Absolute_change": abs_change
            })

        if not results:
            print(f"Skipping event {i}: no valid metrics.")
            continue

        res_df = pd.DataFrame(results)
        reject, q_vals, _, _ = multipletests(res_df["p_value"].fillna(1.0), method="fdr_bh")
        res_df["q_value"] = q_vals

        out_csv = os.path.join(OUT_DATA_DIR, f"permtest_event{i}.csv")
        res_df.to_csv(out_csv, index=False)

        title = (
            f"Permutation Test — Event {i}: "
            f"{pd.Timestamp(ext['start']).date()}→{pd.Timestamp(ext['end']).date()} "
            f"vs {pd.Timestamp(norm['start']).date()}→{pd.Timestamp(norm['end']).date()}"
        )

        plot_event_table_from_csv(i)

        print(f"Event {i} done — saved {out_csv}")

    print("\n=== Running aggregate comparison across all days ===")
    all_ext_dates = vol_df.loc[mask_extreme, "Date"]
    all_norm_dates = vol_df.loc[~mask_extreme, "Date"]

    extreme_df = all_metrics[all_metrics["Date"].isin(all_ext_dates)]
    normal_df = all_metrics[all_metrics["Date"].isin(all_norm_dates)]

    metrics = [c for c in all_metrics.columns if c not in ["Date"]]
    agg_results = []

    for metric in tqdm(metrics, desc="Aggregate permutation test"):
        x = extreme_df[metric].dropna().values
        y = normal_df[metric].dropna().values
        if len(x) < 2 or len(y) < 2:
            continue
        delta_obs, p_val = permutation_test(x, y, N_PERM)
        d_val = cohens_d(x, y)
        obs_ext, obs_norm = np.mean(x), np.mean(y)
        pct_change = 100 * (obs_ext - obs_norm) / abs(obs_norm) if obs_norm != 0 else np.nan
        abs_change = obs_ext - obs_norm
        agg_results.append({
            "Metric": metric,
            "Obs_ext": obs_ext,
            "Obs_norm": obs_norm,
            "Delta_obs": delta_obs,
            "p_value": p_val,
            "Cohens_d": d_val,
            "Percent_change": pct_change,
            "Absolute_change": abs_change
        })

    agg_df = pd.DataFrame(agg_results)
    reject, q_vals, _, _ = multipletests(agg_df["p_value"].fillna(1.0), method="fdr_bh")
    agg_df["q_value"] = q_vals

    agg_csv = os.path.join(OUT_DATA_DIR, "permtest_aggregate.csv")
    agg_df.to_csv(agg_csv, index=False)

    agg_title = "Permutation Test — Aggregate: All Extreme vs All Normal Days"
    fig_p = os.path.join(OUT_TABLE_DIR, "aggregate_pvals.png")
    fig_q = os.path.join(OUT_TABLE_DIR, "aggregate_qvals.png")
    plot_table(agg_df, fig_p, agg_title + " (p-values)", color_col="p_value", include_q=False)
    plot_table(agg_df, fig_q, agg_title + " (q-values)", color_col="q_value", include_q=True)

    print(f"Aggregate comparison done — saved {agg_csv}")


