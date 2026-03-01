"""
Permutation tests for network metrics: extreme vs normal periods.

Requires Results/volatility_traces/regime_vol.csv and Results/networks.
Outputs Results/event_tables/*.csv and Results/event_tables/*.tex
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

os.makedirs("Results/event_tables", exist_ok=True)

VOL_FILE = "Results/volatility_traces/regime_vol.csv"
METRICS_DIR = "Results/networks"
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
        return "---"
    try:
        v = float(x)
        if v < 1e-10:
            return r"$<10^{-10}$"
        return np.format_float_scientific(v, precision=precision)
    except Exception:
        return str(x)


def export_permtest_latex(res_df, out_tex, caption, label, metric_map=None):
    """Export permutation results to LaTeX (condensed: Metric, % Change, p, q)."""
    df = res_df.copy()
    if metric_map:
        df["Metric"] = df["Metric"].map(lambda m: metric_map.get(m, m))
    df["% Change"] = df["Percent_change"].map(lambda x: f"{x:.1f}\\%" if pd.notna(x) else "---")
    df["p"] = df["p_value"].apply(safe_fmt_scientific)
    df["q"] = df["q_value"].apply(safe_fmt_scientific)
    df = df[["Metric", "% Change", "p", "q"]]
    tex_str = df.to_latex(index=False, escape=False, column_format="lccc")
    with open(out_tex, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(tex_str)
        f.write("\\end{table}\n")
    print(f"Saved {out_tex}")


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

vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"]).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
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
    "AvgClusteringCoeff": "Avg. Clustering Coef.",
    "GlobalClustering": "Global Clustering",
    "Efficiency": "Efficiency",
    "AvgShortestPathLength": "Avg. Path Length",
    "Diameter": "Diameter",
    "DegreeCentralization": "Degree Centralization",
    "BetweennessCentralization": "Betweenness Centralization",
    "Modularity": "Modularity",
    "CommunitySizeEntropy": "Community Size Entropy",
    "NumCommunities": "Number of Communities",
    "Assortativity": "Assortativity",
    "ScaleFreeAlpha": "Scale-Free Alpha"
}

def export_event_permtest_latex(event_id, ext, norm):
    """Export event permutation results to LaTeX."""
    FILE = f"Results/event_tables/permtest_event{event_id}.csv"
    OUT_TEX = f"Results/event_tables/permtest_event{event_id}.tex"
    df = pd.read_csv(FILE)
    # Drop NumNodes (no variation)
    df = df[df["Metric"] != "NumNodes"]
    caption = (
        f"Permutation test Event {event_id}: "
        f"{pd.Timestamp(ext['start']).date()}--{pd.Timestamp(ext['end']).date()} vs "
        f"{pd.Timestamp(norm['start']).date()}--{pd.Timestamp(norm['end']).date()}."
    )
    export_permtest_latex(df, OUT_TEX, caption, f"tab:permtest_event{event_id}", METRIC_NAME_MAP)

#========================================================================================================
#Main Loop
#========================================================================================================

if __name__ == "__main__":
    import sys
    export_only = "--export-only" in sys.argv
    if export_only:
        # Export LaTeX from existing CSVs (skip permutation tests)
        for i in range(1, N_EVENTS + 1):
            csv_path = os.path.join(OUT_DATA_DIR, f"permtest_event{i}.csv")
            if os.path.exists(csv_path):
                ext = matched_pairs[i - 1]["ext"]
                norm = matched_pairs[i - 1]["norm"]
                export_event_permtest_latex(i, ext, norm)
        agg_path = os.path.join(OUT_DATA_DIR, "permtest_aggregate.csv")
        if os.path.exists(agg_path):
            agg_df = pd.read_csv(agg_path)
            export_permtest_latex(
                agg_df, os.path.join(OUT_DATA_DIR, "permtest_aggregate.tex"),
                "Permutation test aggregate: all extreme vs all normal days.",
                "tab:permtest_aggregate", METRIC_NAME_MAP
            )
            # Condensed Table 1 for main text
            TABLE1_METRICS = ["Density", "ScaleFreeAlpha", "AvgClusteringCoeff", "CommunitySizeEntropy"]
            TABLE1_NAMES = {"Density": "Density", "ScaleFreeAlpha": "Scale-Free Alpha",
                           "AvgClusteringCoeff": "Avg Clustering", "CommunitySizeEntropy": "Community Entropy"}
            t1 = agg_df[agg_df["Metric"].isin(TABLE1_METRICS)].copy()
            t1["Metric"] = t1["Metric"].map(TABLE1_NAMES)
            t1["Extreme"] = t1["Obs_ext"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            t1["Normal"] = t1["Obs_norm"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            t1["% Change"] = t1["Percent_change"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
            t1["p"] = t1["p_value"].apply(safe_fmt_scientific)
            t1_main = t1[["Metric", "Extreme", "Normal", "% Change", "p"]]
            t1_main.to_csv(os.path.join(OUT_DATA_DIR, "Table01_main.csv"), index=False, encoding="utf-8-sig")
            t1_tex = t1_main.to_latex(index=False, escape=False, column_format="lcccr")
            with open(os.path.join(OUT_DATA_DIR, "Table01_main.tex"), "w") as f:
                f.write("\\begin{table}[ht]\n\\centering\n")
                f.write("\\caption{Network metrics between extreme and normal volatility periods. Columns report means and percent change; p values are from permutation tests (100,000 permutations).}\n")
                f.write("\\label{tab:permtest_main}\n")
                f.write(t1_tex)
                f.write("\\end{table}\n")
            print("Saved Table01_main.csv and Table01_main.tex")
        print("LaTeX export done.")
        sys.exit(0)

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

        export_event_permtest_latex(i, ext, norm)

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

    agg_tex = os.path.join(OUT_DATA_DIR, "permtest_aggregate.tex")
    export_permtest_latex(
        agg_df, agg_tex,
        "Permutation test aggregate: all extreme vs all normal days.",
        "tab:permtest_aggregate",
        METRIC_NAME_MAP
    )

    # Condensed Table 1 for main text (4 metrics, aggregate only)
    TABLE1_METRICS = ["Density", "ScaleFreeAlpha", "AvgClusteringCoeff", "CommunitySizeEntropy"]
    TABLE1_NAMES = {"Density": "Density", "ScaleFreeAlpha": "Scale-Free Alpha",
                   "AvgClusteringCoeff": "Avg Clustering", "CommunitySizeEntropy": "Community Entropy"}
    t1 = agg_df[agg_df["Metric"].isin(TABLE1_METRICS)].copy()
    t1["Metric"] = t1["Metric"].map(TABLE1_NAMES)
    t1["Extreme"] = t1["Obs_ext"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    t1["Normal"] = t1["Obs_norm"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    t1["% Change"] = t1["Percent_change"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    t1["p"] = t1["p_value"].apply(safe_fmt_scientific)
    t1_main = t1[["Metric", "Extreme", "Normal", "% Change", "p"]]
    t1_main.to_csv(os.path.join(OUT_DATA_DIR, "Table01_main.csv"), index=False, encoding="utf-8-sig")
    t1_tex = t1_main.to_latex(index=False, escape=False, column_format="lcccr")
    with open(os.path.join(OUT_DATA_DIR, "Table01_main.tex"), "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Network metrics between extreme and normal volatility periods. Columns report means and percent change; p values are from permutation tests (100,000 permutations).}\n")
        f.write("\\label{tab:permtest_main}\n")
        f.write(t1_tex)
        f.write("\\end{table}\n")
    print(f"Saved {OUT_DATA_DIR}/Table01_main.csv and Table01_main.tex")

    print(f"Aggregate comparison done — saved {agg_csv}")


