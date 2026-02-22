"""
Bootstrap confidence intervals for network metrics in extreme vs normal periods.

Requires Results/volatility_traces/regime_vol.csv and Results/networks.
Outputs Results/event_tables/*.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm

VOL_FILE = "Results/volatility_traces/regime_vol.csv"   
METRICS_DIR = "Results/networks"           
OUT_DATA_DIR = "Results/event_tables"
VOL_THRESHOLD = 0.4
BOOTSTRAP_ITERS = 5000
RANDOM_SEED = 12345
N_EVENTS = 3   

os.makedirs(OUT_DATA_DIR, exist_ok=True)

METRICS = [
    "NumNodes","NumEdges","Density","GiantComponentSizePct","AvgClusteringCoeff",
    "GlobalClustering","Efficiency","AvgShortestPathLength","Diameter",
    "DegreeCentralization","BetweennessCentralization","Modularity",
    "CommunitySizeEntropy","NumCommunities","Assortativity","ScaleFreeAlpha"
]

def find_blocks(dates, mask):
    """
    dates: pd.DatetimeIndex or array-like of datelike
    mask: boolean array-like (True = extreme)
    returns list of dicts: {"start": Timestamp, "end": Timestamp, "idx_start": i0, "idx_end": i1, "length": L}
    inclusive indices [idx_start, idx_end]
    """
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

vol_df = pd.read_csv(VOL_FILE, parse_dates=["Date"])
vol_df = vol_df.sort_values("Date").reset_index(drop=True)
dates = vol_df["Date"].values
mask_extreme = vol_df["Volatility"].values > VOL_THRESHOLD
ext_blocks = find_blocks(vol_df["Date"].values, mask_extreme)
norm_blocks = find_blocks(vol_df["Date"].values, ~mask_extreme)

for b in ext_blocks:
    i0, i1 = b["idx_start"], b["idx_end"]
    b["max_vol"] = vol_df.loc[i0:i1, "Volatility"].max()
    b["mean_vol"] = vol_df.loc[i0:i1, "Volatility"].mean()

matched_pairs = []
norm_index_map = {(b["idx_start"], b["idx_end"]): b for b in norm_blocks}
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
    print("No extreme blocks have an immediately preceding normal block.")
    raise SystemExit(1)

matched_pairs = sorted(matched_pairs, key=lambda x: (-x["ext"]["length"], -x["ext"]["mean_vol"]))

metric_files = sorted(glob.glob(os.path.join(METRICS_DIR, "*_metrics.csv")))
if not metric_files:
    raise FileNotFoundError(f"No metric files found in {METRICS_DIR}")

dfs = []
for f in metric_files:
    df = pd.read_csv(f, parse_dates=["Date"])
    for m in METRICS:
        if m not in df.columns:
            df[m] = np.nan
    dfs.append(df[["Date"] + METRICS])

metrics_df = pd.concat(dfs, ignore_index=True)
metrics_df = metrics_df.drop_duplicates(subset=["Date"])
metrics_df = metrics_df.sort_values("Date").reset_index(drop=True)
metrics_df["Date"] = pd.to_datetime(metrics_df["Date"])

rng = np.random.default_rng(RANDOM_SEED)

def bootstrap_period_ci(values_df, metrics, n_iter=5000, alpha=0.05):
    obs_map = {}
    ci_map = {}
    samples_map = {}
    n_days = len(values_df)
    if n_days == 0:
        for m in metrics:
            obs_map[m] = np.nan
            ci_map[m] = (np.nan, np.nan, np.nan)
            samples_map[m] = np.array([])
        return obs_map, ci_map, samples_map

    for m in metrics:
        arr = values_df[m].dropna().values
        if arr.size == 0:
            obs_map[m] = np.nan
            ci_map[m] = (np.nan, np.nan, np.nan)
            samples_map[m] = np.array([])
            continue
        obs = np.mean(arr)
        obs_map[m] = obs

        boot_means = np.empty(n_iter)
        for i in range(n_iter):
            idx = rng.integers(0, arr.size, arr.size)
            boot_means[i] = arr[idx].mean()
        lo = np.percentile(boot_means, 100 * (alpha / 2))
        med = np.percentile(boot_means, 50)
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
        ci_map[m] = (lo, med, hi)
        samples_map[m] = boot_means
    return obs_map, ci_map, samples_map

#========================================================================================================
#Main Loop for Events
#========================================================================================================

for i, pair in enumerate(matched_pairs[:N_EVENTS], start=1):
    ext = pair["ext"]
    norm = pair["norm"]
    ext_dates = vol_df.loc[ext["idx_start"]:ext["idx_end"], "Date"].dt.normalize()
    norm_dates = vol_df.loc[norm["idx_start"]:norm["idx_end"], "Date"].dt.normalize()

    ext_rows = metrics_df[metrics_df["Date"].isin(ext_dates)]
    norm_rows = metrics_df[metrics_df["Date"].isin(norm_dates)]

    obs_ext, ci_ext, samples_ext = bootstrap_period_ci(ext_rows, METRICS, n_iter=BOOTSTRAP_ITERS)
    obs_norm, ci_norm, samples_norm = bootstrap_period_ci(norm_rows, METRICS, n_iter=BOOTSTRAP_ITERS)

    base_label = f"event{i}_{pd.to_datetime(ext['start']).strftime('%Y%m%d')}_{pd.to_datetime(ext['end']).strftime('%Y%m%d')}"
    out_ci_path = os.path.join(OUT_DATA_DIR, base_label + "_ci.csv")
    out_obs_path = os.path.join(OUT_DATA_DIR, base_label + "_obs.csv")

    ci_rows = []
    for m in METRICS:
        lo_e, me_e, hi_e = ci_ext.get(m, (np.nan, np.nan, np.nan))
        lo_n, me_n, hi_n = ci_norm.get(m, (np.nan, np.nan, np.nan))
        ci_rows.append({
            "Metric": m,
            "Ext_lo": lo_e, "Ext_med": me_e, "Ext_hi": hi_e,
            "Norm_lo": lo_n, "Norm_med": me_n, "Norm_hi": hi_n
        })
    pd.DataFrame(ci_rows).to_csv(out_ci_path, index=False)

    obs_rows = []
    for m in METRICS:
        obs_rows.append({
            "Metric": m,
            "Ext_obs": obs_ext.get(m, np.nan),
            "Norm_obs": obs_norm.get(m, np.nan)
        })
    pd.DataFrame(obs_rows).to_csv(out_obs_path, index=False)

#========================================================================================================
#Main Loop for Aggregate
#========================================================================================================

all_ext_dates = []
all_norm_dates = []
for b in ext_blocks:
    all_ext_dates.extend(vol_df.loc[b["idx_start"]:b["idx_end"], "Date"].dt.normalize().tolist())
for b in norm_blocks:
    all_norm_dates.extend(vol_df.loc[b["idx_start"]:b["idx_end"], "Date"].dt.normalize().tolist())

all_ext_dates = pd.to_datetime(sorted(set(all_ext_dates)))
all_norm_dates = pd.to_datetime(sorted(set(all_norm_dates)))

ext_all_df = metrics_df[metrics_df["Date"].isin(all_ext_dates)]
norm_all_df = metrics_df[metrics_df["Date"].isin(all_norm_dates)]

agg_obs_ext, agg_ci_ext, _ = bootstrap_period_ci(ext_all_df, METRICS, n_iter=BOOTSTRAP_ITERS)
agg_obs_norm, agg_ci_norm, _ = bootstrap_period_ci(norm_all_df, METRICS, n_iter=BOOTSTRAP_ITERS)

agg_ci_rows = []
for m in METRICS:
    lo_e, me_e, hi_e = agg_ci_ext.get(m, (np.nan, np.nan, np.nan))
    lo_n, me_n, hi_n = agg_ci_norm.get(m, (np.nan, np.nan, np.nan))
    agg_ci_rows.append({
        "Metric": m,
        "Ext_lo": lo_e, "Ext_med": me_e, "Ext_hi": hi_e,
        "Norm_lo": lo_n, "Norm_med": me_n, "Norm_hi": hi_n
    })
pd.DataFrame(agg_ci_rows).to_csv(os.path.join(OUT_DATA_DIR, "aggregate_ci.csv"), index=False)

agg_obs_rows = []
for m in METRICS:
    agg_obs_rows.append({
        "Metric": m,
        "Ext_obs": agg_obs_ext.get(m, np.nan),
        "Norm_obs": agg_obs_norm.get(m, np.nan)
    })
pd.DataFrame(agg_obs_rows).to_csv(os.path.join(OUT_DATA_DIR, "aggregate_obs.csv"), index=False)
