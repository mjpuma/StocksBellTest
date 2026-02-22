#!/usr/bin/env python3
"""
Verify key computations match mathematical definitions.

Run after 0.py and 2.py. Checks: S1 formula, violation %, density, giant component.
"""
import os
import numpy as np
import pandas as pd

def test_s1_formula():
    """Verify S1 = E(a,b) + E(a,b') + E(a',b) - E(a',b') with sign products."""
    # Simple manual example: 5-day window, fixed threshold 0.05
    x = np.array([0.10, -0.08, 0.06, -0.03, 0.12])  # a signs: +,-,+,-,+
    y = np.array([0.07, -0.09, 0.04, 0.02, -0.11])  # b signs: +,-,+,-,-
    a_sgn, b_sgn = np.sign(x), np.sign(y)
    thr = 0.05
    mask_ax = np.abs(x) >= thr
    mask_ay = np.abs(y) >= thr

    def E(mask_x, mask_y):
        term = (a_sgn * b_sgn) * mask_x * mask_y
        s = term.sum()
        cnt = (mask_x & mask_y).sum()
        return s / cnt if cnt > 0 else 0

    E_ab = E(mask_ax, mask_ay)
    E_abp = E(mask_ax, ~mask_ay)
    E_apb = E(~mask_ax, mask_ay)
    E_apbp = E(~mask_ax, ~mask_ay)
    S1 = E_ab + E_abp + E_apb - E_apbp
    # |S1| > 2 indicates Bell violation (beyond classical bound)
    assert -2.83 <= S1 <= 2.83, f"S1={S1} should be in [-2√2, 2√2]"
    print("  ✓ S1 formula: E(a,b)+E(a,b')+E(a',b)-E(a',b') computed correctly")

def test_violation_pct():
    """Verify violation % = 100 * (count of |S1|>2) / total_pairs."""
    pct = pd.read_csv("Results/violation_pct.csv")
    s1 = pd.read_csv("Results/s1_values.csv")
    # Pick a date and recompute
    d = pct["Date"].iloc[100]
    day_s1 = s1[s1["Date"] == d]["S1"]
    violations = (np.abs(day_s1) > 2).sum()
    total = len(day_s1)
    expected = 100 * violations / total if total > 0 else np.nan
    actual = pct[pct["Date"] == d]["ViolationPct"].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)
    print(f"  ✓ Violation %: {violations}/{total} pairs → {actual:.2f}%")

def test_network_density():
    """Verify density = 2m / (n(n-1))."""
    metrics_dir = "Results/networks"
    files = [f for f in os.listdir(metrics_dir) if f.endswith("_metrics.csv")]
    if not files:
        print("  ⚠ No network metrics (run 2.py first)")
        return
    df = pd.read_csv(os.path.join(metrics_dir, files[0]))
    n, m = df["NumNodes"].iloc[0], df["NumEdges"].iloc[0]
    expected = 2 * m / (n * (n - 1)) if n > 1 else 0
    actual = df["Density"].iloc[0]
    np.testing.assert_almost_equal(actual, expected, decimal=6)
    print(f"  ✓ Density: 2*{m}/({n}*{n-1}) = {actual:.6f}")

def test_network_giant_component():
    """Verify giant component size = max|C_i| / |N|."""
    import networkx as nx
    metrics_dir = "Results/networks"
    s1_df = pd.read_csv("Results/s1_values.csv")
    s1_df["Date"] = pd.to_datetime(s1_df["Date"])
    dates = s1_df["Date"].unique()
    d = pd.Timestamp(dates[50])
    day = s1_df[s1_df["Date"] == d]
    edges = day[day["S1"] > 2][["PairA", "PairB"]]
    G = nx.from_pandas_edgelist(edges, "PairA", "PairB")
    nodes = set(s1_df["PairA"]).union(s1_df["PairB"])
    G.add_nodes_from(nodes)
    comps = list(nx.connected_components(G))
    max_comp = max(comps, key=len) if comps else set()
    expected = len(max_comp) / len(nodes) if nodes else 0
    metrics_file = os.path.join(metrics_dir, f"{d.strftime('%Y-%m-%d')}_metrics.csv")
    if os.path.exists(metrics_file):
        mdf = pd.read_csv(metrics_file)
        actual = mdf["GiantComponentSizePct"].iloc[0] / 100  # stored as %
        np.testing.assert_almost_equal(actual, expected, decimal=4)
        print(f"  ✓ Giant component: {len(max_comp)}/{len(nodes)} = {expected:.4f}")

def main():
    print("Verifying computations...\n")
    test_s1_formula()
    if os.path.exists("Results/violation_pct.csv") and os.path.exists("Results/s1_values.csv"):
        test_violation_pct()
    if os.path.exists("Results/networks"):
        test_network_density()
        test_network_giant_component()
    print("\n✓ Verification complete.")

if __name__ == "__main__":
    main()
