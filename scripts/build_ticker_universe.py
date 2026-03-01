#!/usr/bin/env python3
"""
Build agriculture ticker universe from MOO ETF constituents.

Fetches MOO top holdings from yfinance, supplements with curated MOO US list
(since yfinance returns only top 10), fetches sector/industry for each ticker,
maps to objective groups via config, and outputs yfinance_tickers.csv.

Usage:
    python scripts/build_ticker_universe.py [--scope moo|moo_plus|global] [--min-cap 0] [--output yfinance_tickers.csv]
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "yfinance_tickers.csv")

# MOO US-listed constituents (curated; yfinance returns only top 10)
# Source: VanEck Agribusiness ETF, stockanalysis.com
MOO_US_TICKERS = [
    "DE", "ZTS", "CTVA", "NTR", "ADM", "TSN", "CF", "BG", "CNH", "DAR",
    "TTC", "ELAN", "MOS", "AGCO", "CALM",
]

# Ag-adjacent tickers for moo_plus scope (from current list, ag-focused)
MOO_PLUS_EXTRA = [
    "FMC", "AVD", "ICL", "LW", "SYY", "LAND", "FPI", "BYND",
    "IPI", "SMG", "UAN", "HRL", "CPB", "GIS", "CAG",
    "UNFI", "TSCO", "TITN", "ALG", "STKL", "OTLY",
]

# MOO international constituents (yfinance-compatible with exchange suffix)
# Source: VanEck Agribusiness ETF holdings
MOO_INTERNATIONAL = [
    "BAYN.DE",   # Bayer AG (Germany)
    "YAR.OL",    # Yara International (Norway)
    "MOWI.OL",   # Mowi ASA (Norway)
    "6326.T",    # Kubota (Japan)
    "SALM.OL",   # SalMar ASA (Norway)
    "SDF.DE",    # K+S AG (Germany)
]

# MNC cap threshold (market cap >= this -> category MNC)
MNC_CAP_THRESHOLD = 50e9  # $50B


def load_moo_from_yfinance():
    """Fetch MOO top holdings from yfinance (top 10 only)."""
    try:
        moo = yf.Ticker("MOO")
        fd = moo.funds_data
        if hasattr(fd, "top_holdings") and fd.top_holdings is not None:
            tickers = list(fd.top_holdings.index)
            out = []
            for t in tickers:
                if "." in t:
                    exch = t.split(".")[1]
                    if exch == "TO":
                        out.append(t.split(".")[0])  # NTR.TO -> NTR (NYSE)
                    else:
                        out.append(t)  # Keep BAYN.DE, MOWI.OL, 6326.T, etc.
                else:
                    out.append(t)
            return list(dict.fromkeys(out))
    except Exception:
        pass
    return []


def load_mapping(config_dir):
    """Load industry->group, ticker override, and category override mappings."""
    industry_path = os.path.join(config_dir, "industry_to_group.csv")
    override_path = os.path.join(config_dir, "ticker_to_group_override.csv")
    category_path = os.path.join(config_dir, "ticker_to_category_override.csv")

    industry_map = {}
    if os.path.exists(industry_path):
        df = pd.read_csv(industry_path)
        for _, row in df.iterrows():
            industry_map[str(row["industry_key"]).strip()] = str(row["group"]).strip()

    override_map = {}
    if os.path.exists(override_path):
        odf = pd.read_csv(override_path)
        for _, row in odf.iterrows():
            override_map[str(row["ticker"]).strip()] = str(row["group"]).strip()

    category_override = {}
    if os.path.exists(category_path):
        cdf = pd.read_csv(category_path)
        for _, row in cdf.iterrows():
            category_override[str(row["ticker"]).strip()] = str(row["category"]).strip()

    return industry_map, override_map, category_override


def get_group(ticker, industry, industry_map, override_map):
    """Resolve group: override first, then industry mapping."""
    if ticker in override_map:
        return override_map[ticker]
    if industry and industry in industry_map:
        return industry_map[industry]
    # Partial match for industry
    if industry:
        for key, group in industry_map.items():
            if key.lower() in (industry or "").lower():
                return group
    return "Other"


def fmt_market_cap(val):
    """Format market cap for display (e.g. 1.5B, 400M)."""
    if pd.isna(val) or val is None or val <= 0:
        return ""
    if val >= 1e12:
        return f"{val/1e12:.1f}T"
    if val >= 1e9:
        return f"{val/1e9:.1f}B"
    if val >= 1e6:
        return f"{val/1e6:.0f}M"
    return f"{val:.0f}"


def get_category(ticker, cap, category_override):
    """MNC if international listing, cap >= threshold, or explicit override."""
    if ticker in category_override:
        return category_override[ticker]
    if "." in ticker and ticker.split(".")[1] in ("DE", "OL", "T", "TO", "HK", "L"):
        return "MNC"
    if cap and cap >= MNC_CAP_THRESHOLD:
        return "MNC"
    return "Pure Ag"


def build_universe(scope="moo_plus", min_cap=0, output_path=None):
    """Build ticker universe and write CSV."""
    industry_map, override_map, category_override = load_mapping(CONFIG_DIR)

    if scope == "moo":
        tickers = list(dict.fromkeys(MOO_US_TICKERS))
    elif scope == "global":
        tickers = list(dict.fromkeys(
            MOO_US_TICKERS + MOO_PLUS_EXTRA + MOO_INTERNATIONAL
        ))
    else:
        tickers = list(dict.fromkeys(MOO_US_TICKERS + MOO_PLUS_EXTRA))

    # Merge with any from yfinance MOO top holdings (top 10)
    yf_tickers = load_moo_from_yfinance()
    seen = set(tickers)
    for t in yf_tickers:
        base = t.split(".")[0] if "." in t else t
        if t not in seen and base not in seen:
            tickers.append(t)
            seen.add(t)
            seen.add(base)

    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            name = info.get("shortName") or info.get("longName") or ticker
            sector = info.get("sector") or ""
            industry = info.get("industry") or ""
            cap = info.get("marketCap")
            if cap is None:
                cap = 0

            if min_cap and cap and cap < min_cap:
                continue

            group = get_group(ticker, industry, industry_map, override_map)
            category = get_category(ticker, cap, category_override)
            cap_str = fmt_market_cap(cap) if cap else ""

            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "group": group,
                "category": category,
                "market_cap_approx": cap_str,
            })
        except Exception as e:
            print(f"Warning: skip {ticker}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    out = output_path or DEFAULT_OUTPUT
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} tickers to {out} (MNC: {(df['category']=='MNC').sum()}, Pure Ag: {(df['category']=='Pure Ag').sum()})")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build agriculture ticker universe from MOO")
    parser.add_argument("--scope", choices=["moo", "moo_plus", "global"], default="global",
                        help="moo = MOO US only (~15); moo_plus = MOO + ag-adjacent (~37); global = + international (~43)")
    parser.add_argument("--min-cap", type=float, default=0,
                        help="Minimum market cap (exclude smaller)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    build_universe(scope=args.scope, min_cap=args.min_cap, output_path=args.output)


if __name__ == "__main__":
    main()
