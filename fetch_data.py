#!/usr/bin/env python3
"""
Collect historical prices + light fundamentals, compute 3y/5y drawdowns,
and produce two outputs:
  1) data/prices.parquet         (full daily timeseries for all tickers)
  2) data/latest_snapshot.csv    (one row per ticker with screening columns)

Usage examples:
  python collect_data.py --tickers AAPL MSFT META NVDA --start 2014-01-01
  python collect_data.py --tickers-file tickers_sp500.txt --start 2010-01-01 \
                         --outdir data --apply-screen
"""

from __future__ import annotations
import argparse
import os
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# yfinance is lightweight and free; pip install yfinance
import yfinance as yf


# ------------------------------- Utils -------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Historical data collector (prices + fundamentals + drawdowns).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", nargs="+", help="Space-separated tickers.")
    g.add_argument("--tickers-file", type=str, help="Text file with one ticker per line.")
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD). Defaults to today.")
    p.add_argument("--outdir", type=str, default="data", help="Output directory.")
    p.add_argument("--window3y", type=int, default=252*3, help="Trading-day window for ~3y high.")
    p.add_argument("--window5y", type=int, default=252*5, help="Trading-day window for ~5y high.")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between tickers to be polite.")
    p.add_argument("--apply-screen", action="store_true",
                   help="Apply first-pass screen (drawdown<=-30% on 3y or 5y AND mcap>=1B).")
    return p.parse_args()


def load_tickers(args: argparse.Namespace) -> List[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers if t.strip()]
    with open(args.tickers_file, "r", encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def safe_fastinfo(tkr: yf.Ticker) -> Dict[str, Any]:
    """Pull a few fields; tolerate missing keys."""
    out = {"market_cap": np.nan, "beta": np.nan, "shares_outstanding": np.nan,
           "long_name": None, "sector": None}
    try:
        # fast_info is quicker than .info and breaks less often
        fi = tkr.fast_info
        out["market_cap"] = getattr(fi, "market_cap", np.nan)
        out["shares_outstanding"] = getattr(fi, "shares_outstanding", np.nan)
    except Exception:
        pass

    # Try slower fields only if available
    try:
        info = tkr.get_info()  # yfinance >=0.2.40
        out["long_name"] = info.get("longName")
        out["sector"] = info.get("sector")
        out["beta"] = info.get("beta", np.nan)
    except Exception:
        pass

    return out


def compute_drawdowns(df_px: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """
    Rolling-maximum drawdown over 'window' trading days.
    D(t) = Close(t) / rolling_max(Close, window) - 1
    """
    roll_max = df_px[price_col].rolling(window=window, min_periods=window).max()
    return (df_px[price_col] / roll_max) - 1.0


# --------------------------- Core Collection --------------------------- #

def fetch_one_ticker(ticker: str, start: str, end: str | None,
                     w3y: int, w5y: int, sleep_sec: float) -> Dict[str, Any]:
    """
    Returns dict with:
      - 'prices': DataFrame with OHLCV + dd_3y, dd_5y
      - 'snapshot': dict of latest per-ticker fields for screening table
    """
    tkr = yf.Ticker(ticker)

    # Price history (adjusted Close; OHLCV included)
    hist = tkr.history(start=start, end=end, interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No price data for {ticker}")

    hist = hist.reset_index().rename(columns={"index": "Date"})
    hist["Ticker"] = ticker

    # Compute drawdowns from rolling highs
    hist["DD_3Y"] = compute_drawdowns(hist, w3y, price_col="Close")
    hist["DD_5Y"] = compute_drawdowns(hist, w5y, price_col="Close")

    # Fundamentals snapshot
    finfo = safe_fastinfo(tkr)

    # Build latest snapshot row (most recent valid trading day)
    last = hist.dropna(subset=["Close"]).iloc[-1]
    snapshot = {
        "ticker": ticker,
        "date": pd.to_datetime(last["Date"]).date(),
        "close": float(last["Close"]),
        "dd_3y": float(last["DD_3Y"]) if pd.notna(last["DD_3Y"]) else np.nan,
        "dd_5y": float(last["DD_5Y"]) if pd.notna(last["DD_5Y"]) else np.nan,
        "market_cap": finfo["market_cap"],
        "beta": finfo["beta"],
        "shares_outstanding": finfo["shares_outstanding"],
        "long_name": finfo["long_name"],
        "sector": finfo["sector"],
    }

    time.sleep(sleep_sec)
    return {"prices": hist, "snapshot": snapshot}


def collect_all(tickers: List[str], start: str, end: str | None,
                w3y: int, w5y: int, sleep_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_prices = []
    snapshots = []

    for i, t in enumerate(tickers, 1):
        try:
            res = fetch_one_ticker(t, start, end, w3y, w5y, sleep_sec)
            all_prices.append(res["prices"])
            snapshots.append(res["snapshot"])
            print(f"[{i}/{len(tickers)}] {t}: ok")
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR -> {e}")

    if not all_prices:
        raise RuntimeError("No data collected. Check tickers or network.")

    prices_df = pd.concat(all_prices, ignore_index=True)
    snaps_df = pd.DataFrame(snapshots)
    return prices_df, snaps_df


def apply_first_screen(snaps_df: pd.DataFrame) -> pd.DataFrame:
    """Your initial rule: ≥30% drawdown (3y OR 5y) AND market cap ≥ $1B."""
    cond_dd = (snaps_df["dd_3y"] <= -0.30) | (snaps_df["dd_5y"] <= -0.30)
    cond_mcap = snaps_df["market_cap"] >= 1_000_000_000
    return snaps_df.loc[cond_dd & cond_mcap].sort_values(by=["dd_3y", "dd_5y"]).reset_index(drop=True)


# ------------------------------- Main ---------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tickers = load_tickers(args)
    print(f"Collecting {len(tickers)} tickers from {args.start} to {args.end or 'today'} ...")

    prices_df, snaps_df = collect_all(
        tickers=tickers,
        start=args.start,
        end=args.end,
        w3y=args.window3y,
        w5y=args.window5y,
        sleep_sec=args.sleep
    )

    # Save full timeseries (parquet = compact + fast)
    prices_path = os.path.join(args.outdir, "prices.parquet")
    prices_df.to_parquet(prices_path, index=False)

    # Save latest snapshot
    snapshot_path = os.path.join(args.outdir, "latest_snapshot.csv")
    snaps_df.sort_values("ticker").to_csv(snapshot_path, index=False)

    print(f"Saved timeseries -> {prices_path}")
    print(f"Saved snapshot  -> {snapshot_path}")

    if args.apply_screen:
        screened = apply_first_screen(snaps_df)
        screened_path = os.path.join(args.outdir, "screened_candidates.csv")
        screened.to_csv(screened_path, index=False)
        print(f"Saved first-pass screen -> {screened_path} (rows: {len(screened)})")


if __name__ == "__main__":
    main()
