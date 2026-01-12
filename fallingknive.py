#!/usr/bin/env python3
"""
Analyze 'falling knife' recoveries.

This script:
  - Loads historical OHLCV (from a parquet created earlier OR via yfinance)
  - Computes indicators (RSI, rolling volatility, MA20, MA50)
  - Detects drawdown events where price <= (rolling 3y high)*(1 - dd_threshold)
  - For each event:
      * finds the local bottom after the trigger
      * checks if price recovers +X% from bottom within Y days
      * measures 'time to mean reversion' via MA50 cross
      * records volatility contraction metrics
  - Writes:
      * data/event_metrics.csv (per-event metrics)
      * data/ticker_summary.csv (per-ticker aggregates)

Examples:
  python analyze_recoveries.py --parquet data/prices.parquet --dd-threshold 0.30
  python analyze_recoveries.py --tickers AAPL NVDA GM GIB --start 2015-01-01
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

# ------------------------------ Utils -------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure post-drawdown recovery probabilities & time to mean reversion.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--parquet", type=str, help="Path to prices.parquet (with columns: Date, Ticker, Open, High, Low, Close, Volume)")
    g.add_argument("--tickers", nargs="+", help="Fetch via yfinance: space-separated tickers")
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date if fetching")
    p.add_argument("--end", type=str, default=None, help="End date if fetching")
    p.add_argument("--outdir", type=str, default="data", help="Output directory")
    p.add_argument("--dd-threshold", type=float, default=0.30, help="Drawdown threshold (e.g., 0.30 = 30%%)")
    p.add_argument("--dd-lookback-days", type=int, default=252*3, help="Rolling high lookback for drawdown (~3y)")
    p.add_argument("--bottom_window_days", type=int, default=60, help="Max days after trigger to find local bottom")
    p.add_argument("--recovery_gain", type=float, default=0.20, help="Recovery target relative to bottom (e.g., 0.20 = +20%%)")
    p.add_argument("--recovery_window_days", type=int, default=252, help="Days to achieve recovery after bottom")
    p.add_argument("--rsi_len", type=int, default=14, help="RSI length")
    p.add_argument("--vol_window", type=int, default=20, help="Rolling std window for volatility")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Simple RSI implementation (Wilder's smoothing approximated with EMA).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Use exponential moving averages to approximate Wilder
    roll_up = gain.ewm(alpha=1/length, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)  # neutral where unknown


def compute_indicators(df: pd.DataFrame,
                       rsi_len: int = 14,
                       vol_window: int = 20) -> pd.DataFrame:
    """
    Expects df with columns: Date, Ticker, Close (and optionally others).
    Returns df with RSI, VOL (rolling std of returns), MA20, MA50.
    """
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    df["MA20"]   = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(20).mean())
    df["MA50"]   = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(50).mean())
    df["RSI"]    = df.groupby("Ticker")["Close"].transform(lambda s: rsi(s, rsi_len))
    df["VOL"]    = df.groupby("Ticker")["Return"].transform(lambda s: s.rolling(vol_window).std())
    # For drawdown detection relative to rolling 3y high:
    df["ROLL_MAX_3Y"] = df.groupby("Ticker")["Close"].transform(
        lambda s: s.rolling(window=252*3, min_periods=252).max()
    )
    df["DD_3Y"] = (df["Close"] / df["ROLL_MAX_3Y"]) - 1.0
    return df


@dataclass
class EventResult:
    ticker: str
    trigger_date: pd.Timestamp
    trigger_close: float
    drawdown_at_trigger: float
    bottom_date: Optional[pd.Timestamp]
    bottom_close: Optional[float]
    days_to_bottom: Optional[int]
    recovered: int
    recovery_date: Optional[pd.Timestamp]
    days_bottom_to_recovery: Optional[int]
    crossed_ma50: int
    ma50_cross_date: Optional[pd.Timestamp]
    days_bottom_to_ma50: Optional[int]
    rsi_at_trigger: Optional[float]
    rsi_at_bottom: Optional[float]
    vol_before_bottom: Optional[float]
    vol_after_bottom: Optional[float]


def detect_drawdown_events(df_t: pd.DataFrame,
                           dd_threshold: float,
                           lookback_days: int) -> List[pd.Timestamp]:
    """
    Return dates where Close <= ROLL_MAX_3Y*(1 - dd_threshold)
    Debounce events: require at least 63 trading days between consecutive triggers.
    """
    mask = df_t["ROLL_MAX_3Y"].notna() & (df_t["Close"] <= df_t["ROLL_MAX_3Y"] * (1 - dd_threshold))
    candidates = df_t.loc[mask, "Date"].tolist()
    triggers = []
    last = None
    for d in candidates:
        if last is None or (d - last).days >= 63:  # ~3 months spacing
            triggers.append(d)
            last = d
    return triggers


def analyze_events_for_ticker(df_t: pd.DataFrame,
                              dd_threshold: float,
                              lookback_days: int,
                              bottom_window_days: int,
                              recovery_gain: float,
                              recovery_window_days: int) -> List[EventResult]:
    """
    For a single ticker DataFrame (sorted by Date).
    """
    results: List[EventResult] = []
    triggers = detect_drawdown_events(df_t, dd_threshold, lookback_days)

    for td in triggers:
        # Row at trigger
        row_trig = df_t.loc[df_t["Date"] == td].iloc[0]
        trig_idx = df_t.index[df_t["Date"] == td][0]

        # 1) Find bottom within next bottom_window_days (local min close)
        end_idx = df_t.index.min() + len(df_t)  # fallback
        future_window = df_t.iloc[trig_idx : trig_idx + bottom_window_days + 1]
        if future_window.empty:
            continue

        i_min = future_window["Close"].idxmin()
        bottom_row = future_window.loc[i_min]
        bottom_date = bottom_row["Date"]
        bottom_close = float(bottom_row["Close"])
        days_to_bottom = int((bottom_date - td).days)

        # 2) Recovery check: +recovery_gain from bottom within recovery_window_days
        recovery_level = bottom_close * (1 + recovery_gain)
        future_after_bottom = df_t[df_t["Date"] >= bottom_date].copy()
        future_after_bottom = future_after_bottom[future_after_bottom["Date"] <= bottom_date + pd.Timedelta(days=recovery_window_days)]
        reached = future_after_bottom[future_after_bottom["Close"] >= recovery_level]
        if not reached.empty:
            rec_row = reached.iloc[0]
            recovered = 1
            recovery_date = rec_row["Date"]
            days_bottom_to_recovery = int((recovery_date - bottom_date).days)
        else:
            recovered = 0
            recovery_date = None
            days_bottom_to_recovery = None

        # 3) Time to mean reversion proxy: first cross above MA50 after bottom
        cross = future_after_bottom[future_after_bottom["Close"] >= future_after_bottom["MA50"]]
        if not cross.empty:
            ma50_row = cross.iloc[0]
            crossed_ma50 = 1
            ma50_cross_date = ma50_row["Date"]
            days_bottom_to_ma50 = int((ma50_cross_date - bottom_date).days)
        else:
            crossed_ma50 = 0
            ma50_cross_date = None
            days_bottom_to_ma50 = None

        # 4) Volatility contraction snapshot (mean VOL 20d before vs after bottom)
        look = 20
        pre = df_t[(df_t["Date"] < bottom_date) & (df_t["Date"] >= bottom_date - pd.Timedelta(days=look*2))]["VOL"].mean()
        post = df_t[(df_t["Date"] > bottom_date) & (df_t["Date"] <= bottom_date + pd.Timedelta(days=look*2))]["VOL"].mean()

        res = EventResult(
            ticker=str(row_trig["Ticker"]),
            trigger_date=td,
            trigger_close=float(row_trig["Close"]),
            drawdown_at_trigger=float(row_trig["DD_3Y"]),
            bottom_date=bottom_date,
            bottom_close=bottom_close,
            days_to_bottom=days_to_bottom,
            recovered=recovered,
            recovery_date=recovery_date,
            days_bottom_to_recovery=days_bottom_to_recovery,
            crossed_ma50=crossed_ma50,
            ma50_cross_date=ma50_cross_date,
            days_bottom_to_ma50=days_bottom_to_ma50,
            rsi_at_trigger=float(row_trig["RSI"]) if pd.notna(row_trig["RSI"]) else None,
            rsi_at_bottom=float(bottom_row["RSI"]) if pd.notna(bottom_row["RSI"]) else None,
            vol_before_bottom=float(pre) if pd.notna(pre) else None,
            vol_after_bottom=float(post) if pd.notna(post) else None,
        )
        results.append(res)

    return results


def load_data_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure column names and dtypes
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "Date" not in df.columns:
        raise ValueError("Parquet must include a 'Date' column")
    df["Date"] = pd.to_datetime(df["Date"])
    # Keep key columns if present
    keep = [c for c in ["Date","Ticker","Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep].dropna(subset=["Close"])


def fetch_with_yfinance(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. `pip install yfinance`")
    all_rows = []
    for t in tickers:
        h = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False)
        if h.empty:
            continue
        h = h.reset_index().rename(columns={"index":"Date"})
        h["Ticker"] = t
        all_rows.append(h[["Date","Ticker","Open","High","Low","Close","Volume"]])
    if not all_rows:
        raise RuntimeError("No data fetched. Check tickers or dates.")
    return pd.concat(all_rows, ignore_index=True)


def summarize_events(events: List[EventResult]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append({
            "ticker": e.ticker,
            "trigger_date": e.trigger_date,
            "trigger_close": e.trigger_close,
            "drawdown_at_trigger": e.drawdown_at_trigger,
            "bottom_date": e.bottom_date,
            "bottom_close": e.bottom_close,
            "days_to_bottom": e.days_to_bottom,
            "recovered_+{:.0f}%".format(100*args.recovery_gain): e.recovered,
            "recovery_date": e.recovery_date,
            "days_bottom_to_recovery": e.days_bottom_to_recovery,
            "crossed_ma50": e.crossed_ma50,
            "ma50_cross_date": e.ma50_cross_date,
            "days_bottom_to_ma50": e.days_bottom_to_ma50,
            "rsi_at_trigger": e.rsi_at_trigger,
            "rsi_at_bottom": e.rsi_at_bottom,
            "vol_before_bottom": e.vol_before_bottom,
            "vol_after_bottom": e.vol_after_bottom,
            "vol_contraction": (e.vol_before_bottom - e.vol_after_bottom) if (e.vol_before_bottom is not None and e.vol_after_bottom is not None) else None
        })
    return pd.DataFrame(rows)


def per_ticker_summary(events_df: pd.DataFrame,
                       recovery_col: str) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    grp = events_df.groupby("ticker")
    out = grp.agg(
        events=("ticker","count"),
        recovery_rate=(recovery_col, "mean"),
        median_days_to_recovery=("days_bottom_to_recovery","median"),
        median_days_to_ma50=("days_bottom_to_ma50","median"),
        median_days_to_bottom=("days_to_bottom","median"),
        median_vol_contraction=("vol_contraction","median"),
        avg_trigger_dd=("drawdown_at_trigger","mean")
    ).reset_index()
    return out.sort_values(["recovery_rate","events"], ascending=[False, False])


# ------------------------------ Main --------------------------------- #

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    ensure_outdir(args.outdir)

    # 1) Load data
    if args.parquet:
        prices = load_data_from_parquet(args.parquet)
    else:
        prices = fetch_with_yfinance(args.tickers, args.start, args.end)

    # 2) Indicators & drawdowns
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = compute_indicators(
        prices,
        rsi_len=args.rsi_len,
        vol_window=args.vol_window
    )

    # 3) Per-ticker event scan
    all_events: List[EventResult] = []
    for tkr, df_t in prices.sort_values("Date").groupby("Ticker"):
        df_t = df_t.reset_index(drop=True)
        evs = analyze_events_for_ticker(
            df_t=df_t,
            dd_threshold=args.dd_threshold,
            lookback_days=args.dd_lookup_days if hasattr(args, "dd_lookup_days") else args.dd_lookback_days,
            bottom_window_days=args.bottom_window_days,
            recovery_gain=args.recovery_gain,
            recovery_window_days=args.recovery_window_days
        )
        all_events.extend(evs)

    # 4) Outputs
    events_df = summarize_events(all_events)
    recovery_col = f"recovered_+{int(100*args.recovery_gain)}%"
    events_path = os.path.join(args.outdir, "event_metrics.csv")
    events_df.to_csv(events_path, index=False)

    summary_df = per_ticker_summary(events_df, recovery_col=recovery_col)
    summary_path = os.path.join(args.outdir, "ticker_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Wrote per-event metrics -> {events_path} (rows={len(events_df)})")
    print(f"[OK] Wrote per-ticker summary -> {summary_path} (rows={len(summary_df)})")
