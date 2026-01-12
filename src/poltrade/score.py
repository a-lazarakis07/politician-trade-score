# -*- coding: utf-8 -*-
"""Scoring helpers and a thin class wrapper (logic kept the same)."""
from __future__ import annotations

import pandas as pd
import datetime as date

# ---------------- Core helpers ----------------

def get_recency(df: pd.DataFrame, ticker: str) -> float:
    """Checks if the most recent trade was a buy or sell (unchanged)."""
    trades = df.loc[df['Ticker'] == ticker, ['Type', 'Traded']]

    buy_date  = trades.loc[trades['Type'] == 'buy',  'Traded'].max()
    sell_date = trades.loc[trades['Type'] == 'sell', 'Traded'].max()

    if pd.isna(sell_date):  # no sells
        return 1.0
    if pd.isna(buy_date):   # no buys
        return 0.0

    if buy_date > sell_date:
        return 1.0
    elif buy_date < sell_date:
        return 0.0
    else:
        return 0.5

def get_buy_ratio(df: pd.DataFrame) -> float:
    """Calculates the buy ratio of the data frame (unchanged)."""
    sell_data = df[df['Type'] == 'sell']
    buy_data  = df[df['Type'] == 'buy']

    sell_weight = sell_data['Weight'].sum()
    buy_weight  = buy_data['Weight'].sum()

    denom = (sell_weight + buy_weight)
    if denom == 0:
        return 0.0
    return float(buy_weight / denom)

def buy_score(x: float, mu: float = 0.3) -> float:
    """Piecewise score function (unchanged exponents/shape)."""
    if x <= mu:
        return 0.5 * (x / mu) ** 4
    else:
        return 0.5 + 0.5 * ((x - mu) / (1 - mu)) ** 0.45

def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Ensure we catch tickers embedded in text (unchanged idea)."""
    filtered_df = df[df['Ticker'].str.contains(ticker, na=False)].copy()
    filtered_df['Ticker'] = ticker
    return filtered_df

# ---------------- Final score (copied) ----------------

def final_score(
    df: pd.DataFrame,
    ticker: str,
    buy_ratio_weight: float = 0.8,
    recency_weight: float = 0.2,
    details: bool = False,
    prints: bool = True,
    tutorial: bool = True,
) -> float:
    """Calculate the final score for a ticker (unchanged logic).

    Assumes `buy_ratio_weight + recency_weight = 1`.
    """
    ticker = ticker.upper()
    ticker_data = df.loc[df['Ticker'] == ticker]

    # Clean if missing due to naming edge cases
    if ticker_data.empty:
        ticker_data = clean_data(df, ticker)
        if ticker_data.empty:
            if prints:
                print(f"No entries for {ticker}")
                print("No recommendation")
            return 0.5

    if tutorial and prints:
        print("Ranges from 0-1, Buy recommended if above 0.5")

    # Buy-ratio vs aggregate mean
    mu = get_buy_ratio(df)
    ticker_buy_ratio = get_buy_ratio(ticker_data)
    buy = buy_score(ticker_buy_ratio, mu)

    recency = get_recency(df, ticker)

    if details and prints:
        print(f"Aggregate Buy Ratio: {mu}")
        print(f"{ticker} Buy Ratio: {ticker_buy_ratio}")
        print(f"{ticker} Buy Score: {buy}")
        print(f"{ticker} Recency Score: {recency}")

    score = buy_ratio_weight * buy + recency_weight * recency
    score = round(float(score), 2)

    if prints:
        action = "Buy recommended" if score > 0.5 else "Sell recommended"
        print(f"{ticker}: {action} | Score: {score}")

    return score

# ---------------- Thin class wrapper ----------------

class PoliticianTradeScorer:
    """Small OO wrapper around the original functions.

    """
    def __init__(
        self,
        buy_ratio_weight: float = 0.8,
        recency_weight: float = 0.2,
    ) -> None:
        if abs((buy_ratio_weight + recency_weight) - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.0")
        self.buy_ratio_weight = buy_ratio_weight
        self.recency_weight = recency_weight

    def final_score(
        self,
        df: pd.DataFrame,
        ticker: str,
        details: bool = False,
        prints: bool = True,
        tutorial: bool = True,
    ) -> float:
        return final_score(
            df,
            ticker,
            buy_ratio_weight=self.buy_ratio_weight,
            recency_weight=self.recency_weight,
            details=details,
            prints=prints,
            tutorial=tutorial,
        )