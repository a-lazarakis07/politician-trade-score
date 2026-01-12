import pandas as pd
import datetime as date

from poltrade.score import get_buy_ratio, get_recency, final_score

def _dummy_df():
    now = date.datetime(2025, 9, 1)
    rows = [
        {"Ticker": "AAPL", "Type": "buy",  "Traded": now,               "Weight": 10},
        {"Ticker": "AAPL", "Type": "sell", "Traded": now - date.timedelta(days=3), "Weight":  5},
        {"Ticker": "MSFT", "Type": "sell", "Traded": now - date.timedelta(days=1), "Weight": 10},
    ]
    return pd.DataFrame(rows)

def test_buy_ratio():
    df = _dummy_df()
    r = get_buy_ratio(df)
    assert 0 <= r <= 1

def test_recency():
    df = _dummy_df()
    assert get_recency(df, "AAPL") in (0.0, 0.5, 1.0)

def test_final_score_runs():
    df = _dummy_df()
    s = final_score(df, "AAPL", prints=False, tutorial=False)
    assert 0 <= s <= 1