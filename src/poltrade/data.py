# -*- coding: utf-8 -*-
"""Data retrieval jhelpers"""

from __future__ import annotations

import pandas as pd
import datetime as date

# Weight mapping 
_WEIGHT_DICT = {
    "< 1K": 1, "1K–15K": 15, "15K–50K": 50, "50K–100K": 100, "100K–250K": 250,
    "250K–500K": 500, "500K–1M": 1000, "1M–5M": 5000, "5M–25M": 25000, "25M–50M": 50000
}

# Original base URL 
_DEFAULT_TRADES_LINK = (
    "https://www.capitoltrades.com/trades?pageSize=96&assetType=etf&assetType=stock&assetType=stock-options&assetType=stock-appreciation-right&page="
)

def create_date_range(date1: date.date, date2: date.date) -> str:
    """Create the date range suffix for Capitol Trades URLs (unchanged)."""
    return "&txDate=" + str(date1) + "%2C" + str(date2)

def convert_to_date(date_str: str) -> date.datetime:
    """Convert CapitolTrades date strings to a `datetime` (unchanged logic)."""
    if 'Yesterday' in date_str:
        return date.datetime.today() - date.timedelta(days=1)
    elif 'Today' in date_str:
        return date.datetime.today()
    date_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'ept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    year = date_str[-4:]
    month = date_dict[date_str[-7:-4]]
    # Handle 'Sept' (length 4) vs 3-letter months
    if month != 9:
        day = int(date_str[:-7])
    else:
        day = int(date_str[:-8])
    return date.datetime(int(year), month, day)

def gather_data(
    base_url: str = _DEFAULT_TRADES_LINK,
    pages: int = 25, 
    start_date: date.date | None = None,
    end_date: date.date | None = None,
) -> pd.DataFrame:
    """Scrape buy/sell information from Capitol Trades (logic unchanged).

    Filters should be adjusted on the website itself (e.g., trade size).
    The function expects :US equities (public US tickers).

    Returns a DataFrame with filing date, type, and ticker, similar to the original.
    """
    all_tables = []

    # Build optional date suffix
    if start_date is None or end_date is None:
        date_string = ''
    else:
        date_string = create_date_range(start_date, end_date)

    # Loop over pages
    for page in range(1, pages + 1):
        url = base_url + str(page) + date_string
        tables = pd.read_html(url)  # extract tables

        # Ensure page has results
        if tables[0]['Politician'][0] != 'No results.':
            all_tables.append(tables[0])
        else:
            break

    if not all_tables:
        # Mirror the original behavior (print + empty frame)
        print("No tables found matching the criteria.")
        return pd.DataFrame()

    # Combine tables
    df = pd.concat(all_tables, ignore_index=True)

    # Extract ticker before ':US' (keep original regex behavior, with ETF/INC edge cases)
    df['Ticker'] = df['Traded Issuer'].str.extract(r'(?:ETF)?\s*([A-Z/]{1,5}):US')
    df['Ticker'] = df['Ticker'].str.replace('/', '-', regex=True)

    # Convert 'Traded' column to datetime
    df['Traded'] = df['Traded'].apply(convert_to_date)

    # Map weights; default to 15 if missing (unchanged)
    df['Weight'] = df['Size'].map(_WEIGHT_DICT)
    df['Weight'] = df['Weight'].fillna(15)

    # Drop unused columns to match original output
    drop_cols = [c for c in ['Owner', 'Unnamed: 9', 'Size', 'Published'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df