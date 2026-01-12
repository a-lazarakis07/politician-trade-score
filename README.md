# politician-trade-score


## Install (editable)

```bash
pip install -e .
```
### On Windows PowerShell

If PowerShell blocks virtual-environment activation, use this one-time PATH update to point `pip` and `poltrade` to your venv:

```powershell
# from your project root
py -3.13 -m venv .venv
$env:Path = "$PWD\.venv\Scripts;$env:Path"
where python
where pip

pip install -e .

#Then run the CLI
## CLI usage

Fetch recent trades and score tickers:

```bash
poltrade --tickers AAPL TSLA --pages 50 --hold-length 180 --buy-weight 0.8 --details
```

- `--tickers`        One or more tickers.
- `--pages`          Number of pages to fetch from Capitol Trades (default 303 in the original).
- `--hold-length`    Lookback window in days for the date filter (default 180 in the original).
- `--buy-weight`     Weight on buy-ratio in final score. Recency weight is `1 - buy-weight`.
- `--details`        Print the intermediate details shown by the original script.

You can also pass explicit start and end dates (YYYY-MM-DD) instead of `--hold-length`:

```bash
poltrade --tickers NVDA --start-date 2025-06-01 --end-date 2025-09-01
```

## Library usage

```python
from poltrade.data import gather_data
from poltrade.score import PoliticianTradeScorer

data = gather_data()
scorer = PoliticianTradeScorer(buy_ratio_weight=0.8, recency_weight=0.2)
score = scorer.final_score(data, "AAPL", details=True, tutorial=True)
```

## Notes

- The `gather_data` function relies on `pandas.read_html` to scrape `capitoltrades.com` pages.