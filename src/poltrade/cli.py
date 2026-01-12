from __future__ import annotations
import argparse
import datetime as date

from .data import gather_data
from .score import PoliticianTradeScorer

def _parse_date(s: str) -> date.date:
    return date.date.fromisoformat(s)

def main() -> None:
    parser = argparse.ArgumentParser(description="Score tickers based on Capitol Trades activity.")
    parser.add_argument("--tickers", nargs="+", required=True, help="One or more ticker symbols")
    parser.add_argument("--pages", type=int, default=303, help="Pages to scrape (default 303 from original script)")
    parser.add_argument("--hold-length", type=int, default=180, help="Lookback window in days (default 180)")
    parser.add_argument("--start-date", type=_parse_date, help="YYYY-MM-DD (overrides hold-length if both dates given)")
    parser.add_argument("--end-date", type=_parse_date, help="YYYY-MM-DD (overrides hold-length if both dates given)")
    parser.add_argument("--buy-weight", type=float, default=0.8, help="Weight for buy-ratio; recency weight is 1 - this")
    parser.add_argument("--details", action="store_true", help="Print intermediate scoring details")
    parser.add_argument("--no-tutorial", action="store_true", help="Disable tutorial text in output")
    args = parser.parse_args()

    # Determine date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        end_date = date.date.today()
        start_date = end_date - date.timedelta(days=args.hold_length)

    # Fetch data
    data = gather_data(pages=args.pages, start_date=start_date, end_date=end_date)

    # Abort early if nothing came back
    if data is None or data.empty:
        print(f"No trades found for {start_date} → {end_date} with pages={args.pages}. "
              "Try increasing --pages or widening the date range.")
        return

    # Build scorer and evaluate
    scorer = PoliticianTradeScorer(
        buy_ratio_weight=args.buy_weight,
        recency_weight=1.0 - args.buy_weight
    )

    print("—————————————————————————————————————————————")
    for t in args.tickers:
        s = scorer.final_score(
            data, t,
            details=args.details,
            tutorial=not args.no_tutorial,
        )
        print("—————————————————————————————————————————————")

if __name__ == "__main__":
    main()
