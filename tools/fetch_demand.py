#!/usr/bin/env python3
"""Fetch demand-side proxy data: buyer stock prices and consumer confidence.

Downloads major buyer stock prices for coffee, sugar, and cocoa,
plus consumer sentiment from FRED (if API key available).

Usage:
    python3 tools/fetch_demand.py                # fetch all commodities
    python3 tools/fetch_demand.py --commodity coffee
    python3 tools/fetch_demand.py --commodity sugar --period 5y
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.retry import download_with_retry

logger = logging.getLogger("commodities.fetch_demand")

# ── Ticker mappings per commodity ──────────────────────────────────────────

COMMODITY_TICKERS = {
    "coffee": {
        "SBUX": "sbux",    # Starbucks
        "NSRGY": "nsrgy",  # Nestle
    },
    "sugar": {
        "KO": "ko",        # Coca-Cola
        "PEP": "pep",      # PepsiCo
    },
    "chocolate": {
        "MDLZ": "mdlz",    # Mondelez
        "HSY": "hsy",       # Hershey
    },
}

# S&P 500 as general market proxy (included for completeness)
GENERAL_TICKERS = {
    "^GSPC": "sp500",
}


def fetch_buyer_stocks(commodity: str, period: str = "10y") -> pd.DataFrame:
    """Fetch buyer stock close prices for a given commodity.

    Args:
        commodity: One of 'coffee', 'sugar', 'chocolate'.
        period: yfinance period string (e.g. '10y', '5y').

    Returns:
        DataFrame with date index and lowercase ticker columns.
    """
    tickers = COMMODITY_TICKERS.get(commodity, {})
    all_tickers = {**tickers, **GENERAL_TICKERS}

    frames = {}
    for yf_ticker, col_name in all_tickers.items():
        try:
            logger.info(f"Downloading {yf_ticker} for {commodity}...")
            df = download_with_retry(yf_ticker, period=period)
            if df is not None and not df.empty:
                frames[col_name] = df["Close"]
                logger.info(f"  {yf_ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                logger.warning(f"  {yf_ticker}: no data returned")
        except Exception as e:
            logger.error(f"  {yf_ticker} failed: {e}")

    if not frames:
        logger.warning(f"No buyer stock data fetched for {commodity}")
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index.name = "Date"
    result = result.sort_index()
    return result


def fetch_consumer_confidence() -> pd.DataFrame | None:
    """Fetch University of Michigan Consumer Sentiment from FRED.

    Requires FRED_API_KEY environment variable.
    Returns monthly data that should be forward-filled to daily.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.info("FRED_API_KEY not set, skipping consumer confidence fetch")
        return None

    try:
        import fredapi
        fred = fredapi.Fred(api_key=api_key)
        umcsent = fred.get_series("UMCSENT")
        df = pd.DataFrame({"umcsent": umcsent})
        df.index.name = "Date"
        logger.info(f"Consumer confidence: {len(df)} monthly observations")
        return df
    except ImportError:
        logger.warning("fredapi not installed, trying direct CSV download")
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=UMCSENT&api_key={api_key}&file_type=json"
            )
            import requests
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = [
                {"Date": obs["date"], "umcsent": float(obs["value"])}
                for obs in data["observations"]
                if obs["value"] != "."
            ]
            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            logger.info(f"Consumer confidence (JSON): {len(df)} observations")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch consumer confidence: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch consumer confidence: {e}")
        return None


def save_demand_data(commodity: str, period: str = "10y") -> None:
    """Fetch and save demand data for a single commodity.

    Saves to {commodity}/data/demand_data.csv
    """
    data_dir = REPO_ROOT / commodity / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist, creating it")
        data_dir.mkdir(parents=True, exist_ok=True)

    # Fetch buyer stocks
    stocks_df = fetch_buyer_stocks(commodity, period=period)

    # Fetch consumer confidence
    cc_df = fetch_consumer_confidence()

    # Merge if both available
    if not stocks_df.empty and cc_df is not None:
        # Forward-fill monthly consumer confidence to daily
        combined = stocks_df.join(cc_df, how="left")
        combined["umcsent"] = combined["umcsent"].ffill()
    elif not stocks_df.empty:
        combined = stocks_df
    elif cc_df is not None:
        combined = cc_df
    else:
        logger.warning(f"No demand data available for {commodity}")
        return

    out_path = data_dir / "demand_data.csv"
    combined.to_csv(out_path)
    logger.info(f"Saved {len(combined)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch demand-side proxy data")
    parser.add_argument(
        "--commodity",
        choices=["coffee", "sugar", "chocolate", "all"],
        default="all",
        help="Which commodity to fetch demand data for (default: all)",
    )
    parser.add_argument(
        "--period",
        default="10y",
        help="How far back to fetch (default: 10y)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    commodities = ["coffee", "sugar", "chocolate"]
    if args.commodity != "all":
        commodities = [args.commodity]

    for commodity in commodities:
        logger.info(f"=== Fetching demand data for {commodity} ===")
        try:
            save_demand_data(commodity, period=args.period)
        except Exception as e:
            logger.error(f"Failed to fetch demand data for {commodity}: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
