"""Fetch Live Cattle futures data and supplementary market data."""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.retry import download_with_retry

DATA_DIR = Path(__file__).parent / "data"


def fetch_prices(ticker: str = "LE=F", period: str = "10y") -> pd.DataFrame:
    print(f"Fetching {ticker} data for the last {period}...")
    df = download_with_retry(ticker, period=period)
    df.index.name = "Date"
    df = df.dropna()
    print(f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
    tickers = {
        "usd_index": "DX-Y.NYB",
        "sp500": "^GSPC",
        "us10y": "^TNX",
        "vix": "^VIX",
    }
    supplementary = {}
    for name, ticker in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        df = download_with_retry(ticker, period="10y")
        df.index.name = "Date"
        if not df.empty:
            supplementary[name] = df[["Close"]].rename(columns={"Close": name})
    return supplementary


def main():
    DATA_DIR.mkdir(exist_ok=True)
    prices = fetch_prices()
    prices.to_csv(DATA_DIR / "live_cattle_prices.csv")

    supplementary = fetch_supplementary_data()
    combined = prices[["Close"]].rename(columns={"Close": "cattle_close"})
    for name, df in supplementary.items():
        combined = combined.join(df, how="left")
    combined = combined.ffill(limit=7)
    combined.to_csv(DATA_DIR / "combined_features.csv")
    print(f"Saved combined features to {DATA_DIR / 'combined_features.csv'}")


if __name__ == "__main__":
    main()
