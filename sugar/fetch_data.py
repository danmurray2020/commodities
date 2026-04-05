"""Fetch historical sugar futures data and supplementary market data."""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.retry import download_with_retry

DATA_DIR = Path(__file__).parent / "data"


def fetch_prices(ticker: str = "SB=F", period: str = "10y") -> pd.DataFrame:
    print(f"Fetching {ticker} data for the last {period}...")
    df = download_with_retry(ticker, period=period)
    df.index.name = "Date"
    df = df.dropna()
    print(f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
    tickers = {
        "usd_index": "DX-Y.NYB",
        "crude_oil": "CL=F",       # Energy costs + ethanol link
        "coffee": "KC=F",          # Correlated soft commodity
        "cocoa": "CC=F",           # Correlated soft commodity
        "sp500": "^GSPC",
        "brl_usd": "BRLUSD=X",    # Brazil = #1 sugar producer
        "inr_usd": "INRUSD=X",    # India = #2 producer
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
    sugar = fetch_prices()
    sugar.to_csv(DATA_DIR / "sugar_prices.csv")
    print(f"Saved sugar prices to {DATA_DIR / 'sugar_prices.csv'}")

    supplementary = fetch_supplementary_data()
    combined = sugar[["Close"]].rename(columns={"Close": "sugar_close"})
    for name, df in supplementary.items():
        combined = combined.join(df, how="left")
    combined = combined.ffill(limit=7)
    combined.to_csv(DATA_DIR / "combined_features.csv")
    print(f"Saved combined features to {DATA_DIR / 'combined_features.csv'}")


if __name__ == "__main__":
    main()
