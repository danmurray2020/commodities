"""Fetch historical cocoa futures data and supplementary market data."""

import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def fetch_cocoa_prices(ticker: str = "CC=F", period: str = "10y") -> pd.DataFrame:
    """Download cocoa futures OHLCV data from Yahoo Finance."""
    print(f"Fetching {ticker} data for the last {period}...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df = df.dropna()
    print(f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
    """Fetch related market data."""
    tickers = {
        "usd_index": "DX-Y.NYB",      # US Dollar Index
        "crude_oil": "CL=F",           # Crude oil (shipping/energy costs)
        "sugar": "SB=F",               # Sugar (competing crop in some regions)
        "sp500": "^GSPC",              # S&P 500 (risk sentiment)
        "ghs_usd": "GHSUSD=X",        # Ghanaian Cedi (Ghana = #2 producer)
        "eur_usd": "EURUSD=X",        # EUR/USD (London ICE cocoa trades in EUR)
        "coffee": "KC=F",              # Coffee (correlated soft commodity)
    }
    supplementary = {}
    for name, ticker in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        df = yf.download(ticker, period="10y", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        if not df.empty:
            supplementary[name] = df[["Close"]].rename(columns={"Close": name})
    return supplementary


def main():
    DATA_DIR.mkdir(exist_ok=True)

    cocoa = fetch_cocoa_prices()
    cocoa.to_csv(DATA_DIR / "cocoa_prices.csv")
    print(f"Saved cocoa prices to {DATA_DIR / 'cocoa_prices.csv'}")

    supplementary = fetch_supplementary_data()
    combined = cocoa[["Close"]].rename(columns={"Close": "cocoa_close"})
    for name, df in supplementary.items():
        combined = combined.join(df, how="left")

    combined = combined.ffill(limit=7)
    combined.to_csv(DATA_DIR / "combined_features.csv")
    print(f"Saved combined features to {DATA_DIR / 'combined_features.csv'}")


if __name__ == "__main__":
    main()
