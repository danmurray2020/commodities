"""Fetch natural gas futures data and supplementary market data."""

import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def fetch_prices(ticker: str = "NG=F", period: str = "10y") -> pd.DataFrame:
    print(f"Fetching {ticker} data for the last {period}...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df = df.dropna()
    print(f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
    tickers = {
        "crude_oil": "CL=F",        # Correlated energy commodity
        "heating_oil": "HO=F",      # Direct heating demand proxy
        "usd_index": "DX-Y.NYB",
        "sp500": "^GSPC",
        "coal": "MTF=F",            # Competing fuel source
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
    ng = fetch_prices()
    ng.to_csv(DATA_DIR / "natgas_prices.csv")

    supplementary = fetch_supplementary_data()
    combined = ng[["Close"]].rename(columns={"Close": "natgas_close"})
    for name, df in supplementary.items():
        combined = combined.join(df, how="left")
    combined = combined.ffill(limit=7)
    combined.to_csv(DATA_DIR / "combined_features.csv")
    print(f"Saved combined features to {DATA_DIR / 'combined_features.csv'}")


if __name__ == "__main__":
    main()
