"""Fetch historical coffee futures data and save to CSV."""

import yfinance as yf
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def fetch_coffee_prices(
    ticker: str = "KC=F",
    period: str = "10y",
) -> pd.DataFrame:
    """Download coffee futures OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol. KC=F is Arabica coffee futures.
        period: How far back to fetch. Options: 1y, 2y, 5y, 10y, max.

    Returns:
        DataFrame with Date index and OHLCV columns.
    """
    print(f"Fetching {ticker} data for the last {period}...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    # Flatten multi-level columns from newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df = df.dropna()
    print(f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
    """Fetch related market data that may help prediction.

    Returns:
        Dict mapping name -> DataFrame for each supplementary series.
    """
    tickers = {
        "usd_index": "DX-Y.NYB",  # US Dollar Index
        "crude_oil": "CL=F",      # Crude oil futures
        "sugar": "SB=F",          # Sugar futures (correlated commodity)
        "sp500": "^GSPC",         # S&P 500 (risk sentiment)
        "brl_usd": "BRLUSD=X",   # Brazilian Real (Brazil = 35% of global coffee)
        "robusta": "RB=F",        # Robusta coffee futures (London ICE)
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

    # Fetch coffee prices
    coffee = fetch_coffee_prices()
    coffee.to_csv(DATA_DIR / "coffee_prices.csv")
    print(f"Saved coffee prices to {DATA_DIR / 'coffee_prices.csv'}")

    # Fetch supplementary data and merge on date
    supplementary = fetch_supplementary_data()
    combined = coffee[["Close"]].rename(columns={"Close": "coffee_close"})
    for name, df in supplementary.items():
        combined = combined.join(df, how="left")

    combined = combined.ffill(limit=7)  # forward-fill missing dates
    combined.to_csv(DATA_DIR / "combined_features.csv")
    print(f"Saved combined features to {DATA_DIR / 'combined_features.csv'}")


if __name__ == "__main__":
    main()
