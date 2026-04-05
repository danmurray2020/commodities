"""Equity-Commodity Correlation Scanner.

For each commodity, finds equities that are most sensitive to its price moves.
Measures commodity beta, correlation, and R² to identify tradeable equity plays.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "equity_data"

# Equity universe mapped to commodities
EQUITY_MAP = {
    "coffee": {
        "producers": ["BG", "ADM"],                    # Grain traders
        "consumers": ["SBUX", "KDP", "MDLZ"],         # Coffee buyers
    },
    "cocoa": {
        "producers": ["BG", "ADM"],
        "consumers": ["HSY", "MDLZ", "TR"],           # Chocolate makers (Tootsie Roll)
    },
    "sugar": {
        "producers": ["BG", "ADM"],
        "consumers": ["KO", "PEP", "MDLZ"],           # Beverage/snack companies
    },
    "natgas": {
        "producers": ["EQT", "CTRA", "AR", "RRC", "SWN"],  # Gas producers
        "consumers": ["XLU", "SO", "DUK", "NEE"],          # Utilities
    },
    "soybeans": {
        "producers": ["BG", "ADM", "INGR"],            # Processors
        "consumers": ["TSN", "HRL", "PPC"],            # Meat producers (feed costs)
    },
    "wheat": {
        "producers": ["BG", "ADM"],
        "consumers": ["GIS", "FLO", "SJM", "CPB"],    # Food manufacturers
    },
    "copper": {
        "producers": ["FCX", "SCCO", "TECK", "RIO"],  # Miners
        "consumers": ["XHB", "DHI", "LEN"],            # Home builders
    },
}

# Commodity data paths
COMMODITY_PATHS = {
    "coffee": Path(__file__).parent.parent / "coffee" / "data" / "combined_features.csv",
    "cocoa": Path(__file__).parent.parent / "chocolate" / "data" / "combined_features.csv",
    "sugar": Path(__file__).parent.parent / "sugar" / "data" / "combined_features.csv",
    "natgas": Path(__file__).parent.parent / "natgas" / "data" / "combined_features.csv",
    "soybeans": Path(__file__).parent.parent / "soybeans" / "data" / "combined_features.csv",
    "wheat": Path(__file__).parent.parent / "wheat" / "data" / "combined_features.csv",
    "copper": Path(__file__).parent.parent / "copper" / "data" / "combined_features.csv",
}

PRICE_COLS = {
    "coffee": "coffee_close", "cocoa": "cocoa_close", "sugar": "sugar_close",
    "natgas": "natgas_close", "soybeans": "soybeans_close", "wheat": "wheat_close",
    "copper": "copper_close",
}


def fetch_equity_data(tickers: list, period: str = "5y") -> pd.DataFrame:
    """Fetch equity price data for a list of tickers."""
    all_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                all_data[ticker] = df["Close"]
        except Exception as e:
            print(f"  WARNING: Failed to fetch {ticker}: {e}")
    return pd.DataFrame(all_data)


def compute_commodity_beta(commodity_returns: pd.Series, equity_returns: pd.Series,
                           windows: list = [21, 63, 126]) -> dict:
    """Compute commodity beta, correlation, and R² for an equity."""
    combined = pd.DataFrame({"commodity": commodity_returns, "equity": equity_returns}).dropna()

    if len(combined) < 126:
        return None

    results = {}

    # Overall statistics
    corr = combined["commodity"].corr(combined["equity"])
    # OLS: equity = alpha + beta * commodity
    from numpy.polynomial.polynomial import polyfit
    beta, alpha = np.polyfit(combined["commodity"], combined["equity"], 1)
    residuals = combined["equity"] - (alpha + beta * combined["commodity"])
    ss_res = (residuals ** 2).sum()
    ss_tot = ((combined["equity"] - combined["equity"].mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    results["overall"] = {
        "correlation": round(float(corr), 4),
        "beta": round(float(beta), 4),
        "r_squared": round(float(r_squared), 4),
        "n_observations": len(combined),
    }

    # Rolling correlations
    for window in windows:
        rolling_corr = combined["commodity"].rolling(window).corr(combined["equity"])
        results[f"corr_{window}d"] = {
            "current": round(float(rolling_corr.iloc[-1]), 4) if len(rolling_corr) > 0 else 0,
            "mean": round(float(rolling_corr.mean()), 4),
            "std": round(float(rolling_corr.std()), 4),
        }

    # Forward-looking: does a commodity move predict equity move 1-5 days later?
    for lag in [1, 3, 5, 10, 21]:
        lagged_equity = combined["equity"].shift(-lag)
        valid = pd.DataFrame({"comm": combined["commodity"], "eq_fwd": lagged_equity}).dropna()
        if len(valid) > 50:
            lag_corr = valid["comm"].corr(valid["eq_fwd"])
            results[f"lead_lag_{lag}d"] = round(float(lag_corr), 4)

    return results


def scan_commodity(commodity: str) -> dict:
    """Scan all related equities for a single commodity."""
    print(f"\n{'='*60}")
    print(f"  {commodity.upper()} — Equity Correlation Scan")
    print(f"{'='*60}")

    # Load commodity data
    comm_path = COMMODITY_PATHS.get(commodity)
    if not comm_path or not comm_path.exists():
        print(f"  No commodity data found")
        return {}

    comm_df = pd.read_csv(comm_path, index_col=0, parse_dates=True)
    price_col = PRICE_COLS[commodity]
    comm_returns = comm_df[price_col].pct_change().dropna()

    # Fetch equity data
    equity_map = EQUITY_MAP.get(commodity, {})
    all_tickers = []
    for role_tickers in equity_map.values():
        all_tickers.extend(role_tickers)
    all_tickers = list(set(all_tickers))

    print(f"  Fetching {len(all_tickers)} equities...")
    equity_prices = fetch_equity_data(all_tickers)

    results = {}
    for role, tickers in equity_map.items():
        for ticker in tickers:
            if ticker not in equity_prices.columns:
                continue

            equity_ret = equity_prices[ticker].pct_change().dropna()
            beta_data = compute_commodity_beta(comm_returns, equity_ret)

            if beta_data is None:
                continue

            results[ticker] = {
                "role": role,
                "ticker": ticker,
                **beta_data,
            }

            overall = beta_data["overall"]
            lead_5d = beta_data.get("lead_lag_5d", 0)
            lead_21d = beta_data.get("lead_lag_21d", 0)
            print(f"  {ticker:<6} ({role:<10}): beta={overall['beta']:+.3f}, "
                  f"corr={overall['correlation']:+.3f}, R²={overall['r_squared']:.3f}, "
                  f"lead_5d={lead_5d:+.3f}, lead_21d={lead_21d:+.3f}")

    return results


def rank_opportunities(all_results: dict) -> list:
    """Rank equity opportunities by signal strength across all commodities."""
    opportunities = []

    for commodity, equities in all_results.items():
        for ticker, data in equities.items():
            overall = data["overall"]
            lead_21d = data.get("lead_lag_21d", 0)

            # Score: combination of R², absolute lead correlation, and beta
            score = (
                abs(overall["correlation"]) * 0.3 +
                overall["r_squared"] * 0.3 +
                abs(lead_21d) * 0.4  # lead-lag is most valuable
            )

            opportunities.append({
                "commodity": commodity,
                "ticker": ticker,
                "role": data["role"],
                "beta": overall["beta"],
                "correlation": overall["correlation"],
                "r_squared": overall["r_squared"],
                "lead_21d": lead_21d,
                "score": round(score, 4),
                "trade_direction": (
                    "SAME" if overall["beta"] > 0 else "INVERSE"
                ),
            })

    return sorted(opportunities, key=lambda x: x["score"], reverse=True)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("EQUITY-COMMODITY CORRELATION SCANNER")
    print("=" * 60)

    all_results = {}
    for commodity in EQUITY_MAP:
        results = scan_commodity(commodity)
        if results:
            all_results[commodity] = results

    # Rank opportunities
    ranked = rank_opportunities(all_results)

    print(f"\n{'='*60}")
    print("TOP EQUITY OPPORTUNITIES (ranked by predictive score)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Ticker':<7} {'Commodity':<12} {'Role':<10} {'Beta':>7} {'Corr':>7} {'R²':>6} {'Lead21d':>8} {'Direction':<10} {'Score':>6}")
    print("-" * 85)
    for i, opp in enumerate(ranked[:20]):
        print(f"{i+1:<5} {opp['ticker']:<7} {opp['commodity']:<12} {opp['role']:<10} "
              f"{opp['beta']:>+6.3f} {opp['correlation']:>+6.3f} {opp['r_squared']:>5.3f} "
              f"{opp['lead_21d']:>+7.3f} {opp['trade_direction']:<10} {opp['score']:>5.3f}")

    # Save results
    output = {
        "scan_date": str(pd.Timestamp.now().date()),
        "detailed_results": {k: {t: {kk: vv for kk, vv in v.items() if kk != "role"}
                                  for t, v in equities.items()}
                             for k, equities in all_results.items()},
        "ranked_opportunities": ranked[:20],
    }
    with open(DATA_DIR / "equity_scan.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {DATA_DIR / 'equity_scan.json'}")

    # Trading implications
    print(f"\n{'='*60}")
    print("TRADING IMPLICATIONS")
    print(f"{'='*60}")
    print("\nWhen commodity model fires a HIGH-CONFIDENCE signal:")
    for opp in ranked[:10]:
        direction = "BUY" if (opp["trade_direction"] == "SAME") else "SHORT"
        if opp["beta"] < 0:
            direction = "SHORT" if opp["trade_direction"] == "SAME" else "BUY"
        # Correct: producers benefit from rising prices, consumers suffer
        if opp["role"] == "producers" and opp["beta"] > 0:
            action_up, action_down = "BUY", "SHORT"
        elif opp["role"] == "consumers" and opp["beta"] < 0:
            action_up, action_down = "SHORT", "BUY"
        elif opp["role"] == "producers":
            action_up, action_down = "BUY", "SHORT"
        else:
            action_up, action_down = "SHORT", "BUY"

        print(f"  {opp['commodity'].upper():>12} UP   → {action_up:<6} {opp['ticker']:<6} (beta={opp['beta']:+.3f}, lead={opp['lead_21d']:+.3f})")
        print(f"  {opp['commodity'].upper():>12} DOWN → {action_down:<6} {opp['ticker']:<6}")


if __name__ == "__main__":
    main()
