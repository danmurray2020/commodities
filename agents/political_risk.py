"""Political risk features for commodities affected by political instability.

Cocoa is the primary target: ~70% of global supply comes from Ivory Coast and Ghana,
where election cycles, export bans, and conflict directly affect production and exports.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Known election dates for key cocoa-producing countries
# ---------------------------------------------------------------------------
ELECTIONS = {
    "cocoa": [
        # Ivory Coast presidential elections
        {"country": "CIV", "date": "2015-10-25", "type": "presidential"},
        {"country": "CIV", "date": "2020-10-31", "type": "presidential"},
        {"country": "CIV", "date": "2025-10-25", "type": "presidential"},
        # Ghana general elections
        {"country": "GHA", "date": "2016-12-07", "type": "general"},
        {"country": "GHA", "date": "2020-12-07", "type": "general"},
        {"country": "GHA", "date": "2024-12-07", "type": "general"},
    ],
}


def _days_to_nearest_election(dates: pd.DatetimeIndex, election_dates: list[str]) -> pd.Series:
    """Return signed days-to-next-election for each date.

    Positive = before election, negative = after most recent election (capped).
    We return the minimum absolute distance to any upcoming election.
    """
    e_dates = sorted(pd.to_datetime(election_dates))
    result = pd.Series(np.nan, index=dates)

    for idx_date in dates:
        # Find nearest future election
        future = [e for e in e_dates if e >= idx_date]
        if future:
            result.loc[idx_date] = (future[0] - idx_date).days
        else:
            # Past the last known election — use distance to most recent
            past = [e for e in e_dates if e < idx_date]
            if past:
                result.loc[idx_date] = -(idx_date - past[-1]).days
            else:
                result.loc[idx_date] = np.nan

    return result


def add_political_risk_features(df: pd.DataFrame, commodity: str = "cocoa") -> pd.DataFrame:
    """Add political risk features for commodities affected by political instability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    commodity : str
        Which commodity config to use. Currently supports "cocoa".

    Returns
    -------
    pd.DataFrame
        DataFrame with additional political-risk columns.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    elections = ELECTIONS.get(commodity)
    if elections is None:
        return df

    # ------------------------------------------------------------------
    # 1. Election cycle features
    # ------------------------------------------------------------------
    election_date_strs = [e["date"] for e in elections]
    election_dates = sorted(pd.to_datetime(election_date_strs))

    # days_to_election: days until next upcoming election (positive),
    # or days since last election (negative) if none upcoming
    df["days_to_election"] = _days_to_nearest_election(df.index, election_date_strs)

    # pre_election_window: 1 if within 90 days before any election
    pre_election = pd.Series(0, index=df.index)
    for edate in election_dates:
        mask = (df.index >= edate - pd.Timedelta(days=90)) & (df.index < edate)
        pre_election = pre_election | mask.astype(int)
    df["pre_election_window"] = pre_election.astype(int)

    # post_election_window: 1 if within 60 days after any election
    post_election = pd.Series(0, index=df.index)
    for edate in election_dates:
        mask = (df.index > edate) & (df.index <= edate + pd.Timedelta(days=60))
        post_election = post_election | mask.astype(int)
    df["post_election_window"] = post_election.astype(int)

    # ------------------------------------------------------------------
    # 2. Seasonal political risk (known patterns)
    # ------------------------------------------------------------------
    month = df.index.month

    # Ivory Coast cocoa season flags
    # Main crop: Oct–Mar (months 10,11,12,1,2,3)
    # Mid-crop: May–Aug (months 5,6,7,8)
    df["ivory_coast_cocoa_season"] = np.where(
        month.isin([10, 11, 12, 1, 2, 3]),
        1,   # main crop
        np.where(month.isin([5, 6, 7, 8]), -1, 0),  # mid-crop / off-season
    )

    # Export ban season: Ivory Coast historically implements export bans Oct–Dec
    df["export_ban_season"] = month.isin([10, 11, 12]).astype(int)

    # Political disruption during main crop is more impactful — interaction
    df["election_x_main_crop"] = df["pre_election_window"] * (df["ivory_coast_cocoa_season"] == 1).astype(int)

    # ------------------------------------------------------------------
    # 3. Proxy indicators from existing data
    # ------------------------------------------------------------------

    # West Africa energy cost proxy (higher oil → transport disruptions)
    oil_cols = [c for c in df.columns if "crude" in c.lower() or "oil" in c.lower()]
    if oil_cols:
        oil_col = oil_cols[0]
        oil = df[oil_col].astype(float)
        df["westafrica_energy_cost"] = oil.pct_change(21)  # 1-month change
        # Normalise to z-score over trailing year
        rolling_mean = df["westafrica_energy_cost"].rolling(252, min_periods=63).mean()
        rolling_std = df["westafrica_energy_cost"].rolling(252, min_periods=63).std()
        df["westafrica_energy_cost_z"] = (
            (df["westafrica_energy_cost"] - rolling_mean) / rolling_std
        )

    # COT speculative positioning extreme — supply anxiety proxy
    spec_cols = [c for c in df.columns if "spec" in c.lower() and "net" in c.lower()]
    if not spec_cols:
        # Fallback: look for any COT net positioning column
        spec_cols = [c for c in df.columns if "cot" in c.lower() and "net" in c.lower()]
    if spec_cols:
        spec_col = spec_cols[0]
        spec = df[spec_col].astype(float)
        spec_rank = spec.rolling(252, min_periods=63).rank(pct=True)
        df["cot_spec_extreme"] = np.where(
            spec_rank > 0.90, 1,
            np.where(spec_rank < 0.10, -1, 0),
        )
    else:
        # If no COT spec column found, create a placeholder of zeros
        df["cot_spec_extreme"] = 0

    return df
