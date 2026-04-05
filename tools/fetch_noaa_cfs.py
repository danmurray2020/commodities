"""Fetch NOAA Climate Forecast System (CFS) seasonal forecast data.

Retrieves temperature and precipitation forecasts from the IRI Data Library
for key agricultural growing regions, and saves to commodity data directories.

Fallback: if the IRI endpoint is unavailable, fetches NOAA CPC seasonal
outlooks instead.

Usage:
    python3 tools/fetch_noaa_cfs.py              # fetch all regions
    python3 tools/fetch_noaa_cfs.py --commodities soybeans wheat
"""

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Add repo root so we can import the retry utility
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.retry import retry_with_backoff

# ---------------------------------------------------------------------------
# Region definitions for key growing areas
# ---------------------------------------------------------------------------
REGIONS = {
    "brazil_central": {
        "label": "Brazil (Central)",
        "lat_min": -25,
        "lat_max": -15,
        "lon_min": -55,
        "lon_max": -45,
        "commodities": ["soybeans", "coffee", "sugar"],
    },
    "us_midwest": {
        "label": "US Midwest",
        "lat_min": 35,
        "lat_max": 45,
        "lon_min": -95,
        "lon_max": -82,
        "commodities": ["soybeans", "wheat"],
    },
    "west_africa": {
        "label": "West Africa",
        "lat_min": 5,
        "lat_max": 10,
        "lon_min": -10,
        "lon_max": 5,
        "commodities": ["chocolate"],
    },
}

# Commodity -> data directory name
COMMODITY_DIRS = {
    "soybeans": "soybeans",
    "wheat": "wheat",
    "coffee": "coffee",
    "sugar": "sugar",
    "chocolate": "chocolate",
    "natgas": "natgas",
    "copper": "copper",
}

# ---------------------------------------------------------------------------
# IRI Data Library helpers
# ---------------------------------------------------------------------------
IRI_BASE = "https://iridl.ldeo.columbia.edu"

# CFS monthly forecast — spatial average over a bounding box
# Variables: surface temp (tmp2m) and precip rate (prate)
IRI_CFS_TEMP_URL = (
    "{base}/SOURCES/.NOAA/.NCEP/.CFS/.DepIC/.realtime_mean/.ensemble24"
    "/.MONTHLY/.tmp2m"
    "/Y/{lat_min}/{lat_max}/RANGEEDGES"
    "/X/{lon_min}/{lon_max}/RANGEEDGES"
    "/[X+Y]average"
    "/L/1/3/RANGEEDGES"
    "/data.csv"
)

IRI_CFS_PRECIP_URL = (
    "{base}/SOURCES/.NOAA/.NCEP/.CFS/.DepIC/.realtime_mean/.ensemble24"
    "/.MONTHLY/.prate"
    "/Y/{lat_min}/{lat_max}/RANGEEDGES"
    "/X/{lon_min}/{lon_max}/RANGEEDGES"
    "/[X+Y]average"
    "/L/1/3/RANGEEDGES"
    "/data.csv"
)


@retry_with_backoff(max_retries=2, base_delay=5.0)
def _http_get(url: str, timeout: int = 90) -> requests.Response:
    """GET with retry."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def fetch_iri_variable(
    var_url_template: str,
    region_name: str,
    region: dict,
    var_label: str,
) -> pd.DataFrame | None:
    """Fetch a single CFS variable for a region from IRI Data Library.

    Returns a small DataFrame with columns [date, lead, value] or None on failure.
    """
    url = var_url_template.format(
        base=IRI_BASE,
        lat_min=region["lat_min"],
        lat_max=region["lat_max"],
        lon_min=region["lon_min"],
        lon_max=region["lon_max"],
    )
    print(f"    Fetching {var_label} for {region_name} from IRI ...")
    try:
        resp = _http_get(url)
        # IRI CSV typically has header lines starting with the variable name
        text = resp.text
        df = pd.read_csv(io.StringIO(text), comment="#", na_values=["NaN"])
        if df.empty:
            print(f"      WARNING: empty response for {var_label}/{region_name}")
            return None
        df["region"] = region_name
        df["variable"] = var_label
        return df
    except Exception as e:
        print(f"      WARNING: IRI fetch failed for {var_label}/{region_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# CPC fallback — simpler text-based outlooks
# ---------------------------------------------------------------------------
CPC_TEMP_OUTLOOK_URL = (
    "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/"
    "lead01/off01_temp.txt"
)
CPC_PRECIP_OUTLOOK_URL = (
    "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/"
    "lead01/off01_prcp.txt"
)


def fetch_cpc_outlook(url: str, label: str) -> pd.DataFrame | None:
    """Fetch a CPC text outlook and parse into a simple DataFrame."""
    print(f"  Fetching CPC {label} outlook (fallback) ...")
    try:
        resp = _http_get(url, timeout=30)
        lines = resp.text.strip().splitlines()
        # CPC outlooks are whitespace-delimited grids with lat/lon headers.
        # We do best-effort parsing: extract numeric rows.
        rows = []
        for line in lines:
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) >= 3:
                rows.append(nums)
        if not rows:
            print(f"    WARNING: could not parse CPC {label} outlook")
            return None
        # Use first column as lat proxy, second as lon proxy, rest as values
        max_cols = max(len(r) for r in rows)
        col_names = ["col_" + str(i) for i in range(max_cols)]
        df = pd.DataFrame(rows, columns=col_names[: len(rows[0])])
        df["source"] = f"CPC_{label}"
        df["fetched"] = datetime.utcnow().isoformat()
        return df
    except Exception as e:
        print(f"    WARNING: CPC {label} fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def fetch_region_forecasts(region_name: str, region: dict) -> pd.DataFrame:
    """Fetch CFS forecasts for a single region; fall back to CPC on failure."""
    frames: list[pd.DataFrame] = []

    # Try IRI CFS first
    temp_df = fetch_iri_variable(IRI_CFS_TEMP_URL, region_name, region, "temperature")
    if temp_df is not None:
        frames.append(temp_df)

    precip_df = fetch_iri_variable(IRI_CFS_PRECIP_URL, region_name, region, "precipitation")
    if precip_df is not None:
        frames.append(precip_df)

    # If IRI yielded nothing, try CPC fallback
    if not frames:
        print(f"    IRI unavailable for {region_name}, trying CPC fallback ...")
        for url, label in [
            (CPC_TEMP_OUTLOOK_URL, "temperature"),
            (CPC_PRECIP_OUTLOOK_URL, "precipitation"),
        ]:
            cpc_df = fetch_cpc_outlook(url, label)
            if cpc_df is not None:
                cpc_df["region"] = region_name
                frames.append(cpc_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["fetched_utc"] = datetime.utcnow().isoformat()
    return combined


def save_forecasts(df: pd.DataFrame, commodity: str) -> None:
    """Save forecast DataFrame to a commodity's data directory."""
    dir_name = COMMODITY_DIRS.get(commodity)
    if dir_name is None:
        return
    out_dir = REPO_ROOT / dir_name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "noaa_cfs_forecast.csv"

    # Append if file exists to avoid losing older fetches
    if out_path.exists():
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        df.drop_duplicates(inplace=True)

    df.to_csv(out_path, index=False)
    print(f"    Saved {len(df)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NOAA CFS seasonal forecast data for commodity growing regions"
    )
    parser.add_argument(
        "--commodities",
        nargs="+",
        default=list(COMMODITY_DIRS.keys()),
        help="Commodities to fetch forecasts for (default: all)",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(REGIONS.keys()),
        help="Regions to fetch (default: all)",
    )
    args = parser.parse_args()

    target_commodities = set(c.lower() for c in args.commodities)
    target_regions = set(args.regions)

    print("Fetching NOAA CFS seasonal forecasts")
    print(f"  Commodities: {', '.join(sorted(target_commodities))}")
    print(f"  Regions:     {', '.join(sorted(target_regions))}")
    print()

    # Collect forecasts per commodity
    commodity_data: dict[str, list[pd.DataFrame]] = {c: [] for c in target_commodities}

    for region_name, region in REGIONS.items():
        if region_name not in target_regions:
            continue
        print(f"[{region['label']}]")
        df = fetch_region_forecasts(region_name, region)
        if df.empty:
            print(f"  WARNING: no data for {region_name}\n")
            continue

        # Distribute to relevant commodities
        for commodity in region["commodities"]:
            if commodity in target_commodities:
                commodity_data[commodity].append(df)
        print()

    # Save results
    print("Saving results ...")
    for commodity, frames in commodity_data.items():
        if not frames:
            print(f"  {commodity}: no data to save")
            continue
        combined = pd.concat(frames, ignore_index=True)
        save_forecasts(combined, commodity)

    print("\nDone.")


if __name__ == "__main__":
    main()
