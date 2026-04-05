"""Refresh all data sources."""
import subprocess, sys

def run(script):
    print(f"\n{"="*60}\nRunning {script}...\n{"="*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}")

def main():
    run("fetch_data.py")
    run("fetch_cot.py")
    run("fetch_weather.py")
    run("fetch_enso.py")

    # Validate data freshness
    from pathlib import Path
    from datetime import datetime
    import pandas as pd

    data_dir = Path(__file__).parent / "data"
    csv_path = data_dir / "combined_features.csv"
    warnings = []

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not df.empty:
            age_days = (datetime.now() - df.index[-1]).days
            print(f"\nData freshness: latest date = {df.index[-1].strftime('%Y-%m-%d')} ({age_days} days ago)")
            if age_days > 7:
                warnings.append(f"WARNING: Data is {age_days} days old — may be stale!")
            if age_days > 14:
                warnings.append("CRITICAL: Data older than 2 weeks — predictions unreliable!")
        else:
            warnings.append("WARNING: combined_features.csv is empty!")
    else:
        warnings.append("WARNING: combined_features.csv not found!")

    # Check supplementary data
    for name in ["weather.csv", "enso.csv"]:
        path = data_dir / name
        if path.exists():
            sub = pd.read_csv(path, index_col=0, parse_dates=True)
            if sub.empty:
                warnings.append(f"WARNING: {name} is empty!")
        else:
            warnings.append(f"WARNING: {name} not found!")

    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")

    print(f"\nAll data refreshed. Run 'python3 train.py' to train.")

if __name__ == "__main__":
    main()
