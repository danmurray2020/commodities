"""Diagnose why each model loses accuracy out-of-sample."""

import json, sys, subprocess, numpy as np, pandas as pd_top
from pathlib import Path

COMMODITIES = [
    ("coffee", Path(__file__).parent.parent / "coffee", "coffee_close"),
    ("cocoa", Path(__file__).parent.parent / "chocolate", "cocoa_close"),
    ("sugar", Path(__file__).parent.parent / "sugar", "sugar_close"),
    ("natgas", Path(__file__).parent.parent / "natgas", "natgas_close"),
    ("soybeans", Path(__file__).parent.parent / "soybeans", "soybeans_close"),
    ("wheat", Path(__file__).parent.parent / "wheat", "wheat_close"),
    ("copper", Path(__file__).parent.parent / "copper", "copper_close"),
]


OOS_SPLIT = (pd_top.Timestamp.now() - pd_top.DateOffset(years=1)).strftime("%Y-%m-%d")


def diagnose(name, project_dir, price_col):
    script = """
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

df = pd.read_csv("data/combined_features.csv", index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col="{price_col}", horizon=63)
df = df.dropna()

train = df[df.index < "{oos_split}"]
test = df[df.index >= "{oos_split}"]

if len(test) < 20:
    print(json.dumps({"error": "insufficient test data"}))
    sys.exit(0)

price_train = train["{price_col}"]
price_test = test["{price_col}"]

train_up = float(train["target_direction"].mean())
test_up = float(test["target_direction"].mean())
train_vol = float(price_train.pct_change().std() * np.sqrt(252))
test_vol = float(price_test.pct_change().std() * np.sqrt(252))

models_dir = "models"
for mf in ["v3_production_metadata.json", "v2_production_metadata.json", "production_metadata.json"]:
    try:
        with open(f"{models_dir}/{mf}") as f:
            meta = json.load(f)
        break
    except FileNotFoundError:
        continue

feature_cols = [f for f in meta["features"] if f in df.columns]
shifts = {}
for feat in feature_cols:
    tm = float(train[feat].mean())
    ts = float(train[feat].std())
    testm = float(test[feat].mean())
    if ts > 0:
        shifts[feat] = round((testm - tm) / ts, 2)

result = {
    "n_train": len(train), "n_test": len(test),
    "train_up": round(train_up, 2), "test_up": round(test_up, 2),
    "class_shift": round(abs(test_up - train_up), 2),
    "train_vol": round(train_vol, 3), "test_vol": round(test_vol, 3),
    "vol_ratio": round(test_vol / train_vol, 2) if train_vol > 0 else 0,
    "train_price_range": [round(float(price_train.min()), 2), round(float(price_train.max()), 2)],
    "test_price_range": [round(float(price_test.min()), 2), round(float(price_test.max()), 2)],
    "feature_shifts": dict(sorted(shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:8]),
    "n_features": len(feature_cols),
}
print(json.dumps(result))
""".replace("{price_col}", price_col).replace("{oos_split}", OOS_SPLIT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=120,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except:
        return None


def main():
    print("=" * 70)
    print("OOS ACCURACY DIAGNOSIS")
    print(f"Train: pre-{OOS_SPLIT} | Test: {OOS_SPLIT}+ | What changed?")
    print("=" * 70)

    oos_accs = {
        "coffee": 0.696, "cocoa": 0.599, "sugar": 0.788,
        "natgas": 0.431, "soybeans": 0.662, "wheat": 0.648, "copper": 0.629,
    }
    oos_hc = {
        "coffee": 0.78, "cocoa": 0.75, "sugar": 0.90,
        "natgas": None, "soybeans": 0.72, "wheat": 0.67, "copper": None,
    }

    for name, project_dir, price_col in COMMODITIES:
        r = diagnose(name, project_dir, price_col)
        if r is None:
            print(f"\n{name.upper()}: ERROR")
            continue

        oos = oos_accs.get(name, 0)
        hc = oos_hc.get(name)

        print(f"\n{name.upper()} — OOS: {oos:.0%}" + (f", high-conf: {hc:.0%}" if hc else ", no high-conf signals"))
        print(f"  Class balance: train={r['train_up']:.0%} UP → test={r['test_up']:.0%} UP (shift={r['class_shift']:.0%})")
        print(f"  Volatility:    train={r['train_vol']:.1%} → test={r['test_vol']:.1%} ({r['vol_ratio']}x)")
        print(f"  Price range:   train=${r['train_price_range'][0]}-${r['train_price_range'][1]} → test=${r['test_price_range'][0]}-${r['test_price_range'][1]}")

        # Diagnose root cause
        issues = []
        if r["class_shift"] > 0.15:
            issues.append(f"CLASS IMBALANCE: test is {r['test_up']:.0%} UP vs {r['train_up']:.0%} in training")
        if r["vol_ratio"] > 1.5:
            issues.append(f"VOLATILITY SPIKE: {r['vol_ratio']}x higher than training period")
        if r["vol_ratio"] < 0.7:
            issues.append(f"VOLATILITY COLLAPSE: {r['vol_ratio']}x lower than training")

        big_shifts = {k: v for k, v in r["feature_shifts"].items() if abs(v) > 1.5}
        if big_shifts:
            issues.append(f"FEATURE DRIFT: {len(big_shifts)} features >1.5σ from train mean")

        print(f"  Feature distribution shifts (σ from train):")
        for feat, shift in list(r["feature_shifts"].items())[:6]:
            flag = " !!!" if abs(shift) > 2 else " !" if abs(shift) > 1 else ""
            print(f"    {feat:<35} {shift:+.1f}σ{flag}")

        if issues:
            print(f"  ROOT CAUSES:")
            for issue in issues:
                print(f"    - {issue}")

        # Prescription
        print(f"  PRESCRIPTION:")
        if r["class_shift"] > 0.15:
            print(f"    → Retrain with scale_pos_weight tuned for recent class balance")
        if r["vol_ratio"] > 1.5:
            print(f"    → Add volatility regime as explicit feature, widen stops")
        if big_shifts:
            print(f"    → Features are out-of-distribution. Consider:")
            print(f"      - Normalizing features relative to rolling windows (not absolute values)")
            print(f"      - Dropping features with biggest drift")
            print(f"      - Retraining with 2024+ data included")
        if not issues:
            print(f"    → No obvious distribution shift — model may need more data from current regime")


if __name__ == "__main__":
    main()
