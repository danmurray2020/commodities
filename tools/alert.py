"""Alert system — checks for trading signals and sends notifications.

Run weekly (e.g., Saturday morning via cron) after refreshing data.
Supports: terminal output, macOS notifications, and writing to a log file.
"""

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"
SUGAR_DIR = Path(__file__).parent.parent / "sugar"
NATGAS_DIR = Path(__file__).parent.parent / "natgas"
SOYBEANS_DIR = Path(__file__).parent.parent / "soybeans"
WHEAT_DIR = Path(__file__).parent.parent / "wheat"
COPPER_DIR = Path(__file__).parent.parent / "copper"
ALERT_LOG = Path(__file__).parent / "alerts.log"
CONFIG_FILE = Path(__file__).parent / "optimal_config.json"


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def check_commodity(project_dir: Path, name: str, ticker: str, price_col: str) -> dict | None:
    """Check a single commodity for signals."""
    import os
    orig_dir = os.getcwd()
    os.chdir(project_dir)

    try:
        if "features" in sys.modules:
            del sys.modules["features"]
        sys.path.insert(0, str(project_dir))

        from features import prepare_dataset
        import joblib

        models_dir = project_dir / "models"
        for meta_file in ["v2_production_metadata.json", "production_metadata.json"]:
            if (models_dir / meta_file).exists():
                with open(models_dir / meta_file) as f:
                    meta = json.load(f)
                break
        else:
            return None

        version = "v2" if "v2" in meta_file else "v1"
        reg_file = f"v2_production_regressor.joblib" if version == "v2" else "production_regressor.joblib"
        clf_file = f"v2_production_classifier.joblib" if version == "v2" else "production_classifier.joblib"

        reg = joblib.load(models_dir / reg_file)
        clf = joblib.load(models_dir / clf_file)

        df, all_cols = prepare_dataset(horizon=63)
        feature_cols = [f for f in meta["features"] if f in all_cols]
        latest = df.iloc[[-1]]
        X = latest[feature_cols].values

        pred_return = float(reg.predict(X)[0])
        pred_dir = int(clf.predict(X)[0])
        pred_proba = clf.predict_proba(X)[0]
        confidence = float(pred_proba[pred_dir])
        current_price = float(latest[price_col].values[0])
        predicted_price = current_price * (1 + pred_return)
        as_of = latest.index[0].strftime("%Y-%m-%d")

        strategy = meta.get("strategy", {"confidence_threshold": 0.70, "stop_loss_pct": 0.10})
        threshold = strategy.get("confidence_threshold", 0.70)

        # Data staleness
        data_age = (datetime.now() - latest.index[0]).days

        return {
            "name": name, "ticker": ticker,
            "as_of": as_of, "data_age_days": data_age,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_return": pred_return,
            "direction": "UP" if pred_dir == 1 else "DOWN",
            "confidence": confidence,
            "threshold": threshold,
            "signal": confidence >= threshold,
            "action": ("LONG" if pred_dir == 1 else "SHORT") if confidence >= threshold else "NO TRADE",
        }
    except Exception as e:
        print(f"  ERROR checking {name}: {e}")
        return None
    finally:
        os.chdir(orig_dir)


def send_macos_notification(title: str, message: str):
    """Send a macOS notification via osascript."""
    try:
        # Escape double quotes to prevent AppleScript injection
        safe_title = title.replace('"', '\\"').replace("\\", "\\\\")
        safe_message = message.replace('"', '\\"').replace("\\", "\\\\")
        subprocess.run([
            "osascript", "-e",
            f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'
        ], capture_output=True, timeout=5)
    except Exception:
        pass


def main():
    print(f"{'='*60}")
    print(f"COMMODITIES ALERT CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    commodities = [
        (COFFEE_DIR, "Coffee", "KC=F", "coffee_close"),
        (COCOA_DIR, "Cocoa", "CC=F", "cocoa_close"),
        (SUGAR_DIR, "Sugar", "SB=F", "sugar_close"),
        (NATGAS_DIR, "Natural Gas", "NG=F", "natgas_close"),
        (SOYBEANS_DIR, "Soybeans", "ZS=F", "soybeans_close"),
        (WHEAT_DIR, "Wheat", "ZW=F", "wheat_close"),
        (COPPER_DIR, "Copper", "HG=F", "copper_close"),
    ]

    alerts = []
    warnings = []
    config = load_config()

    for project_dir, name, ticker, price_col in commodities:
        print(f"Checking {name} ({ticker})...")
        result = check_commodity(project_dir, name, ticker, price_col)

        if result is None:
            warnings.append(f"{name}: Could not load model")
            print(f"  ERROR: Could not load model\n")
            continue

        # Use calibrated threshold
        commodity_cfg = config.get(name.lower(), {})
        calibrated_threshold = commodity_cfg.get("confidence_threshold", 0.75)
        result["threshold"] = calibrated_threshold
        result["signal"] = result["confidence"] >= calibrated_threshold

        # Data staleness warning
        if result["data_age_days"] > 7:
            warnings.append(f"{name}: Data is {result['data_age_days']} days old (last: {result['as_of']})")

        action = result["action"] if result["signal"] else "NO TRADE"
        print(f"  Price:      ${result['current_price']:.2f}")
        print(f"  Prediction: ${result['predicted_price']:.2f} ({result['predicted_return']:+.1%})")
        print(f"  Direction:  {result['direction']} (confidence: {result['confidence']:.1%})")
        print(f"  Threshold:  {calibrated_threshold:.0%} (calibrated)")
        print(f"  Action:     {action}")

        if result["signal"]:
            alert_msg = (
                f"{action} {name} @ ${result['current_price']:.2f} → "
                f"${result['predicted_price']:.2f} ({result['predicted_return']:+.1%}), "
                f"conf: {result['confidence']:.0%}"
            )
            alerts.append(alert_msg)
            print(f"  ** SIGNAL FIRED **")
        print()

    # Summary
    print("=" * 60)
    if alerts:
        print("ACTIVE SIGNALS:")
        for alert in alerts:
            print(f"  {alert}")

        # macOS notification
        title = f"{len(alerts)} Trading Signal{'s' if len(alerts) > 1 else ''}"
        body = "\n".join(alerts)
        send_macos_notification(title, alerts[0])  # first signal in notification
    else:
        print("No active signals. Check again next week.")

    if warnings:
        print(f"\nWARNINGS:")
        for w in warnings:
            print(f"  {w}")

    # Log to file
    with open(ALERT_LOG, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        if alerts:
            for a in alerts:
                f.write(f"SIGNAL: {a}\n")
        else:
            f.write("No signals.\n")
        if warnings:
            for w in warnings:
                f.write(f"WARNING: {w}\n")

    print(f"\nLogged to {ALERT_LOG}")


if __name__ == "__main__":
    main()
