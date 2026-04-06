#!/usr/bin/env python3
"""Scaffold new commodity directories with standard fetch_data, features, and train modules.

Usage:
    python tools/scaffold_commodity.py crude_oil gold silver corn
    python tools/scaffold_commodity.py --all   # scaffold all commodities without directories
"""

import argparse
import sys
import textwrap
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from agents.config import COMMODITIES


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def fetch_data_template(commodity_key: str, ticker: str, price_col: str, name: str) -> str:
    return textwrap.dedent(f'''\
        """Fetch {name} futures data and supplementary market data."""

        import sys
        import pandas as pd
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.retry import download_with_retry

        DATA_DIR = Path(__file__).parent / "data"


        def fetch_prices(ticker: str = "{ticker}", period: str = "10y") -> pd.DataFrame:
            print(f"Fetching {{ticker}} data for the last {{period}}...")
            df = download_with_retry(ticker, period=period)
            df.index.name = "Date"
            df = df.dropna()
            print(f"Fetched {{len(df)}} rows from {{df.index.min()}} to {{df.index.max()}}")
            return df


        def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
            tickers = {{
                "usd_index": "DX-Y.NYB",
                "sp500": "^GSPC",
                "us10y": "^TNX",
                "vix": "^VIX",
            }}
            supplementary = {{}}
            for name, ticker in tickers.items():
                print(f"Fetching {{name}} ({{ticker}})...")
                df = download_with_retry(ticker, period="10y")
                df.index.name = "Date"
                if not df.empty:
                    supplementary[name] = df[["Close"]].rename(columns={{"Close": name}})
            return supplementary


        def main():
            DATA_DIR.mkdir(exist_ok=True)
            prices = fetch_prices()
            prices.to_csv(DATA_DIR / "{commodity_key}_prices.csv")

            supplementary = fetch_supplementary_data()
            combined = prices[["Close"]].rename(columns={{"Close": "{price_col}"}})
            for name, df in supplementary.items():
                combined = combined.join(df, how="left")
            combined = combined.ffill(limit=7)
            combined.to_csv(DATA_DIR / "combined_features.csv")
            print(f"Saved combined features to {{DATA_DIR / 'combined_features.csv'}}")


        if __name__ == "__main__":
            main()
    ''')


def features_template(commodity_key: str, price_col: str, name: str) -> str:
    return textwrap.dedent(f'''\
        """Feature engineering for {name} price prediction."""

        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.regime_features import add_regime_features

        import pandas as pd
        import numpy as np
        import ta

        DATA_DIR = Path(__file__).parent / "data"


        def add_price_features(df: pd.DataFrame, price_col: str = "{price_col}") -> pd.DataFrame:
            df = df.copy()
            price = df[price_col]

            # Returns
            for lag in [1, 5, 10, 21]:
                df[f"return_{{lag}}d"] = price.pct_change(lag)

            # Moving averages
            for window in [5, 10, 21, 50, 200]:
                df[f"sma_{{window}}"] = price.rolling(window).mean()
                df[f"price_vs_sma_{{window}}"] = price / df[f"sma_{{window}}"] - 1

            # Volatility
            for window in [10, 21, 63]:
                df[f"volatility_{{window}}d"] = price.pct_change().rolling(window).std() * np.sqrt(252)

            # RSI
            df["rsi_14"] = ta.momentum.rsi(price, window=14)

            # MACD
            macd = ta.trend.MACD(price)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(price, window=20)
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["bb_pct"] = bb.bollinger_pband()

            # ATR
            if all(c in df.columns for c in ["High", "Low"]):
                df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], price, window=14)

            # Price lags
            for lag in [1, 2, 3, 5, 10]:
                df[f"price_lag_{{lag}}"] = price.shift(lag)

            # Seasonality
            if isinstance(df.index, pd.DatetimeIndex):
                df["day_of_week"] = df.index.dayofweek
                df["month"] = df.index.month
                day_of_year = df.index.dayofyear
                for harmonic in [1, 2]:
                    df[f"season_sin_{{harmonic}}"] = np.sin(2 * np.pi * harmonic * day_of_year / 365.25)
                    df[f"season_cos_{{harmonic}}"] = np.cos(2 * np.pi * harmonic * day_of_year / 365.25)

            # Z-scores (mean reversion)
            for window in [126, 252]:
                rm = price.rolling(window).mean()
                rs = price.rolling(window).std()
                df[f"zscore_{{window}}d"] = (price - rm) / rs
            if "zscore_252d" in df.columns:
                df["zscore_252d_change_21d"] = df["zscore_252d"].diff(21)
                df["extreme_high"] = (df["zscore_252d"] > 2).astype(int)
                df["extreme_low"] = (df["zscore_252d"] < -2).astype(int)

            # Trend features
            daily_ret = price.pct_change()
            for window in [21, 63, 126]:
                df[f"pct_up_days_{{window}}d"] = daily_ret.rolling(window).apply(lambda x: (x > 0).mean())
                df[f"trend_slope_{{window}}d"] = price.rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 1 else 0
                )
            if "sma_50" in df.columns and "sma_200" in df.columns:
                df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
                df["sma_50_200_gap"] = df["sma_50"] / df["sma_200"] - 1
            ret_21d = price.pct_change(21)
            df["momentum_rank_252d"] = ret_21d.rolling(252).rank(pct=True)

            return df


        def build_target(df, price_col="{price_col}", horizon=63):
            df = df.copy()
            future = df[price_col].shift(-horizon)
            df["target_return"] = np.log(future / df[price_col])
            df["target_direction"] = (df["target_return"] > 0).astype(int)
            return df


        def prepare_dataset(csv_path=str(DATA_DIR / "combined_features.csv"), horizon=63):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = add_price_features(df)
            df = add_regime_features(df, price_col="{price_col}")
            df = build_target(df, horizon=horizon)
            df = df.ffill()
            df = df.dropna()
            exclude = {{"{price_col}", "Open", "High", "Low", "Volume", "target_return", "target_direction"}}
            feature_cols = [c for c in df.columns if c not in exclude]
            return df, feature_cols
    ''')


def train_template(commodity_key: str, price_col: str, name: str, ticker: str) -> str:
    return textwrap.dedent(f'''\
        """Train and evaluate {name} price prediction models."""

        import json
        import sys
        import os
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
        from sklearn.inspection import permutation_importance
        from xgboost import XGBRegressor, XGBClassifier
        import optuna
        import joblib

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.train_utils import (
            walk_forward_split, evaluate_predictions, evaluate_classification,
            filter_stable_features, reg_objective_spearman, clf_objective_spearman,
        )

        from features import prepare_dataset

        MODELS_DIR = Path(__file__).parent / "models"
        HORIZON = int(os.environ.get("QUALITY_HORIZON", "63"))
        OPTUNA_TRIALS = int(os.environ.get("QUALITY_OPTUNA_TRIALS", "200"))
        TEST_SIZE = int(os.environ.get("QUALITY_TEST_SIZE", "252"))
        N_SPLITS = int(os.environ.get("QUALITY_N_SPLITS", "5"))
        MIN_GAMMA = float(os.environ.get("QUALITY_MIN_GAMMA", "0.0"))
        MIN_REG_ALPHA = float(os.environ.get("QUALITY_MIN_REG_ALPHA", "1e-8"))
        MIN_REG_LAMBDA = float(os.environ.get("QUALITY_MIN_REG_LAMBDA", "1e-8"))
        MAX_DEPTH_MIN = int(os.environ.get("QUALITY_MAX_DEPTH_MIN", "2"))
        MAX_DEPTH_MAX = int(os.environ.get("QUALITY_MAX_DEPTH_MAX", "10"))


        def select_features(df, feature_cols, splits):
            """Permutation importance feature selection."""
            X = df[feature_cols].values
            y = df["target_direction"].values
            all_importances = np.zeros((len(splits), len(feature_cols)))

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                print(f"  Feature selection fold {{fold_i}}...")
                val_size = min(63, len(train_idx) // 5)
                fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
                neg = np.sum(y[fit_idx] == 0)
                pos = np.sum(y[fit_idx] == 1)
                spw = neg / pos if pos > 0 else 1.0
                model = XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.6,
                    min_child_weight=10, gamma=1.0,
                    scale_pos_weight=spw,
                    eval_metric="logloss", early_stopping_rounds=30, random_state=42,
                )
                model.fit(X[fit_idx], y[fit_idx],
                          eval_set=[(X[val_idx], y[val_idx])], verbose=False)
                result = permutation_importance(
                    model, X[test_idx], y[test_idx],
                    n_repeats=10, random_state=42, scoring="accuracy",
                )
                all_importances[fold_i] = result.importances_mean

            mean_imp = all_importances.mean(axis=0)
            ranking = sorted(zip(feature_cols, mean_imp), key=lambda x: x[1], reverse=True)
            selected = [f for f, imp in ranking if imp > 0]
            if len(selected) < 3:
                selected = [f for f, _ in ranking[:max(10, len(ranking) // 4)]]
                print(f"  Warning: only {{len([f for f, imp in ranking if imp > 0])}} features had positive importance. Using top {{len(selected)}} instead.")

            print(f"\\n  Selected {{len(selected)}} features (from {{len(feature_cols)}}):")
            for f, imp in ranking[:20]:
                marker = " *" if imp > 0 else ""
                print(f"    {{f:<40}} {{imp:.4f}}{{marker}}")
            return selected


        def main():
            MODELS_DIR.mkdir(exist_ok=True)

            print("=" * 60)
            print("{name.upper()} FUTURES PREDICTION MODEL")
            print("=" * 60)

            print("\\nPreparing dataset...")
            df, all_feature_cols = prepare_dataset(horizon=HORIZON)

            selection_splits = walk_forward_split(df, n_splits=3, purge_gap=HORIZON)
            print(f"Dataset: {{len(df)}} rows, {{len(all_feature_cols)}} features")
            print(f"Feature selection: {{len(selection_splits)}} folds (separate from training)\\n")

            print("Running feature selection...")
            selected = select_features(df, all_feature_cols, selection_splits)

            print("\\nFiltering for feature stability...")
            stable_features, stability_diag = filter_stable_features(df, selected)
            n_dropped = len(selected) - len(stable_features)
            if n_dropped > 0:
                dropped = [d["feature"] for d in stability_diag if not d["stable"]]
                print(f"  Dropped {{n_dropped}} unstable features: {{dropped[:10]}}")
            if len(stable_features) >= 3:
                feature_cols = stable_features
            else:
                print("  Warning: too few stable features, using all selected features")
                feature_cols = selected

            splits = walk_forward_split(df, n_splits=N_SPLITS, test_size=TEST_SIZE, purge_gap=HORIZON)
            print(f"Training: {{len(splits)}} folds (full walk-forward CV)")
            X = df[feature_cols].values
            y_ret = df["target_return"].values
            y_dir = df["target_direction"].values

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            print(f"\\nTuning regression model ({{OPTUNA_TRIALS}} trials)...")
            reg_study = optuna.create_study(direction="maximize", study_name="{commodity_key}_reg")
            reg_study.optimize(lambda t: reg_objective_spearman(t, X, y_ret, splits, horizon=HORIZON), n_trials=OPTUNA_TRIALS)
            print(f"Best regression score: {{reg_study.best_value:.4f}}")

            print(f"\\nTuning classification model ({{OPTUNA_TRIALS}} trials)...")
            clf_study = optuna.create_study(direction="maximize", study_name="{commodity_key}_clf")
            clf_study.optimize(lambda t: clf_objective_spearman(t, X, y_dir, splits, horizon=HORIZON), n_trials=OPTUNA_TRIALS)
            print(f"Best classification score: {{clf_study.best_value:.4f}}")

            print("\\n" + "=" * 60)
            print("EVALUATION (purged walk-forward CV)")
            print("=" * 60)
            reg_params = {{**reg_study.best_params, "early_stopping_rounds": 30, "random_state": 42}}
            clf_params = {{**clf_study.best_params, "eval_metric": "logloss",
                          "early_stopping_rounds": 30, "random_state": 42}}

            reg_params["gamma"] = max(reg_params.get("gamma", 0), MIN_GAMMA)
            reg_params["reg_alpha"] = max(reg_params.get("reg_alpha", 0), MIN_REG_ALPHA)
            reg_params["reg_lambda"] = max(reg_params.get("reg_lambda", 0), MIN_REG_LAMBDA)
            reg_params["max_depth"] = min(reg_params.get("max_depth", 10), MAX_DEPTH_MAX)
            clf_params["gamma"] = max(clf_params.get("gamma", 0), MIN_GAMMA)
            clf_params["reg_alpha"] = max(clf_params.get("reg_alpha", 0), MIN_REG_ALPHA)
            clf_params["reg_lambda"] = max(clf_params.get("reg_lambda", 0), MIN_REG_LAMBDA)
            clf_params["max_depth"] = min(clf_params.get("max_depth", 10), MAX_DEPTH_MAX)

            reg_fold_spearman = []
            reg_fold_dir_acc = []
            reg_fold_maes = []
            reg_fold_rmses = []
            clf_fold_acc_ind = []
            for fold_i, (train_idx, test_idx) in enumerate(splits):
                val_size = min(63, len(train_idx) // 5)
                fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]

                reg = XGBRegressor(**reg_params)
                reg.fit(X[fit_idx], y_ret[fit_idx],
                        eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
                reg_preds = reg.predict(X[test_idx])
                reg_metrics = evaluate_predictions(y_ret[test_idx], reg_preds, horizon=HORIZON)
                reg_fold_spearman.append(reg_metrics["spearman"])
                reg_fold_dir_acc.append(reg_metrics["dir_acc_independent"])
                reg_fold_maes.append(reg_metrics["mae"])
                reg_fold_rmses.append(reg_metrics["rmse"])

                neg = np.sum(y_dir[fit_idx] == 0)
                pos = np.sum(y_dir[fit_idx] == 1)
                spw = neg / pos if pos > 0 else 1.0
                clf = XGBClassifier(**clf_params, scale_pos_weight=spw)
                clf.fit(X[fit_idx], y_dir[fit_idx],
                        eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
                clf_metrics = evaluate_classification(y_dir[test_idx], clf.predict(X[test_idx]), horizon=HORIZON)
                clf_fold_acc_ind.append(clf_metrics["acc_independent"])
                print(f"  Fold {{fold_i}}: Reg Spearman={{reg_metrics['spearman']:.4f}} DirAcc={{reg_metrics['dir_acc_independent']:.2%}} MAE={{reg_metrics['mae']:.4f}}, Clf={{clf_metrics['acc_independent']:.2%}}")

            print(f"\\n  REGRESSION  — Spearman: {{np.mean(reg_fold_spearman):.4f}}, DirAcc: {{np.mean(reg_fold_dir_acc):.2%}}, MAE: {{np.mean(reg_fold_maes):.4f}}, RMSE: {{np.mean(reg_fold_rmses):.4f}}")
            print(f"  CLASSIFIER  — Acc(ind): {{np.mean(clf_fold_acc_ind):.2%}}, Std: {{np.std(clf_fold_acc_ind):.2%}}")

            last_train, last_test = splits[-1]
            val_size = min(63, len(last_train) // 5)
            fit_idx, val_idx = last_train[:-val_size], last_train[-val_size:]
            final_reg = XGBRegressor(**reg_params)
            final_reg.fit(X[fit_idx], y_ret[fit_idx],
                          eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
            neg = np.sum(y_dir[fit_idx] == 0)
            pos = np.sum(y_dir[fit_idx] == 1)
            spw = neg / pos if pos > 0 else 1.0
            final_clf = XGBClassifier(**clf_params, scale_pos_weight=spw)
            final_clf.fit(X[fit_idx], y_dir[fit_idx],
                          eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)

            holdout_size = 126
            if len(df) > holdout_size + 504:
                holdout_idx = np.arange(len(df) - holdout_size, len(df))
                X_holdout = X[holdout_idx]
                y_ret_holdout = y_ret[holdout_idx]
                y_dir_holdout = y_dir[holdout_idx]

                holdout_reg_preds = final_reg.predict(X_holdout)
                holdout_reg_metrics = evaluate_predictions(y_ret_holdout, holdout_reg_preds, horizon=HORIZON)
                holdout_reg_acc = holdout_reg_metrics["dir_acc_independent"]
                holdout_reg_spearman = holdout_reg_metrics["spearman"]
                holdout_reg_mae = holdout_reg_metrics["mae"]

                holdout_clf_preds = final_clf.predict(X_holdout)
                holdout_clf_metrics = evaluate_classification(y_dir_holdout, holdout_clf_preds, horizon=HORIZON)
                holdout_clf_acc = holdout_clf_metrics["acc_independent"]

                print(f"\\n  HELD-OUT TEST ({{holdout_size}} days, never seen during CV):")
                print(f"    Regression Spearman: {{holdout_reg_spearman:.4f}}, DirAcc: {{holdout_reg_acc:.2%}}, MAE: {{holdout_reg_mae:.4f}}")
                print(f"    Classification(ind): {{holdout_clf_acc:.2%}}")
            else:
                holdout_reg_acc = None
                holdout_clf_acc = None
                holdout_reg_mae = None
                holdout_reg_spearman = None
                print("\\n  HELD-OUT TEST: Skipped (insufficient data)")

            joblib.dump(final_reg, MODELS_DIR / "production_regressor.joblib")
            joblib.dump(final_clf, MODELS_DIR / "production_classifier.joblib")

            strategy_config = {{
                "confidence_threshold": 0.70, "stop_loss_pct": 0.10,
                "take_profit_multiplier": 1.0, "max_hold_days": 63,
                "allow_short": True,
            }}

            metadata = {{
                "commodity": "{commodity_key}",
                "ticker": "{ticker}",
                "horizon": HORIZON,
                "features": feature_cols,
                "n_features": len(feature_cols),
                "purge_gap": HORIZON,
                "strategy": strategy_config,
                "regression": {{
                    "params": reg_study.best_params,
                    "fold_spearman": reg_fold_spearman,
                    "avg_spearman": float(np.mean(reg_fold_spearman)),
                    "fold_dir_acc_independent": reg_fold_dir_acc,
                    "avg_dir_acc_independent": float(np.mean(reg_fold_dir_acc)),
                    "fold_accuracies": reg_fold_dir_acc,
                    "avg_accuracy": float(np.mean(reg_fold_dir_acc)),
                    "std_accuracy": float(np.std(reg_fold_dir_acc)),
                    "fold_maes": reg_fold_maes,
                    "fold_rmses": reg_fold_rmses,
                    "avg_mae": float(np.mean(reg_fold_maes)),
                    "avg_rmse": float(np.mean(reg_fold_rmses)),
                }},
                "classification": {{
                    "params": clf_study.best_params,
                    "fold_acc_independent": clf_fold_acc_ind,
                    "avg_acc_independent": float(np.mean(clf_fold_acc_ind)),
                    "fold_accuracies": clf_fold_acc_ind,
                    "avg_accuracy": float(np.mean(clf_fold_acc_ind)),
                    "std_accuracy": float(np.std(clf_fold_acc_ind)),
                }},
                "holdout": {{
                    "size": holdout_size if len(df) > holdout_size + 504 else 0,
                    "reg_direction_accuracy": holdout_reg_acc,
                    "reg_spearman": holdout_reg_spearman,
                    "reg_mae": holdout_reg_mae,
                    "clf_accuracy": holdout_clf_acc,
                }},
            }}
            with open(MODELS_DIR / "production_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            print("\\n" + "=" * 60)
            print("{name.upper()} 63-DAY PREDICTION")
            print("=" * 60)
            latest = df.iloc[[-1]]
            X_latest = latest[feature_cols].values
            pred_return = float(final_reg.predict(X_latest)[0])
            pred_dir = int(final_clf.predict(X_latest)[0])
            pred_proba = final_clf.predict_proba(X_latest)[0]
            confidence = float(pred_proba[pred_dir])
            current_price = float(latest["{price_col}"].values[0])
            predicted_price = current_price * (1 + pred_return)

            print(f"Date: {{latest.index[0].strftime('%Y-%m-%d')}}")
            print(f"Current price: ${{current_price:.2f}}")
            print(f"Predicted 63-day return: {{pred_return:+.2%}}")
            print(f"Predicted price: ${{predicted_price:.2f}}")
            print(f"Direction: {{'UP' if pred_dir == 1 else 'DOWN'}} (confidence: {{confidence:.1%}})")

            if confidence >= strategy_config["confidence_threshold"]:
                direction = "LONG" if pred_dir == 1 else "SHORT"
                tp = abs(pred_return)
                sl = strategy_config["stop_loss_pct"]
                tp_price = current_price * (1 + tp) if direction == "LONG" else current_price * (1 - tp)
                sl_price = current_price * (1 - sl) if direction == "LONG" else current_price * (1 + sl)
                print(f"\\nStrategy: {{direction}}")
                print(f"  Entry: ${{current_price:.2f}}, TP: ${{tp_price:.2f}}, SL: ${{sl_price:.2f}}")
            else:
                print(f"\\nStrategy: NO TRADE (confidence {{confidence:.1%}} < 70%)")

            print(f"\\nModels saved to {{MODELS_DIR}}/")


        if __name__ == "__main__":
            main()
    ''')


# ---------------------------------------------------------------------------
# Scaffold logic
# ---------------------------------------------------------------------------

def scaffold_commodity(key: str, force: bool = False) -> bool:
    """Create directory structure and files for a commodity. Returns True if created."""
    if key not in COMMODITIES:
        print(f"  ERROR: '{key}' not found in COMMODITIES config. Skipping.")
        return False

    cfg = COMMODITIES[key]
    project_dir = cfg.project_dir

    if project_dir.exists() and not force:
        # Check if Python files exist — if not, still scaffold
        if (project_dir / "features.py").exists():
            print(f"  SKIP: {project_dir} already exists.")
            return False

    print(f"  Creating {project_dir}/")
    project_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.models_dir.mkdir(exist_ok=True)

    # fetch_data.py
    fetch_path = project_dir / "fetch_data.py"
    fetch_path.write_text(fetch_data_template(key, cfg.ticker, cfg.price_col, cfg.name))
    print(f"    wrote {fetch_path}")

    # features.py
    features_path = project_dir / "features.py"
    features_path.write_text(features_template(key, cfg.price_col, cfg.name))
    print(f"    wrote {features_path}")

    # train.py
    train_path = project_dir / "train.py"
    train_path.write_text(train_template(key, cfg.price_col, cfg.name, cfg.ticker))
    print(f"    wrote {train_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Scaffold new commodity directories")
    parser.add_argument("commodities", nargs="*", help="Commodity keys to scaffold (e.g. crude_oil gold)")
    parser.add_argument("--all", action="store_true", help="Scaffold all commodities without directories")
    args = parser.parse_args()

    if args.all:
        keys = list(COMMODITIES.keys())
    elif args.commodities:
        keys = args.commodities
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Scaffolding {len(keys)} commodities...\n")
    created = 0
    for key in keys:
        if scaffold_commodity(key):
            created += 1
    print(f"\nDone. Created {created} new commodity directories.")


if __name__ == "__main__":
    main()
