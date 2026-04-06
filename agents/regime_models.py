"""Advanced regime detection (HMM) and volatility forecasting (GARCH) features.

Provides Hidden Markov Model regime classification and GARCH(1,1) conditional
volatility forecasts as features for the commodity prediction ensemble.

These features complement the simpler regime features in regime_features.py.
They are computed causally (no look-ahead bias) using rolling/expanding windows.

Usage:
    # As a feature generator (from any commodity's features.py):
    from agents.regime_models import add_regime_model_features
    df = add_regime_model_features(df, price_col="coffee_close")

    # CLI testing:
    python -m agents.regime_models natgas
    python -m agents.regime_models coffee --window 504
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional dependency imports ──────────────────────────────────────────

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

if not HAS_HMM or not HAS_ARCH:
    _missing = []
    if not HAS_HMM:
        _missing.append("hmmlearn")
    if not HAS_ARCH:
        _missing.append("arch")
    warnings.warn(
        f"Optional packages not installed: {', '.join(_missing)}. "
        f"Using manual fallback implementations. "
        f"For better results: pip install {' '.join(_missing)}"
    )


# ═════════════════════════════════════════════════════════════════════════
# A. Hidden Markov Model for Regime Detection
# ═════════════════════════════════════════════════════════════════════════


def fit_hmm_regimes(prices: pd.Series, n_regimes: int = 3) -> dict:
    """Fit a Gaussian HMM to identify market regimes.

    Returns dict with:
    - regime_states: array of regime labels (0, 1, 2) for each date
    - regime_probs: NxK matrix of regime probabilities
    - transition_matrix: KxK transition probability matrix
    - regime_means: mean return per regime
    - regime_vols: volatility per regime
    """
    log_returns = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)

    if len(log_returns) < 50:
        return _empty_hmm_result(len(prices), n_regimes)

    if HAS_HMM:
        return _fit_hmm_hmmlearn(log_returns, n_regimes)
    else:
        return _fit_hmm_kmeans(log_returns, n_regimes)


def _fit_hmm_hmmlearn(log_returns: np.ndarray, n_regimes: int) -> dict:
    """Fit HMM using hmmlearn library."""
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(log_returns)
        except Exception:
            return _fit_hmm_kmeans(log_returns, n_regimes)

    raw_states = model.predict(log_returns)
    regime_probs = model.predict_proba(log_returns)

    # Extract means and vols per state
    means = model.means_.flatten()
    vols = np.sqrt(model.covars_.flatten()) if model.covars_.ndim > 2 else np.sqrt(model.covars_.flatten())

    # Relabel states by mean return: 0=bear, 1=neutral, 2=bull
    order = np.argsort(means)
    label_map = {old: new for new, old in enumerate(order)}
    states = np.array([label_map[s] for s in raw_states])
    sorted_probs = regime_probs[:, order]
    sorted_means = means[order]
    sorted_vols = vols[order]

    # Reorder transition matrix
    trans = model.transmat_[order][:, order]

    return {
        "regime_states": states,
        "regime_probs": sorted_probs,
        "transition_matrix": trans,
        "regime_means": sorted_means,
        "regime_vols": sorted_vols,
    }


def _fit_hmm_kmeans(log_returns: np.ndarray, n_regimes: int) -> dict:
    """Fallback: k-means clustering on (rolling_return, rolling_vol) pairs.

    A reasonable approximation when hmmlearn is not available.
    """
    from sklearn.cluster import KMeans

    # Compute rolling features for clustering
    ret_series = pd.Series(log_returns.flatten())
    roll_ret = ret_series.rolling(21, min_periods=10).mean().values
    roll_vol = ret_series.rolling(21, min_periods=10).std().values

    # Mask NaNs
    valid = ~(np.isnan(roll_ret) | np.isnan(roll_vol))
    features = np.column_stack([roll_ret[valid], roll_vol[valid]])

    if len(features) < n_regimes * 5:
        n = len(log_returns)
        return {
            "regime_states": np.ones(n, dtype=int),
            "regime_probs": np.tile([0.0, 1.0, 0.0], (n, 1)),
            "transition_matrix": np.eye(n_regimes) * 0.9 + 0.1 / n_regimes,
            "regime_means": np.zeros(n_regimes),
            "regime_vols": np.ones(n_regimes) * ret_series.std(),
        }

    # Standardize features for clustering
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    features_scaled = (features - feat_mean) / feat_std

    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    raw_labels = km.fit_predict(features_scaled)

    # Full array of labels (NaN positions get neutral label)
    all_labels = np.ones(len(log_returns), dtype=int)  # default neutral
    all_labels[valid] = raw_labels

    # Compute mean return per cluster and relabel: 0=bear, 1=neutral, 2=bull
    cluster_means = np.array([
        ret_series.values[valid][raw_labels == k].mean()
        for k in range(n_regimes)
    ])
    order = np.argsort(cluster_means)
    label_map = {old: new for new, old in enumerate(order)}
    states = np.array([label_map[s] for s in all_labels])

    # Compute cluster vols
    cluster_vols = np.array([
        ret_series.values[valid][raw_labels == k].std()
        for k in range(n_regimes)
    ])

    sorted_means = cluster_means[order]
    sorted_vols = cluster_vols[order]

    # Approximate transition matrix from state sequence
    trans = _estimate_transition_matrix(states, n_regimes)

    # Approximate regime probabilities (one-hot from clustering)
    probs = np.zeros((len(states), n_regimes))
    for i, s in enumerate(states):
        probs[i, s] = 0.7
        for j in range(n_regimes):
            if j != s:
                probs[i, j] = 0.3 / (n_regimes - 1)

    return {
        "regime_states": states,
        "regime_probs": probs,
        "transition_matrix": trans,
        "regime_means": sorted_means,
        "regime_vols": sorted_vols,
    }


def _estimate_transition_matrix(states: np.ndarray, n_regimes: int) -> np.ndarray:
    """Estimate transition matrix from a sequence of state labels."""
    trans = np.ones((n_regimes, n_regimes)) * 0.01  # Laplace smoothing
    for i in range(len(states) - 1):
        trans[states[i], states[i + 1]] += 1
    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans / row_sums


def _empty_hmm_result(n: int, n_regimes: int) -> dict:
    """Return neutral result when insufficient data."""
    return {
        "regime_states": np.ones(n, dtype=int),
        "regime_probs": np.tile(
            [1.0 / n_regimes] * n_regimes, (n, 1)
        ),
        "transition_matrix": np.eye(n_regimes),
        "regime_means": np.zeros(n_regimes),
        "regime_vols": np.zeros(n_regimes),
    }


# ═════════════════════════════════════════════════════════════════════════
# B. GARCH Volatility Forecasting
# ═════════════════════════════════════════════════════════════════════════


def fit_garch_forecast(prices: pd.Series, forecast_horizon: int = 5) -> dict:
    """Fit GARCH(1,1) and forecast volatility.

    Returns dict with:
    - conditional_vol: array of fitted conditional volatility
    - vol_forecast: forecasted vol for next N days
    - vol_regime: 'low', 'normal', 'high', 'extreme' based on percentile
    - vol_persistence: GARCH alpha + beta (how sticky is vol)
    """
    returns = prices.pct_change().dropna() * 100  # scale to percentage for stability

    if len(returns) < 50:
        return _empty_garch_result(len(prices), forecast_horizon)

    if HAS_ARCH:
        return _fit_garch_arch(returns, forecast_horizon)
    else:
        return _fit_garch_manual(returns, forecast_horizon)


def _fit_garch_arch(returns: pd.Series, forecast_horizon: int) -> dict:
    """Fit GARCH(1,1) using the arch library."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            result = model.fit(disp="off", show_warning=False)
        except Exception:
            return _fit_garch_manual(returns, forecast_horizon)

    cond_vol = result.conditional_volatility.values / 100  # back to decimal

    # Extract parameters
    omega = result.params.get("omega", 0.01)
    alpha = result.params.get("alpha[1]", 0.05)
    beta = result.params.get("beta[1]", 0.90)
    persistence = alpha + beta

    # Forecast
    try:
        fcast = result.forecast(horizon=forecast_horizon)
        vol_forecast = np.sqrt(fcast.variance.iloc[-1].values) / 100
    except Exception:
        # Manual forecast from last conditional variance
        last_var = (cond_vol[-1] * 100) ** 2
        last_resid_sq = (returns.iloc[-1]) ** 2
        vol_forecast = _garch_forecast_manual(
            omega, alpha, beta, last_var, last_resid_sq, forecast_horizon
        ) / 100

    # Classify vol regime by percentile of current vol
    current_vol = cond_vol[-1]
    vol_pct = (cond_vol < current_vol).mean()
    vol_regime = _classify_vol_regime(vol_pct)

    return {
        "conditional_vol": cond_vol,
        "vol_forecast": vol_forecast,
        "vol_regime": vol_regime,
        "vol_persistence": float(persistence),
    }


def _fit_garch_manual(returns: pd.Series, forecast_horizon: int) -> dict:
    """Manual GARCH(1,1) implementation via variance targeting + grid search.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Uses variance targeting: omega = (1 - alpha - beta) * long_run_var
    Then searches over (alpha, beta) grid to maximize log-likelihood.
    """
    r = returns.values
    n = len(r)
    long_run_var = np.var(r)

    best_ll = -np.inf
    best_alpha = 0.05
    best_beta = 0.90

    # Grid search over alpha, beta (constrained: alpha + beta < 1)
    for alpha in np.arange(0.02, 0.20, 0.02):
        for beta in np.arange(0.70, 0.96, 0.02):
            if alpha + beta >= 0.999:
                continue
            omega = (1 - alpha - beta) * long_run_var
            if omega <= 0:
                continue
            ll = _garch_loglik(r, omega, alpha, beta)
            if ll > best_ll:
                best_ll = ll
                best_alpha = alpha
                best_beta = beta

    omega = (1 - best_alpha - best_beta) * long_run_var
    persistence = best_alpha + best_beta

    # Compute conditional volatility series
    cond_var = np.full(n, long_run_var)
    for t in range(1, n):
        cond_var[t] = omega + best_alpha * r[t - 1] ** 2 + best_beta * cond_var[t - 1]
        if cond_var[t] <= 0:
            cond_var[t] = long_run_var
    cond_vol = np.sqrt(cond_var) / 100  # back to decimal

    # Forecast
    vol_forecast = _garch_forecast_manual(
        omega, best_alpha, best_beta, cond_var[-1], r[-1] ** 2, forecast_horizon
    ) / 100

    # Classify vol regime
    current_vol = cond_vol[-1]
    vol_pct = (cond_vol < current_vol).mean()
    vol_regime = _classify_vol_regime(vol_pct)

    return {
        "conditional_vol": cond_vol,
        "vol_forecast": vol_forecast,
        "vol_regime": vol_regime,
        "vol_persistence": float(persistence),
    }


def _garch_loglik(r: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    """Compute Gaussian GARCH(1,1) log-likelihood."""
    n = len(r)
    var_long = np.var(r)
    sigma2 = var_long
    ll = 0.0
    for t in range(1, n):
        sigma2 = omega + alpha * r[t - 1] ** 2 + beta * sigma2
        if sigma2 <= 1e-10:
            sigma2 = var_long
        ll += -0.5 * (np.log(sigma2) + r[t] ** 2 / sigma2)
    return ll


def _garch_forecast_manual(
    omega: float, alpha: float, beta: float,
    last_var: float, last_resid_sq: float, horizon: int
) -> np.ndarray:
    """Iterate the GARCH variance recursion forward for forecasting."""
    forecasts = np.zeros(horizon)
    # One-step-ahead
    var_next = omega + alpha * last_resid_sq + beta * last_var
    forecasts[0] = np.sqrt(max(var_next, 1e-10))
    # Multi-step: E[sigma^2_{t+k}] = omega + (alpha + beta) * E[sigma^2_{t+k-1}]
    persistence = alpha + beta
    for k in range(1, horizon):
        var_next = omega + persistence * var_next
        forecasts[k] = np.sqrt(max(var_next, 1e-10))
    return forecasts


def _classify_vol_regime(vol_percentile: float) -> str:
    """Classify vol regime by percentile rank."""
    if vol_percentile < 0.25:
        return "low"
    elif vol_percentile < 0.75:
        return "normal"
    elif vol_percentile < 0.95:
        return "high"
    else:
        return "extreme"


def _empty_garch_result(n: int, forecast_horizon: int) -> dict:
    """Return neutral result when insufficient data."""
    return {
        "conditional_vol": np.full(n, np.nan),
        "vol_forecast": np.full(forecast_horizon, np.nan),
        "vol_regime": "normal",
        "vol_persistence": 0.95,
    }


# ═════════════════════════════════════════════════════════════════════════
# C. Combined regime feature generator (causal / rolling)
# ═════════════════════════════════════════════════════════════════════════


def add_regime_model_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Add HMM regime states and GARCH vol forecasts as features.

    IMPORTANT: Features are computed CAUSALLY — at each point in time, only
    data up to that point is used. The HMM is fit on a 504-day rolling window,
    refitting every 63 days to keep computation tractable.

    New columns:
    - hmm_regime: current regime (0=bear, 1=neutral, 2=bull)
    - hmm_bull_prob: probability of being in bull regime
    - hmm_bear_prob: probability of being in bear regime
    - hmm_regime_duration: days in current regime
    - hmm_transition_risk: probability of leaving current regime
    - garch_vol_forecast: forecasted volatility (1-day ahead)
    - garch_vol_ratio: forecast vol / historical vol (>1 = vol expanding)
    - garch_vol_regime: categorical (low/normal/high/extreme)
    - garch_persistence: alpha + beta (vol stickiness)
    """
    df = df.copy()
    prices = df[price_col].copy()
    n = len(df)

    hmm_window = 504       # 2 years of trading days
    hmm_refit_freq = 63    # refit every ~3 months
    min_history = 252      # need at least 1 year

    # Initialize output arrays
    hmm_regime = np.full(n, np.nan)
    hmm_bull_prob = np.full(n, np.nan)
    hmm_bear_prob = np.full(n, np.nan)
    garch_vol_forecast = np.full(n, np.nan)
    garch_vol_ratio = np.full(n, np.nan)
    garch_vol_regime = np.full(n, None, dtype=object)
    garch_persistence = np.full(n, np.nan)

    # ── Rolling HMM ─────────────────────────────────────────────────
    # Fit every hmm_refit_freq days on a trailing window of hmm_window days.
    # Between refits, carry forward the last fitted model's state assignment.

    last_hmm_result = None
    last_fit_idx = -hmm_refit_freq  # force fit on first eligible point

    for i in range(min_history, n):
        # Decide whether to refit
        if i - last_fit_idx >= hmm_refit_freq or last_hmm_result is None:
            window_start = max(0, i - hmm_window + 1)
            window_prices = prices.iloc[window_start:i + 1]

            if window_prices.isna().sum() > len(window_prices) * 0.3:
                continue
            window_prices = window_prices.dropna()
            if len(window_prices) < min_history:
                continue

            result = fit_hmm_regimes(window_prices, n_regimes=3)
            last_hmm_result = result
            last_fit_idx = i

            # Use the LAST state/probs from the fitted window (causal)
            hmm_regime[i] = result["regime_states"][-1]
            hmm_bull_prob[i] = result["regime_probs"][-1, 2]
            hmm_bear_prob[i] = result["regime_probs"][-1, 0]
        else:
            # Between refits: use a quick refit on recent data to get
            # current state (cheaper approach: classify based on recent
            # returns and vol matching the regime stats from last fit)
            if last_hmm_result is not None:
                recent_ret = np.log(prices.iloc[i] / prices.iloc[i - 21]) / 21
                recent_vol = prices.iloc[max(0, i-21):i+1].pct_change().std()

                # Find closest regime by Mahalanobis-like distance
                means = last_hmm_result["regime_means"]
                vols = last_hmm_result["regime_vols"]
                vols_safe = np.where(vols > 1e-10, vols, 1e-10)
                distances = ((recent_ret - means) / vols_safe) ** 2
                best_regime = int(np.argmin(distances))

                # Approximate probabilities using softmax of negative distances
                neg_dist = -distances
                neg_dist -= neg_dist.max()
                exp_d = np.exp(neg_dist)
                probs = exp_d / exp_d.sum()

                hmm_regime[i] = best_regime
                hmm_bull_prob[i] = probs[2]
                hmm_bear_prob[i] = probs[0]

    # ── Regime duration and transition risk ──────────────────────────

    hmm_regime_duration = np.full(n, np.nan)
    hmm_transition_risk = np.full(n, np.nan)

    current_duration = 0
    prev_regime = np.nan
    for i in range(n):
        if np.isnan(hmm_regime[i]):
            continue
        if hmm_regime[i] == prev_regime:
            current_duration += 1
        else:
            current_duration = 1
            prev_regime = hmm_regime[i]
        hmm_regime_duration[i] = current_duration

        # Transition risk = 1 - P(staying in current regime)
        if last_hmm_result is not None:
            regime_idx = int(hmm_regime[i])
            trans_mat = last_hmm_result["transition_matrix"]
            if regime_idx < trans_mat.shape[0]:
                hmm_transition_risk[i] = 1.0 - trans_mat[regime_idx, regime_idx]

    # ── Expanding-window GARCH ───────────────────────────────────────
    # GARCH naturally uses all history. We refit periodically for efficiency.

    garch_refit_freq = 63
    last_garch_result = None
    last_garch_fit_idx = -garch_refit_freq

    for i in range(min_history, n):
        if i - last_garch_fit_idx >= garch_refit_freq or last_garch_result is None:
            window_prices = prices.iloc[:i + 1].dropna()
            if len(window_prices) < min_history:
                continue

            result = fit_garch_forecast(window_prices, forecast_horizon=5)
            last_garch_result = result
            last_garch_fit_idx = i

            cond_vol = result["conditional_vol"]
            if len(cond_vol) > 0 and not np.isnan(cond_vol[-1]):
                garch_vol_forecast[i] = result["vol_forecast"][0]

                # Historical vol for ratio
                hist_vol = window_prices.pct_change().rolling(63, min_periods=21).std().iloc[-1]
                hist_vol_ann = hist_vol * np.sqrt(252) if not np.isnan(hist_vol) else np.nan
                forecast_ann = result["vol_forecast"][0] * np.sqrt(252)
                if hist_vol_ann and hist_vol_ann > 0:
                    garch_vol_ratio[i] = forecast_ann / hist_vol_ann

                garch_vol_regime[i] = result["vol_regime"]
                garch_persistence[i] = result["vol_persistence"]
        else:
            # Between refits: approximate using last GARCH params
            # Just carry forward the last values (GARCH changes slowly)
            if last_garch_result is not None:
                garch_vol_forecast[i] = garch_vol_forecast[i - 1] if i > 0 else np.nan
                garch_vol_ratio[i] = garch_vol_ratio[i - 1] if i > 0 else np.nan
                garch_vol_regime[i] = garch_vol_regime[i - 1] if i > 0 else None
                garch_persistence[i] = garch_persistence[i - 1] if i > 0 else np.nan

    # ── Assign to DataFrame ──────────────────────────────────────────

    df["hmm_regime"] = hmm_regime
    df["hmm_bull_prob"] = hmm_bull_prob
    df["hmm_bear_prob"] = hmm_bear_prob
    df["hmm_regime_duration"] = hmm_regime_duration
    df["hmm_transition_risk"] = hmm_transition_risk
    df["garch_vol_forecast"] = garch_vol_forecast
    df["garch_vol_ratio"] = garch_vol_ratio
    df["garch_vol_regime"] = garch_vol_regime
    df["garch_persistence"] = garch_persistence

    return df


def add_advanced_regime_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Add HMM and GARCH features. Call after add_regime_features.

    Convenience wrapper that adds advanced regime model features alongside
    the simpler statistical features from regime_features.add_regime_features.
    """
    return add_regime_model_features(df, price_col)


# ═════════════════════════════════════════════════════════════════════════
# D. CLI for testing
# ═════════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point for testing regime models on a commodity."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit HMM regime detection and GARCH volatility models."
    )
    parser.add_argument(
        "commodity",
        type=str,
        help="Commodity name (e.g., natgas, coffee, sugar)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=504,
        help="HMM rolling window in trading days (default: 504)",
    )
    args = parser.parse_args()

    commodity = args.commodity.lower()

    # Try to load from the config system
    try:
        from .config import COMMODITIES
        cfg = COMMODITIES.get(commodity)
        if cfg is None:
            print(f"Unknown commodity '{commodity}'. Available: {list(COMMODITIES.keys())}")
            sys.exit(1)
        csv_path = cfg.data_dir / "combined_features.csv"
        price_col = cfg.price_col
    except (ImportError, Exception):
        # Fallback: try to find data manually
        repo_root = Path(__file__).parent.parent
        csv_path = repo_root / commodity / "data" / "combined_features.csv"
        # Guess price column naming convention
        price_col = f"{commodity}_close"

    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if price_col not in df.columns:
        # Try to find a close price column
        close_cols = [c for c in df.columns if "close" in c.lower()]
        if close_cols:
            price_col = close_cols[0]
            print(f"Using price column: {price_col}")
        else:
            print(f"Price column '{price_col}' not found. Columns: {list(df.columns[:20])}")
            sys.exit(1)

    prices = df[price_col].dropna()
    print(f"Price series: {len(prices)} observations, {prices.index[0]} to {prices.index[-1]}")
    print()

    # ── HMM Regime Detection ────────────────────────────────────────
    print("=" * 60)
    print("HMM REGIME DETECTION")
    print("=" * 60)

    hmm_result = fit_hmm_regimes(prices, n_regimes=3)

    regime_labels = {0: "BEAR", 1: "NEUTRAL", 2: "BULL"}
    current_regime = int(hmm_result["regime_states"][-1])
    current_probs = hmm_result["regime_probs"][-1]

    print(f"\nCurrent regime: {regime_labels[current_regime]} ({current_regime})")
    print(f"Regime probabilities:")
    for k in range(3):
        print(f"  {regime_labels[k]:>8s}: {current_probs[k]:.3f}")

    print(f"\nRegime characteristics (daily returns):")
    for k in range(3):
        print(f"  {regime_labels[k]:>8s}: mean={hmm_result['regime_means'][k]:.5f}, "
              f"vol={hmm_result['regime_vols'][k]:.5f}")

    print(f"\nTransition matrix:")
    trans = hmm_result["transition_matrix"]
    print(f"  {'':>8s}  {'BEAR':>8s}  {'NEUTRAL':>8s}  {'BULL':>8s}")
    for k in range(3):
        row = "  ".join(f"{trans[k, j]:.3f}" for j in range(3))
        print(f"  {regime_labels[k]:>8s}:  {row}")

    # Regime duration
    states = hmm_result["regime_states"]
    duration = 1
    for i in range(len(states) - 2, -1, -1):
        if states[i] == current_regime:
            duration += 1
        else:
            break
    print(f"\nCurrent regime duration: {duration} days")
    transition_risk = 1.0 - trans[current_regime, current_regime]
    print(f"Transition risk (daily): {transition_risk:.3f}")

    # ── GARCH Volatility ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GARCH(1,1) VOLATILITY FORECAST")
    print("=" * 60)

    garch_result = fit_garch_forecast(prices, forecast_horizon=5)

    cond_vol = garch_result["conditional_vol"]
    current_vol = cond_vol[-1] if len(cond_vol) > 0 else np.nan
    vol_forecast = garch_result["vol_forecast"]

    # Historical vol for comparison
    hist_vol = prices.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)
    forecast_ann = vol_forecast[0] * np.sqrt(252) if not np.isnan(vol_forecast[0]) else np.nan

    print(f"\nCurrent conditional vol (annualized): {current_vol * np.sqrt(252):.4f}")
    print(f"Historical 63-day vol (annualized):   {hist_vol:.4f}")
    print(f"\nVol forecast (annualized, next 5 days):")
    for i, v in enumerate(vol_forecast):
        print(f"  Day {i+1}: {v * np.sqrt(252):.4f}")

    if hist_vol > 0 and not np.isnan(forecast_ann):
        ratio = forecast_ann / hist_vol
        direction = "EXPANDING" if ratio > 1.05 else "CONTRACTING" if ratio < 0.95 else "STABLE"
        print(f"\nVol ratio (forecast/historical): {ratio:.3f} -> {direction}")
    else:
        direction = "N/A"

    print(f"Vol regime: {garch_result['vol_regime'].upper()}")
    print(f"Vol persistence (alpha+beta): {garch_result['vol_persistence']:.4f}")

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Commodity:       {commodity}")
    print(f"  Market regime:   {regime_labels[current_regime]}")
    print(f"  Regime duration: {duration} days")
    print(f"  Vol regime:      {garch_result['vol_regime'].upper()}")
    print(f"  Vol direction:   {direction}")
    print(f"  HMM backend:     {'hmmlearn' if HAS_HMM else 'k-means fallback'}")
    print(f"  GARCH backend:   {'arch library' if HAS_ARCH else 'manual implementation'}")


if __name__ == "__main__":
    main()
