"""Shared Kelly criterion position sizing.

Single implementation used by both per-commodity strategy backtests and the
agents/strategy.py trade plan generator.
"""


def compute_kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """Compute fractional Kelly criterion position size.

    Kelly f* = (p * b - q) / b
    where p = win_rate, q = 1 - p, b = avg_win / abs(avg_loss)

    Args:
        win_rate: Fraction of winning trades (0-1).
        avg_win: Mean profit on winning trades (positive).
        avg_loss: Mean loss on losing trades (absolute value is used).
        fraction: Kelly fraction (0.5 = half-Kelly, the default).

    Returns:
        Position size as a fraction of equity, clamped to [0, 1].
    """
    if avg_loss == 0 or win_rate <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    return max(0.0, min(kelly * fraction, 1.0))
