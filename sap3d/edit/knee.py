"""The knee rule that turns a probe's score matrix into a set of latent tokens.

Each row (one nominating source: a 3D query point or a conditioning-image patch) is scaled onto 0..1,
sorted descending, smoothed by a centered moving average, and its Kneedle bend located. The row admits every
token whose scaled score is at least `factor` times the scaled value at the bend and above the row's
minimum. A row with no bend admits nothing. The region is the union over rows.
"""

import numpy as np

# A row whose span is this small next to its own magnitude carries no bend.
FLAT_SPAN_FRACTION = 1e-12


def moving_average(rows: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average along the last axis, the window shrinking at both ends."""
    if width <= 1:
        return rows
    n = rows.shape[-1]
    prefix = np.concatenate([np.zeros(rows.shape[:-1] + (1,)), np.cumsum(rows, axis=-1)], axis=-1)
    position = np.arange(n)
    low = np.maximum(position - width // 2, 0)
    high = np.minimum(position + width // 2 + 1, n)
    return (prefix[..., high] - prefix[..., low]) / (high - low)


def bend_indices(sorted_desc: np.ndarray, width: int, sensitivity: float) -> tuple:
    """(Kneedle bend index per row, flat-row mask) for a descending-sorted score matrix.

    A fall below the running maximum of the gap confirms a bend only once that maximum has been raised at
    least once after the first position; a flat row reports its last position.
    """
    rows = np.atleast_2d(np.asarray(sorted_desc, dtype=np.float64))
    n = rows.shape[-1]
    if n <= 1:
        return np.zeros(len(rows), dtype=np.intp), np.ones(len(rows), dtype=bool)
    if width >= n:
        raise ValueError(f"a smoothing window of {width} over a curve of {n} positions leaves no bend to find")

    smoothed = moving_average(rows, width)
    lowest = smoothed[:, -1:]
    span = smoothed[:, :1] - lowest
    magnitude = np.maximum(np.abs(smoothed[:, :1]), np.abs(lowest))
    flat = span[:, 0] <= FLAT_SPAN_FRACTION * magnitude[:, 0]

    scaled = (smoothed - lowest) / np.where(span > 0, span, 1.0)
    gap = (1.0 - np.arange(n) / (n - 1)) - scaled
    running_max = np.maximum.accumulate(gap, axis=-1)
    previous_max = np.concatenate([np.full((len(gap), 1), -np.inf), running_max[:, :-1]], axis=-1)
    raised = np.cumsum(gap > previous_max, axis=-1) > 1
    confirmed = ((running_max - gap) > sensitivity / (n - 1)) & raised
    first = np.where(confirmed.any(axis=-1), confirmed.argmax(axis=-1), n)
    before_first = np.where(np.arange(n)[None, :] < first[:, None], gap, -np.inf)
    return np.where(flat, n - 1, before_first.argmax(axis=-1)), flat


def unit_scaled(rows: np.ndarray) -> tuple:
    """(rows mapped onto 0..1, mask of rows too flat to carry a bend), the mask read off the unscaled rows."""
    lowest = rows.min(axis=1, keepdims=True)
    highest = rows.max(axis=1, keepdims=True)
    span = highest - lowest
    magnitude = np.maximum(np.abs(highest), np.abs(lowest))
    degenerate = span[:, 0] <= FLAT_SPAN_FRACTION * magnitude[:, 0]
    return (rows - lowest) / np.where(span > 0, span, 1.0), degenerate


def knee_region(scores: np.ndarray, factor: float, width: int, sensitivity: float) -> set:
    """0-based token indices admitted by any row of `scores` [num_sources, num_tokens] (non-negative)."""
    if not float(factor) > 0:
        raise ValueError(f"the knee factor must be positive, got {factor!r}")
    rows = np.atleast_2d(np.asarray(scores))
    if rows.ndim != 2:
        raise ValueError(f"scores must be [num_sources, num_tokens], got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("scores must be finite")
    if (rows < 0).any():
        raise ValueError("scores must be non-negative")

    unit, degenerate = unit_scaled(rows.astype(np.float64))
    ordered = -np.sort(-unit, axis=1)
    index, flat = bend_indices(ordered, width, sensitivity)
    thresholds = ordered[np.arange(len(ordered)), index] * float(factor)
    admitted = (unit >= thresholds[:, None]) & (unit > 0)
    admitted[flat | degenerate] = False
    return set(np.flatnonzero(admitted.sum(axis=0) >= 1).tolist())
