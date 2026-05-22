'''Metrics for episodic Few-Shot Action Recognition evaluation.

The two staples of FSAR reporting:

- ``accuracy_with_ci``: mean per-episode accuracy with a Student-t 95% CI,
  the convention used since Snell et al. (Prototypical Networks).
- ``per_class_confusion``: confusion matrix as a labeled DataFrame for plots
  and tables.
'''
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def accuracy_with_ci(
    accuracies: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    '''Return ``(mean, half_width)`` for the given per-episode accuracies.

    The half-width is ``t · std / sqrt(n)`` using a Student-t critical value
    with ``n - 1`` degrees of freedom (falls back to z=1.96 if scipy is
    unavailable). Returns ``(mean, 0.0)`` when ``n <= 1``.
    '''
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be in (0, 1)')
    n = len(accuracies)
    if n == 0:
        return 0.0, 0.0
    arr = np.asarray(accuracies, dtype=np.float64)
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0
    try:
        from scipy.stats import t as _t  # noqa: PLC0415

        t_val = float(_t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except ImportError:
        t_val = 1.96 if abs(confidence - 0.95) < 1e-6 else 2.576
    sem = float(arr.std(ddof=1)) / math.sqrt(n)
    return mean, t_val * sem


def per_class_confusion(
    predictions: Sequence[int],
    labels: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    '''Confusion matrix as a DataFrame with class names on both axes.

    Rows are true labels, columns are predicted labels. The label/prediction
    indices must be in ``[0, len(class_names))``.
    '''
    n_classes = len(class_names)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for pred, lbl in zip(predictions, labels, strict=True):
        i, j = int(lbl), int(pred)
        if not 0 <= i < n_classes or not 0 <= j < n_classes:
            raise ValueError(f'label/prediction out of range: label={lbl}, pred={pred}')
        matrix[i, j] += 1
    return pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))


__all__ = ['accuracy_with_ci', 'per_class_confusion']
