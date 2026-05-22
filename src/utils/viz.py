'''Matplotlib-only plotting helpers for training curves and confusions.'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def plot_training_curve(log_json: str | Path, out_png: str | Path) -> Path:
    '''Plot loss + accuracy curves from a meta_trainer ``training.json`` log.

    The log is expected to have an ``epochs`` list of dicts with at least
    ``epoch``, ``train_loss``, ``train_acc``, and optionally ``val_acc``.
    Missing keys are skipped silently.
    '''
    import matplotlib.pyplot as plt  # noqa: PLC0415

    log_path = Path(log_json)
    payload: dict[str, Any] = json.loads(log_path.read_text())
    epochs = payload.get('epochs', [])
    if not epochs:
        raise ValueError(f'no epochs found in {log_path}')

    xs = [int(e['epoch']) for e in epochs]
    train_loss = [float(e.get('train_loss', float('nan'))) for e in epochs]
    train_acc = [float(e.get('train_acc', float('nan'))) for e in epochs]
    val_acc = [float(e['val_acc']) if 'val_acc' in e else float('nan') for e in epochs]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(xs, train_loss, label='train_loss', color='tab:red')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('Training loss')
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(xs, train_acc, label='train_acc', color='tab:blue')
    if not np.all(np.isnan(val_acc)):
        ax_acc.plot(xs, val_acc, label='val_acc', color='tab:green')
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_confusion(df: pd.DataFrame, out_png: str | Path, normalize: bool = True) -> Path:
    '''Render a confusion matrix DataFrame as a PNG heatmap.'''
    import matplotlib.pyplot as plt  # noqa: PLC0415

    matrix = df.to_numpy(dtype=np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(df.columns), 1.0 + 0.55 * len(df.index)))
    im = ax.imshow(matrix, cmap='Blues', vmin=0.0, vmax=1.0 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f'{value:.2f}' if normalize else f'{int(value)}'
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=8)

    fig.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


__all__ = ['plot_training_curve', 'plot_confusion']
