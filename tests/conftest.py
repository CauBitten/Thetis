'''Shared pytest fixtures for THETIS data tests.'''
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope='session')
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def dataset_root(repo_root: Path) -> Path:
    path = repo_root / 'dataset'
    if not path.exists():
        pytest.skip(f'dataset/ not present at {path}')
    return path


@pytest.fixture(scope='session')
def manifest_path(repo_root: Path) -> Path:
    path = repo_root / 'data' / 'processed' / 'manifest.csv'
    if not path.exists():
        pytest.skip(f'manifest not built; run `make preprocess` first ({path})')
    return path


@pytest.fixture(scope='session')
def manifest_df(manifest_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        manifest_path,
        dtype={'actor': str, 'action_code': str},
        keep_default_na=False,
    )


@pytest.fixture
def mock_dataset_tree(tmp_path: Path) -> Path:
    '''Synthetic THETIS-like tree (RGB+Depth, 2 classes, 2 actors, 2 sequences).'''
    layout = {
        'VIDEO_RGB': [
            ('backhand', 'p1_backhand_s1.avi'),
            ('backhand', 'p1_backhand_s2.avi'),
            ('backhand', 'p32_backhand_s1.avi'),
            ('smash', 'p1_smash_s1.avi'),
            ('smash', 'p32_smash_s1.avi'),
            ('smash', 'p32_smash_s2.avi'),
        ],
        'VIDEO_Depth': [
            ('backhand', 'p1_backhand_depth_s1.avi'),
            ('backhand', 'p32_backhand_depth_s1.avi'),
            ('smash', 'p1_smash_depth_s1.avi'),
            ('smash', 'p32_smash_depth_s1.avi'),
        ],
    }
    for modality, items in layout.items():
        for cls, fname in items:
            d = tmp_path / modality / cls
            d.mkdir(parents=True, exist_ok=True)
            (d / fname).write_bytes(b'FAKE_AVI_BYTES')
    return tmp_path
