'''Tests for src/data/episode_sampler.py.'''
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.episode_sampler import EpisodeSampler, split_classes
from src.data.loader import ACTION_LABELS


# ---------------------------------------------------------------------------
# split_classes
# ---------------------------------------------------------------------------


def test_split_classes_disjoint() -> None:
    splits = split_classes(list(ACTION_LABELS), train_n=7, val_n=2, test_n=3, seed=42)
    assert sorted(splits) == ['meta_test', 'meta_train', 'meta_val']
    train, val, test = set(splits['meta_train']), set(splits['meta_val']), set(splits['meta_test'])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(ACTION_LABELS)
    assert (len(train), len(val), len(test)) == (7, 2, 3)


def test_split_classes_deterministic() -> None:
    a = split_classes(list(ACTION_LABELS), 7, 2, 3, seed=42)
    b = split_classes(list(ACTION_LABELS), 7, 2, 3, seed=42)
    assert a == b


def test_split_classes_invalid_sums() -> None:
    with pytest.raises(ValueError):
        split_classes(list(ACTION_LABELS), 5, 5, 5, seed=42)


# ---------------------------------------------------------------------------
# EpisodeSampler — needs a manifest. Skip when not built.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def splits_default() -> dict[str, list[str]]:
    return split_classes(list(ACTION_LABELS), 7, 2, 3, seed=42)


@pytest.fixture(scope='module')
def sampler(manifest_path: Path, splits_default: dict[str, list[str]]) -> EpisodeSampler:
    return EpisodeSampler(
        manifest_path=manifest_path,
        n_way=5,
        k_shot=5,
        q_query=15,
        splits=splits_default,
        seed=42,
        speed_split='none',
    )


def test_episode_structure(sampler: EpisodeSampler) -> None:
    ep = sampler.sample_episode('meta_train', 0)
    assert ep['n_way'] == 5
    assert ep['k_shot'] == 5
    assert ep['q_query'] == 15
    assert len(ep['classes']) == 5
    for cls in ep['classes']:
        assert len(ep['support'][cls]) == 5
        assert len(ep['query'][cls]) == 15
        assert set(ep['support'][cls]).isdisjoint(ep['query'][cls])


def test_episode_classes_inside_split(sampler: EpisodeSampler, splits_default: dict[str, list[str]]) -> None:
    if len(splits_default['meta_train']) >= sampler.n_way:
        ep = sampler.sample_episode('meta_train', 0)
        assert set(ep['classes']).issubset(set(splits_default['meta_train']))


def test_episode_determinism_same_seed(manifest_path: Path, splits_default: dict[str, list[str]]) -> None:
    a = EpisodeSampler(
        manifest_path=manifest_path, n_way=5, k_shot=5, q_query=15,
        splits=splits_default, seed=42,
    )
    b = EpisodeSampler(
        manifest_path=manifest_path, n_way=5, k_shot=5, q_query=15,
        splits=splits_default, seed=42,
    )
    for i in range(3):
        ea = a.sample_episode('meta_train', i)
        eb = b.sample_episode('meta_train', i)
        assert ea['classes'] == eb['classes']
        assert ea['support'] == eb['support']
        assert ea['query'] == eb['query']


def test_episode_different_seed_changes_output(
    manifest_path: Path, splits_default: dict[str, list[str]]
) -> None:
    a = EpisodeSampler(
        manifest_path=manifest_path, n_way=5, k_shot=5, q_query=15,
        splits=splits_default, seed=1,
    )
    b = EpisodeSampler(
        manifest_path=manifest_path, n_way=5, k_shot=5, q_query=15,
        splits=splits_default, seed=2,
    )
    diffs = 0
    for i in range(5):
        if a.sample_episode('meta_train', i)['support'] != b.sample_episode('meta_train', i)['support']:
            diffs += 1
    assert diffs >= 1


def test_episode_speed_split_beginner_to_expert(
    manifest_path: Path, splits_default: dict[str, list[str]]
) -> None:
    sampler = EpisodeSampler(
        manifest_path=manifest_path,
        n_way=2, k_shot=2, q_query=2,
        splits=splits_default, seed=42,
        speed_split='beginner_to_expert',
    )
    ep = sampler.sample_episode('meta_train', 0)
    df = pd.read_csv(manifest_path, dtype={'actor': str}, keep_default_na=False)
    actor_index = dict(zip(df['sample_id'], df['actor_index']))
    skill = dict(zip(df['sample_id'], df['skill_level']))
    for cls in ep['classes']:
        for sid in ep['support'][cls]:
            assert skill[sid] == 'beginner', f'support sample {sid} not a beginner'
            assert actor_index[sid] <= 31
        for sid in ep['query'][cls]:
            assert skill[sid] == 'expert', f'query sample {sid} not an expert'
            assert actor_index[sid] >= 32
    assert ep['speed_split_mode'] == 'beginner_to_expert'


def test_episode_speed_split_expert_to_beginner(
    manifest_path: Path, splits_default: dict[str, list[str]]
) -> None:
    sampler = EpisodeSampler(
        manifest_path=manifest_path,
        n_way=2, k_shot=2, q_query=2,
        splits=splits_default, seed=42,
        speed_split='expert_to_beginner',
    )
    ep = sampler.sample_episode('meta_train', 0)
    df = pd.read_csv(manifest_path, dtype={'actor': str}, keep_default_na=False)
    skill = dict(zip(df['sample_id'], df['skill_level']))
    for cls in ep['classes']:
        assert all(skill[sid] == 'expert' for sid in ep['support'][cls])
        assert all(skill[sid] == 'beginner' for sid in ep['query'][cls])
