'''Tests for src/data/loader.py — parsing, manifest, ThetisDataset.'''
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    ACTION_INDEX,
    ACTION_LABELS,
    PATH_COLUMNS,
    REQUIRED_COLUMNS,
    ThetisDataset,
    build_integrity_report,
    canonical_action,
    collect_records_wide,
    cross_modality_alignment_check,
    build_exclusions,
    duplicate_files_check,
    load_exclusions,
    infer_action_from_token,
    infer_skill_level,
    parse_actor_and_sequence,
)


# ---------------------------------------------------------------------------
# Pure parsing helpers (no filesystem)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('token', 'expected'),
    [
        ('backhand', 'backhand'),
        ('backhand2h', 'backhand2hands'),
        ('bslice', 'backhand_slice'),
        ('bvolley', 'backhand_volley'),
        ('foreflat', 'forehand_flat'),
        ('foreopen', 'forehand_openstands'),
        ('fslice', 'forehand_slice'),
        ('fvolley', 'forehand_volley'),
        ('serflat', 'flat_service'),
        ('serkick', 'kick_service'),
        ('serslice', 'slice_service'),
        ('smash', 'smash'),
    ],
)
def test_canonical_action_known_aliases(token: str, expected: str) -> None:
    assert canonical_action(token) == expected


def test_canonical_action_passthrough_unknown() -> None:
    assert canonical_action('unknown_action') == 'unknown_action'


@pytest.mark.parametrize(
    ('stem', 'actor', 'seq', 'token_substring'),
    [
        ('p10_fvolley_s1', 'p10', 1, 'fvolley'),
        ('p10_fvolley_depth_s2', 'p10', 2, 'fvolley'),
        ('p10_fvolley_mask_s3', 'p10', 3, 'fvolley'),
        ('p10_fvolley_skelet2D_s1', 'p10', 1, 'fvolley'),
        ('p10_fvolley_skelet3D_s1', 'p10', 1, 'fvolley'),
        ('p32_smash_s2', 'p32', 2, 'smash'),
        ('p1_backhand_skelet3D_s3', 'p1', 3, 'backhand'),
    ],
)
def test_parse_actor_and_sequence(stem: str, actor: str, seq: int, token_substring: str) -> None:
    a, s, tok = parse_actor_and_sequence(stem)
    assert a == actor
    assert s == seq
    assert tok is not None and token_substring in tok


def test_parse_actor_and_sequence_handles_windows_dup_suffix() -> None:
    a, s, _ = parse_actor_and_sequence('p1_backhand_s1 (1)')
    assert a == 'p1'
    assert s == 1


def test_parse_actor_and_sequence_rejects_garbage() -> None:
    a, s, t = parse_actor_and_sequence('not_a_thetis_filename')
    assert a is None and s is None and t is None


def test_infer_action_from_token_handles_modality_suffix() -> None:
    assert infer_action_from_token('fvolley_depth', fallback_action_id='forehand_volley') == 'forehand_volley'
    assert infer_action_from_token('backhand_skelet3d', fallback_action_id='backhand') == 'backhand'


def test_infer_action_falls_back_when_token_missing() -> None:
    assert infer_action_from_token(None, fallback_action_id='smash') == 'smash'


@pytest.mark.parametrize(
    ('actor', 'expected'),
    [('p1', 'beginner'), ('p31', 'beginner'), ('p32', 'expert'), ('p55', 'expert')],
)
def test_infer_skill_level(actor: str, expected: str) -> None:
    assert infer_skill_level(actor) == expected


# ---------------------------------------------------------------------------
# Synthetic tree manifest assembly
# ---------------------------------------------------------------------------


def test_collect_records_wide_synthetic(mock_dataset_tree: Path) -> None:
    df, diag = collect_records_wide(mock_dataset_tree)
    assert not df.empty
    expected_keys = {
        ('p1', 'backhand', 1),
        ('p1', 'backhand', 2),
        ('p32', 'backhand', 1),
        ('p1', 'smash', 1),
        ('p32', 'smash', 1),
        ('p32', 'smash', 2),
    }
    actual_keys = {(r.actor, r.action_label, r.sequence_idx) for r in df.itertuples(index=False)}
    assert actual_keys == expected_keys

    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    for col in PATH_COLUMNS.values():
        assert col in df.columns

    assert (df['path_mask'] == '').all()
    assert (df['path_skeleton_2d'] == '').all()
    assert (df['path_skeleton_3d'] == '').all()

    row = df[(df['actor'] == 'p1') & (df['action_label'] == 'backhand') & (df['sequence_idx'] == 2)].iloc[0]
    assert row['path_rgb'].endswith('p1_backhand_s2.avi')
    assert row['path_depth'] == ''

    assert (df.loc[df['actor'] == 'p1', 'skill_level'] == 'beginner').all()
    assert (df.loc[df['actor'] == 'p32', 'skill_level'] == 'expert').all()

    assert diag['parse_failures'] == []
    assert diag['action_mismatches'] == []
    assert diag['unknown_class_dirs'] == []


def test_collect_records_wide_reports_key_collisions(tmp_path: Path) -> None:
    '''Two files of one modality parsing to the same key must not vanish silently.

    THETIS ships this case: ``p19_bvolley_skelet2D_s3 (1).avi`` and ``(2).avi``
    are different takes whose ``' (N)'`` suffix is stripped during parsing, so
    both land on ``(p19, backhand_volley, 3)`` and the last one wins.
    '''
    d = tmp_path / 'VIDEO_Skelet2D' / 'backhand_volley'
    d.mkdir(parents=True)
    (d / 'p19_bvolley_skelet2D_s3 (1).avi').write_bytes(b'take3')
    (d / 'p19_bvolley_skelet2D_s3 (2).avi').write_bytes(b'take2')

    df, diag = collect_records_wide(tmp_path)

    # a colisão colapsa as duas em uma linha só
    assert len(df) == 1
    collisions = diag['key_collisions']
    assert len(collisions) == 1
    c = collisions[0]
    assert c['modality'] == 'skeleton_2d'
    assert (c['actor'], c['action_label'], c['sequence_idx']) == ('p19', 'backhand_volley', 3)
    # ordem alfabética: (2) é processado por último e sobrescreve (1)
    assert c['kept'].endswith('(2).avi')
    assert c['dropped'].endswith('(1).avi')
    assert df.iloc[0]['path_skeleton_2d'] == c['kept']


def test_collect_records_wide_no_collisions_when_names_are_distinct(tmp_path: Path) -> None:
    d = tmp_path / 'VIDEO_Skelet2D' / 'backhand_volley'
    d.mkdir(parents=True)
    (d / 'p19_bvolley_skelet2D_s2.avi').write_bytes(b'take2')
    (d / 'p19_bvolley_skelet2D_s3.avi').write_bytes(b'take3')

    df, diag = collect_records_wide(tmp_path)

    assert len(df) == 2
    assert diag['key_collisions'] == []


def test_cross_modality_alignment_flags_frame_count_disagreement(tmp_path: Path) -> None:
    '''Rows whose modalities disagree on length are surfaced, not swallowed.'''
    import numpy as np
    import pandas as pd
    import pytest as _pytest

    cv2 = _pytest.importorskip('cv2')

    def write_video(path: Path, n_frames: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (32, 32))
        for _ in range(n_frames):
            w.write(np.zeros((32, 32, 3), dtype=np.uint8))
        w.release()

    write_video(tmp_path / 'VIDEO_RGB' / 'smash' / 'p1_smash_s1.avi', 10)
    write_video(tmp_path / 'VIDEO_Depth' / 'smash' / 'p1_smash_depth_s1.avi', 10)
    write_video(tmp_path / 'VIDEO_RGB' / 'smash' / 'p1_smash_s2.avi', 10)
    write_video(tmp_path / 'VIDEO_Depth' / 'smash' / 'p1_smash_depth_s2.avi', 7)

    df, _ = collect_records_wide(tmp_path)
    result = cross_modality_alignment_check(df, tmp_path)

    assert result['checked_rows'] == 2
    ids = {m['sample_id'] for m in result['mismatched']}
    assert ids == {'p1_smash_s2'}, f'esperava so s2 desalinhado, veio {ids}'
    assert result['mismatched'][0]['spread'] == 3


def test_cross_modality_alignment_skipped_without_full_integrity(mock_dataset_tree: Path) -> None:
    df, diagnostics = collect_records_wide(mock_dataset_tree)
    report = build_integrity_report(df, diagnostics, mock_dataset_tree, seed=42, full_integrity=False)
    assert report['cross_modality_alignment']['mismatched'] == []
    assert 'skipped_reason' in report['cross_modality_alignment']
    assert report['key_collisions'] == []


def test_duplicate_files_check_finds_byte_identical_clips(tmp_path: Path) -> None:
    '''THETIS pads missing repetitions with copies; those must be reported.'''
    d = tmp_path / 'VIDEO_RGB' / 'smash'
    d.mkdir(parents=True)
    (d / 'p1_smash_s1.avi').write_bytes(b'TAKE-ONE-BYTES')
    (d / 'p1_smash_s2.avi').write_bytes(b'TAKE-ONE-BYTES')  # cópia byte a byte
    (d / 'p1_smash_s3.avi').write_bytes(b'a-genuinely-different-take')

    df, _ = collect_records_wide(tmp_path)
    result = duplicate_files_check(df, tmp_path)

    assert result['duplicate_file_count'] == 2
    assert len(result['duplicate_groups']) == 1
    paths = result['duplicate_groups'][0]['paths']
    assert sorted(Path(p).name for p in paths) == ['p1_smash_s1.avi', 'p1_smash_s2.avi']
    # pré-filtro por tamanho: o clipe de tamanho único nem chega a ser lido
    assert result['files_hashed'] == 2


def test_duplicate_files_check_clean_tree(mock_dataset_tree: Path) -> None:
    # a árvore sintética usa os mesmos bytes em todo arquivo, então serve de
    # verificação de que o agrupamento é por conteúdo e não por nome
    df, _ = collect_records_wide(mock_dataset_tree)
    result = duplicate_files_check(df, mock_dataset_tree)
    assert result['duplicate_file_count'] == 10
    assert len(result['duplicate_groups']) == 1


def test_build_exclusions_keeps_one_of_each_duplicate_group() -> None:
    df = pd.DataFrame(
        [
            {'sample_id': 'p1_smash_s1', 'path_rgb': 'VIDEO_RGB/smash/a.avi', 'path_depth': '',
             'path_mask': '', 'path_skeleton_2d': '', 'path_skeleton_3d': ''},
            {'sample_id': 'p1_smash_s2', 'path_rgb': 'VIDEO_RGB/smash/b.avi', 'path_depth': '',
             'path_mask': '', 'path_skeleton_2d': '', 'path_skeleton_3d': ''},
            {'sample_id': 'p1_smash_s3', 'path_rgb': 'VIDEO_RGB/smash/c.avi', 'path_depth': '',
             'path_mask': '', 'path_skeleton_2d': '', 'path_skeleton_3d': ''},
        ]
    )
    report = {
        'duplicate_files': {
            'duplicate_groups': [
                {'paths': ['VIDEO_RGB/smash/a.avi', 'VIDEO_RGB/smash/b.avi'], 'count': 2}
            ]
        },
        'degenerate_clips': {'blank_clips': [{'sample_id': 'p1_smash_s3', 'modality': 'rgb',
                                              'path': 'VIDEO_RGB/smash/c.avi'}]},
    }
    out = build_exclusions(report, df)
    rgb = out['excluded_by_modality']['rgb']
    # do par duplicado sobra um representante; o clipe em branco sai
    assert rgb == {'p1_smash_s2': 'duplicate', 'p1_smash_s3': 'blank'}
    assert out['counts']['rgb'] == 2
    assert out['counts']['depth'] == 0


def test_build_exclusions_flags_rows_sharing_clips_without_excluding_them() -> None:
    df = pd.DataFrame(
        [
            {'sample_id': 'p55_x_s1', 'path_rgb': '', 'path_depth': '', 'path_mask': 'M/s1.avi',
             'path_skeleton_2d': '', 'path_skeleton_3d': ''},
            {'sample_id': 'p55_x_s3', 'path_rgb': '', 'path_depth': '', 'path_mask': 'M/s3.avi',
             'path_skeleton_2d': '', 'path_skeleton_3d': ''},
        ]
    )
    report = {'duplicate_files': {'duplicate_groups': [{'paths': ['M/s1.avi', 'M/s3.avi'], 'count': 2}]}}
    out = build_exclusions(report, df)
    assert out['rows_sharing_clips_across_takes'] == [
        {'sample_ids': ['p55_x_s1', 'p55_x_s3'], 'shared_modalities': ['mask']}
    ]


def test_load_exclusions_missing_file_is_not_fatal(tmp_path: Path) -> None:
    (tmp_path / 'manifest.csv').write_text('sample_id\n')
    assert load_exclusions(tmp_path / 'manifest.csv', 'rgb') == frozenset()


def test_load_exclusions_reads_requested_modality(tmp_path: Path) -> None:
    (tmp_path / 'manifest.csv').write_text('sample_id\n')
    (tmp_path / 'excluded_clips.json').write_text(json.dumps({
        'excluded_by_modality': {'rgb': {'a': 'blank'}, 'mask': {'b': 'duplicate', 'c': 'blank'}}
    }))
    assert load_exclusions(tmp_path / 'manifest.csv', 'mask') == frozenset({'b', 'c'})
    assert load_exclusions(tmp_path / 'manifest.csv', 'depth') == frozenset()
    assert load_exclusions(tmp_path / 'manifest.csv', 'mask', enabled=False) == frozenset()


def test_collect_records_wide_skips_unknown_class_dirs(tmp_path: Path) -> None:
    weird = tmp_path / 'VIDEO_RGB' / 'weird_class'
    weird.mkdir(parents=True)
    (weird / 'p1_smash_s1.avi').write_bytes(b'x')
    df, diag = collect_records_wide(tmp_path)
    assert df.empty
    assert diag['unknown_class_dirs'], 'should report unknown class dirs'


# ---------------------------------------------------------------------------
# Manifest produced by the CLI (skipped if not built)
# ---------------------------------------------------------------------------


def test_manifest_required_columns(manifest_df: pd.DataFrame) -> None:
    for col in REQUIRED_COLUMNS:
        assert col in manifest_df.columns
    for col in PATH_COLUMNS.values():
        assert col in manifest_df.columns
    assert 'n_modalities' in manifest_df.columns


def test_manifest_no_nan_in_required(manifest_df: pd.DataFrame) -> None:
    for col in REQUIRED_COLUMNS:
        assert manifest_df[col].notna().all(), f'NaN found in required column {col!r}'


def test_manifest_has_expected_actors_and_actions(manifest_df: pd.DataFrame) -> None:
    assert set(manifest_df['action_label'].unique()) == set(ACTION_LABELS)
    assert manifest_df['actor'].nunique() == 55
    assert 1900 <= len(manifest_df) <= 2000


def test_manifest_skill_level_threshold(manifest_df: pd.DataFrame) -> None:
    beginners = manifest_df[manifest_df['skill_level'] == 'beginner']
    experts = manifest_df[manifest_df['skill_level'] == 'expert']
    assert (beginners['actor_index'] <= 31).all()
    assert (experts['actor_index'] >= 32).all()
    assert (beginners['actor_index'] >= 1).all()
    assert (experts['actor_index'] <= 55).all()


def test_manifest_action_index_consistent(manifest_df: pd.DataFrame) -> None:
    for label, expected_idx in ACTION_INDEX.items():
        rows = manifest_df[manifest_df['action_label'] == label]
        if len(rows):
            assert (rows['action_index'] == expected_idx).all()


def test_manifest_skeleton_coverage_reduced(manifest_df: pd.DataFrame) -> None:
    rgb_count = (manifest_df['path_rgb'] != '').sum()
    skel_count = (manifest_df['path_skeleton_2d'] != '').sum()
    assert skel_count > 0
    assert skel_count < rgb_count


# ---------------------------------------------------------------------------
# ThetisDataset against the real dataset (skipped if absent)
# ---------------------------------------------------------------------------


def test_thetis_dataset_filters_by_modality(manifest_path: Path, dataset_root: Path) -> None:
    ds_rgb = ThetisDataset(manifest_path, modalities=['rgb'], dataset_root=dataset_root, return_tensors=False)
    ds_rgb_skel = ThetisDataset(
        manifest_path, modalities=['rgb', 'skeleton_3d'], dataset_root=dataset_root, return_tensors=False
    )
    assert len(ds_rgb) >= len(ds_rgb_skel) > 0
    assert len(ds_rgb_skel) <= len(ds_rgb)


def test_thetis_dataset_getitem_shapes(manifest_path: Path, dataset_root: Path) -> None:
    ds = ThetisDataset(
        manifest_path,
        modalities=['rgb'],
        dataset_root=dataset_root,
        frame_count=8,
        return_tensors=False,
    )
    if len(ds) == 0:
        pytest.skip('dataset is empty')
    indices = sorted({0, len(ds) // 4, len(ds) // 2, 3 * len(ds) // 4, len(ds) - 1})
    for i in indices[:5]:
        sample = ds[i]
        assert 'rgb' in sample
        arr = sample['rgb']
        assert arr.ndim == 4 and arr.shape[0] == 8 and arr.shape[3] == 3
        assert 0 <= sample['label'] < len(ACTION_LABELS)
        assert sample['action_label'] in ACTION_LABELS
        assert sample['actor'].startswith('p')


def test_thetis_dataset_rejects_unknown_modality(manifest_path: Path, dataset_root: Path) -> None:
    with pytest.raises(ValueError, match='unknown modalities'):
        ThetisDataset(manifest_path, modalities=['nope'], dataset_root=dataset_root)
