'''THETIS dataset manifest builder + PyTorch-style Dataset.

The CLI walks ``dataset/{VIDEO_RGB,VIDEO_Depth,VIDEO_Mask,VIDEO_Skelet2D,VIDEO_Skelet3D}``
and emits ``data/processed/{manifest.csv,integrity_report.json,counts_by_modality_action.csv,
label_to_index.json}``.

Skeletons in THETIS are visualization videos (skeleton rendered on black
background), NOT raw joint coordinates. Coordinates of shape (T, J, C) are
produced downstream by ``src/features/pose.py``. Until then, modalities
``skeleton_2d``/``skeleton_3d`` load the visualization video as a ``(T,H,W,3)``
tensor and the dataset emits the keys ``skeleton_2d_video``/``skeleton_3d_video``.
'''
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITY_DIRS: dict[str, str] = {
    'VIDEO_RGB': 'rgb',
    'VIDEO_Depth': 'depth',
    'VIDEO_Mask': 'mask',
    'VIDEO_Skelet2D': 'skeleton_2d',
    'VIDEO_Skelet3D': 'skeleton_3d',
}

MODALITIES: tuple[str, ...] = tuple(MODALITY_DIRS.values())

# Filename action tokens → canonical class label (= folder name)
ACTION_ALIASES: dict[str, str] = {
    'backhand': 'backhand',
    'backhand_slice': 'backhand_slice',
    'bslice': 'backhand_slice',
    'backhand_volley': 'backhand_volley',
    'bvolley': 'backhand_volley',
    'backhand2hands': 'backhand2hands',
    'backhand2h': 'backhand2hands',
    'flat_service': 'flat_service',
    'serflat': 'flat_service',
    'forehand_flat': 'forehand_flat',
    'foreflat': 'forehand_flat',
    'forehand_openstands': 'forehand_openstands',
    'foreopen': 'forehand_openstands',
    'forehand_slice': 'forehand_slice',
    'fslice': 'forehand_slice',
    'forehand_volley': 'forehand_volley',
    'fvolley': 'forehand_volley',
    'kick_service': 'kick_service',
    'serkick': 'kick_service',
    'slice_service': 'slice_service',
    'serslice': 'slice_service',
    'smash': 'smash',
}

# 12 canonical class labels (= subfolder names in dataset/VIDEO_*)
ACTION_LABELS: tuple[str, ...] = (
    'backhand',
    'backhand2hands',
    'backhand_slice',
    'backhand_volley',
    'flat_service',
    'forehand_flat',
    'forehand_openstands',
    'forehand_slice',
    'forehand_volley',
    'kick_service',
    'slice_service',
    'smash',
)

ACTION_INDEX: dict[str, int] = {label: i for i, label in enumerate(sorted(ACTION_LABELS))}

# Preferred short code per class (the form that appears in RGB filenames)
ACTION_LABEL_TO_CODE: dict[str, str] = {
    'backhand': 'backhand',
    'backhand2hands': 'backhand2h',
    'backhand_slice': 'bslice',
    'backhand_volley': 'bvolley',
    'flat_service': 'serflat',
    'forehand_flat': 'foreflat',
    'forehand_openstands': 'foreopen',
    'forehand_slice': 'fslice',
    'forehand_volley': 'fvolley',
    'kick_service': 'serkick',
    'slice_service': 'serslice',
    'smash': 'smash',
}

FILE_EXTENSIONS: frozenset[str] = frozenset({'.avi'})

PATH_COLUMNS: dict[str, str] = {
    'rgb': 'path_rgb',
    'depth': 'path_depth',
    'mask': 'path_mask',
    'skeleton_2d': 'path_skeleton_2d',
    'skeleton_3d': 'path_skeleton_3d',
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    'sample_id',
    'actor',
    'actor_index',
    'skill_level',
    'action_code',
    'action_label',
    'action_index',
    'sequence_idx',
)


# ---------------------------------------------------------------------------
# Parsing helpers (reused from commit 461719e)
# ---------------------------------------------------------------------------


def canonical_action(raw_value: str) -> str:
    '''Normalise a raw action token to its canonical label, or pass-through.'''
    token = raw_value.strip().lower().replace('-', '_').replace(' ', '_')
    token = re.sub(r'_+', '_', token)
    return ACTION_ALIASES.get(token, token)


def parse_actor_and_sequence(stem: str) -> tuple[str | None, int | None, str | None]:
    '''Parse ``{actor}_{token}_s{seq}`` (modality suffix tolerated inside token).

    Returns ``(actor_id, sequence_index, action_token)``; any field may be ``None``
    when parsing fails.
    '''
    normalized = stem.lower().strip()
    normalized = re.sub(r'\s+\(\d+\)$', '', normalized)

    actor_match = re.match(r'^(p\d+)', normalized)
    if not actor_match:
        return None, None, None

    actor_id = actor_match.group(1)
    actor_end = actor_match.end()

    seq_match = re.search(r'(\d+)$', normalized)
    if not seq_match:
        return actor_id, None, None

    sequence_index = int(seq_match.group(1))
    action_token = normalized[actor_end : seq_match.start()].strip('_-')
    action_token = action_token or None
    return actor_id, sequence_index, action_token


def infer_action_from_token(action_token: str | None, fallback_action_id: str) -> str:
    '''Resolve the canonical class from a (possibly suffixed) filename token.'''
    if not action_token:
        return fallback_action_id

    normalized = action_token.strip().lower().replace('-', '_').replace(' ', '_')
    normalized = re.sub(r'_+', '_', normalized)

    canonical = canonical_action(normalized)
    if canonical in ACTION_INDEX:
        return canonical

    for alias in sorted(ACTION_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf'(^|_){re.escape(alias)}($|_)', normalized):
            return ACTION_ALIASES[alias]

    return fallback_action_id


def infer_skill_level(actor_id: str) -> str:
    '''``beginner`` for p1–p31, ``expert`` for p32–p55.'''
    actor_index = int(actor_id[1:])
    return 'beginner' if actor_index <= 31 else 'expert'


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def collect_records_wide(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    '''Walk every modality once and assemble a wide manifest.

    One row per ``(actor, action_label, sequence_idx)`` with one ``path_<modality>``
    column per modality. Missing modalities are stored as ``''`` (empty string),
    not NaN. Diagnostics include parse failures, action-token mismatches,
    orphan files (regex non-match), missing modality directories and key
    collisions.

    A *key collision* is two files of the same modality parsing to the same
    ``(actor, action, sequence)``. The last one wins, so the other is dropped
    from the manifest entirely — and because the survivor may be a different
    take than the sibling modalities at that key, the row's cross-modality
    pairing can be wrong. THETIS ships one such case: ``VIDEO_Skelet2D`` has
    ``p19_bvolley_skelet2D_s3 (1).avi`` (really sequence 3) and ``(2).avi``
    (really sequence 2), whose ``' (N)'`` suffixes are stripped by
    :func:`parse_actor_and_sequence`. Recording collisions here keeps that
    class of defect out of silence — see ``experiments/configs/README.md``.
    '''
    records: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(dict)
    parse_failures: list[dict[str, str]] = []
    action_mismatches: list[dict[str, str]] = []
    orphans: list[dict[str, str]] = []
    missing_modality_dirs: list[str] = []
    unknown_class_dirs: list[dict[str, str]] = []
    key_collisions: list[dict[str, Any]] = []

    for modality_dir_name, modality in MODALITY_DIRS.items():
        modality_root = dataset_root / modality_dir_name
        if not modality_root.exists():
            missing_modality_dirs.append(modality_dir_name)
            continue

        for action_dir in sorted((p for p in modality_root.iterdir() if p.is_dir()), key=lambda p: p.name):
            folder_class = canonical_action(action_dir.name)
            if folder_class not in ACTION_INDEX:
                unknown_class_dirs.append(
                    {'modality': modality, 'folder': action_dir.name, 'canonical': folder_class}
                )
                continue

            for file_path in sorted(action_dir.iterdir(), key=lambda p: p.name):
                if not file_path.is_file() or file_path.suffix.lower() not in FILE_EXTENSIONS:
                    continue

                actor_id, seq_idx, token = parse_actor_and_sequence(file_path.stem)
                relpath = file_path.relative_to(dataset_root).as_posix()
                if actor_id is None or seq_idx is None:
                    parse_failures.append({'path': relpath, 'reason': 'could_not_parse_actor_or_sequence'})
                    orphans.append({'path': relpath, 'reason': 'regex_no_match'})
                    continue

                inferred_class = infer_action_from_token(token, fallback_action_id=folder_class)
                if inferred_class != folder_class:
                    action_mismatches.append(
                        {
                            'path': relpath,
                            'folder_class': folder_class,
                            'token_class': inferred_class,
                        }
                    )

                key = (actor_id, folder_class, seq_idx)
                # Last write wins (below), so a collision silently drops a clip
                # and can mis-pair the row across modalities. Record it instead.
                previous = records[key].get(modality)
                if previous is not None:
                    key_collisions.append(
                        {
                            'modality': modality,
                            'actor': actor_id,
                            'action_label': folder_class,
                            'sequence_idx': seq_idx,
                            'dropped': previous,
                            'kept': relpath,
                        }
                    )
                records[key][modality] = relpath

    rows: list[dict[str, Any]] = []
    for (actor_id, action_label, seq_idx), paths in records.items():
        actor_index = int(actor_id[1:])
        action_code = ACTION_LABEL_TO_CODE[action_label]
        n_modalities = sum(1 for m in MODALITIES if paths.get(m))
        row: dict[str, Any] = {
            'sample_id': f'{actor_id}_{action_label}_s{seq_idx}',
            'actor': actor_id,
            'actor_index': actor_index,
            'skill_level': infer_skill_level(actor_id),
            'action_code': action_code,
            'action_label': action_label,
            'action_index': ACTION_INDEX[action_label],
            'sequence_idx': seq_idx,
        }
        for modality, column in PATH_COLUMNS.items():
            row[column] = paths.get(modality, '')
        row['n_modalities'] = n_modalities
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['action_label', 'actor_index', 'sequence_idx'], kind='stable').reset_index(drop=True)

    diagnostics = {
        'parse_failures': parse_failures,
        'action_mismatches': action_mismatches,
        'orphans': orphans,
        'missing_modality_dirs': missing_modality_dirs,
        'unknown_class_dirs': unknown_class_dirs,
        'key_collisions': key_collisions,
    }
    return df, diagnostics


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _missing_combinations(df: pd.DataFrame, modality: str) -> list[dict[str, Any]]:
    column = PATH_COLUMNS[modality]
    if df.empty or column not in df.columns:
        return []
    missing = df[df[column] == ''][['actor', 'action_label', 'sequence_idx']]
    return [
        {'actor': r.actor, 'action_label': r.action_label, 'sequence_idx': int(r.sequence_idx)}
        for r in missing.itertuples(index=False)
    ]


def build_counts_table(df: pd.DataFrame) -> pd.DataFrame:
    '''Long-format counts: 12 actions × 5 modalities, with coverage_pct vs the row total.'''
    if df.empty:
        return pd.DataFrame(columns=['modality', 'action_label', 'count', 'coverage_pct'])
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        column = PATH_COLUMNS[modality]
        present = df[df[column] != '']
        for label in ACTION_LABELS:
            denom = int((df['action_label'] == label).sum())
            count = int((present['action_label'] == label).sum())
            coverage = round(count / denom, 4) if denom else 0.0
            rows.append(
                {
                    'modality': modality,
                    'action_label': label,
                    'count': count,
                    'coverage_pct': coverage,
                }
            )
    return pd.DataFrame(rows)


def video_meta_check(
    df: pd.DataFrame,
    dataset_root: Path,
    sample_size: int = 200,
    rng: np.random.Generator | None = None,
    full: bool = False,
) -> dict[str, Any]:
    '''Open a stratified sample of videos to check for zero-byte/unreadable files.

    Uses ``cv2.VideoCapture`` (lazy import) for metadata only — no full decoding.
    '''
    try:
        import cv2  # type: ignore  # noqa: PLC0415
    except ImportError:
        return {
            'checked_count': 0,
            'open_failures': [],
            'zero_byte_files': [],
            'modalities_checked': [],
            'skipped_reason': 'opencv-python not installed',
        }

    if rng is None:
        rng = np.random.default_rng(0)

    selected: list[tuple[str, str]] = []
    for modality in MODALITIES:
        column = PATH_COLUMNS[modality]
        if column not in df.columns:
            continue
        paths = df.loc[df[column] != '', column].tolist()
        if not paths:
            continue
        if full:
            chosen = paths
        else:
            per_modality = max(1, sample_size // len(MODALITIES))
            n = min(per_modality, len(paths))
            chosen = list(rng.choice(paths, size=n, replace=False))
        selected.extend((modality, p) for p in chosen)

    open_failures: list[dict[str, str]] = []
    zero_byte_files: list[str] = []
    for modality, relpath in selected:
        abs_path = dataset_root / relpath
        try:
            size = abs_path.stat().st_size
        except OSError as exc:
            open_failures.append({'path': relpath, 'modality': modality, 'error': f'stat: {exc}'})
            continue
        if size == 0:
            zero_byte_files.append(relpath)
            continue
        cap = cv2.VideoCapture(str(abs_path))
        if not cap.isOpened():
            open_failures.append({'path': relpath, 'modality': modality, 'error': 'cv2 cannot open'})
        cap.release()

    return {
        'checked_count': len(selected),
        'open_failures': open_failures,
        'zero_byte_files': zero_byte_files,
        'modalities_checked': list(MODALITIES),
    }


def cross_modality_alignment_check(df: pd.DataFrame, dataset_root: Path) -> dict[str, Any]:
    '''Compare frame counts across the modalities present in each manifest row.

    A row whose modalities disagree on length is *not* proof of a defect: THETIS
    renders each modality with its own pipeline, so trims legitimately differ by
    a few frames. But it is the signature a mis-paired take leaves, so the rows
    are surfaced for inspection instead of being silently accepted — the same
    principle as ``key_collisions``, which catches only the subset where the
    filenames also collide.

    Reading a frame count is a header read, not a decode, but it still opens
    every file — so this runs only under ``--full-integrity``.
    '''
    try:
        import cv2  # type: ignore  # noqa: PLC0415
    except ImportError:
        return {'checked_rows': 0, 'mismatched': [], 'skipped_reason': 'opencv-python not installed'}

    frames: dict[str, int] = {}

    def frame_count(relpath: str) -> int:
        if relpath not in frames:
            cap = cv2.VideoCapture(str(dataset_root / relpath))
            frames[relpath] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        return frames[relpath]

    mismatched: list[dict[str, Any]] = []
    checked = 0
    for row in df.itertuples(index=False):
        by_modality = {
            modality: frame_count(getattr(row, PATH_COLUMNS[modality]))
            for modality in MODALITIES
            if getattr(row, PATH_COLUMNS[modality], '')
        }
        if len(by_modality) < 2:
            continue
        checked += 1
        counts = set(by_modality.values())
        if len(counts) > 1:
            mismatched.append(
                {
                    'sample_id': row.sample_id,
                    'frames_by_modality': by_modality,
                    'spread': max(counts) - min(counts),
                }
            )

    return {'checked_rows': checked, 'mismatched': mismatched}


def duplicate_files_check(df: pd.DataFrame, dataset_root: Path) -> dict[str, Any]:
    '''Find byte-identical clips shipped under different names.

    THETIS pads missing repetitions by copying a sibling take, and the copies do
    not always agree across modalities — a row can end up with depth from take 2
    and mask from take 1, or (once) a mask belonging to a different subject
    entirely. Duplicates also let an episode draw the same clip into both support
    and query, which flatters accuracy.

    The candidate list comes from the manifest, not from a directory walk, so
    the check sees exactly the clips training would sample — and stays correct
    when only one modality's folder is present (a Colab/Kaggle upload).

    Hashing is two-stage: group by file size first (a stat call), then hash only
    inside groups sharing a size — so the ~13 GB tree is not read in full.
    '''
    relpaths = {
        getattr(row, PATH_COLUMNS[modality], '')
        for row in df.itertuples(index=False)
        for modality in MODALITIES
        if getattr(row, PATH_COLUMNS[modality], '')
    }
    by_size: dict[int, list[Path]] = defaultdict(list)
    for relpath in sorted(relpaths):
        path = dataset_root / relpath
        if path.is_file():
            by_size[path.stat().st_size].append(path)

    by_digest: dict[str, list[str]] = defaultdict(list)
    hashed = 0
    for paths in by_size.values():
        if len(paths) < 2:
            continue  # a unique size cannot have a byte-identical twin
        for path in paths:
            digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 — dedup, not security
            by_digest[digest].append(path.relative_to(dataset_root).as_posix())
            hashed += 1

    groups = sorted(paths for paths in by_digest.values() if len(paths) > 1)
    return {
        'files_hashed': hashed,
        'duplicate_groups': [{'paths': g, 'count': len(g)} for g in groups],
        'duplicate_file_count': sum(len(g) for g in groups),
    }


def degenerate_clips_check(df: pd.DataFrame, dataset_root: Path) -> dict[str, Any]:
    '''Find clips that carry no signal — every frame effectively blank.

    The Kinect player segmentation fails outright on some recordings and writes
    an all-black ``mask`` video. Those clips are valid files with a valid label
    and pass every other check, but they teach nothing and pollute an episode's
    support set. Detected generically (near-zero variance across all frames), so
    a blank clip in any modality is caught, not just ``mask``.

    Only runs under ``--full-integrity`` — it decodes every clip.
    '''
    try:
        import cv2  # type: ignore  # noqa: PLC0415
    except ImportError:
        return {'checked_clips': 0, 'blank_clips': [], 'skipped_reason': 'opencv-python not installed'}

    blank: list[dict[str, Any]] = []
    checked = 0
    for row in df.itertuples(index=False):
        for modality in MODALITIES:
            relpath = getattr(row, PATH_COLUMNS[modality], '')
            if not relpath:
                continue
            checked += 1
            cap = cv2.VideoCapture(str(dataset_root / relpath))
            peak = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                peak = max(peak, float(frame.max()))
                if peak >= 8.0:
                    break  # one bright pixel settles it — no need to decode the rest
            cap.release()
            if peak < 8.0:  # nothing brighter than sensor noise in any frame
                blank.append(
                    {'sample_id': row.sample_id, 'modality': modality, 'path': relpath}
                )

    return {'checked_clips': checked, 'blank_clips': blank}


def build_exclusions(report: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    '''Turn integrity findings into a per-modality list of clips to skip.

    Two rules, both aimed at episodic sampling:

    * **blank clips** — no signal to learn from (``degenerate_clips_check``).
    * **byte-identical duplicates** — THETIS pads missing repetitions by copying
      a sibling take, so the same footage can be drawn into an episode's support
      *and* query set, which flatters accuracy. One representative per group is
      kept (lowest ``sample_id``, so the choice is stable across runs) and the
      copies are dropped.

    Rows whose modalities come from different takes are reported but **not**
    excluded: each clip is still a valid example of its own modality, and the
    single-modality configs never pair them. The list is there for the
    multimodal work, where those rows must go.
    '''
    by_path: dict[str, tuple[str, str]] = {}
    for row in df.itertuples(index=False):
        for modality in MODALITIES:
            relpath = getattr(row, PATH_COLUMNS[modality], '')
            if relpath:
                by_path[relpath] = (row.sample_id, modality)

    excluded: dict[str, dict[str, str]] = {modality: {} for modality in MODALITIES}

    for entry in report.get('degenerate_clips', {}).get('blank_clips', []):
        excluded[entry['modality']][entry['sample_id']] = 'blank'

    for group in report.get('duplicate_files', {}).get('duplicate_groups', []):
        members = [by_path[p] for p in group['paths'] if p in by_path]
        by_modality: dict[str, list[str]] = defaultdict(list)
        for sample_id, modality in members:
            by_modality[modality].append(sample_id)
        for modality, sample_ids in by_modality.items():
            if len(sample_ids) < 2:
                continue
            for sample_id in sorted(sample_ids)[1:]:  # keep the first, drop the copies
                excluded[modality].setdefault(sample_id, 'duplicate')

    inconsistent: list[dict[str, Any]] = []
    for group in report.get('duplicate_files', {}).get('duplicate_groups', []):
        members = [by_path[p] for p in group['paths'] if p in by_path]
        sample_ids = {sample_id for sample_id, _ in members}
        if len(sample_ids) > 1:
            inconsistent.append(
                {
                    'sample_ids': sorted(sample_ids),
                    'shared_modalities': sorted({m for _, m in members}),
                }
            )

    return {
        'excluded_by_modality': {m: dict(sorted(v.items())) for m, v in excluded.items()},
        'counts': {m: len(v) for m, v in excluded.items()},
        'rows_sharing_clips_across_takes': inconsistent,
    }


def load_exclusions(manifest_path: str | Path, modality: str, enabled: bool = True) -> frozenset[str]:
    '''Read the ``excluded_clips.json`` that sits next to ``manifest_path``.

    Returns the sample_ids to drop for ``modality``. Missing file (or
    ``enabled=False``) yields an empty set, so a tree that never ran
    ``--full-integrity`` still trains — just without the filter.
    '''
    if not enabled:
        return frozenset()
    path = Path(manifest_path).parent / 'excluded_clips.json'
    if not path.is_file():
        return frozenset()
    payload = json.loads(path.read_text())
    return frozenset(payload.get('excluded_by_modality', {}).get(modality, {}))


def build_integrity_report(
    df: pd.DataFrame,
    diagnostics: dict[str, Any],
    dataset_root: Path,
    seed: int,
    full_integrity: bool = False,
) -> dict[str, Any]:
    '''Assemble the integrity_report.json payload from the manifest + diagnostics.'''
    counts_by_modality = {
        modality: int((df[PATH_COLUMNS[modality]] != '').sum()) if not df.empty else 0
        for modality in MODALITIES
    }

    counts_by_modality_action: dict[str, dict[str, int]] = {}
    coverage_by_modality_action: dict[str, dict[str, float]] = {}
    if not df.empty:
        for modality in MODALITIES:
            column = PATH_COLUMNS[modality]
            present = df[df[column] != '']
            counts: dict[str, int] = {}
            coverage: dict[str, float] = {}
            for label in ACTION_LABELS:
                denom = int((df['action_label'] == label).sum())
                value = int((present['action_label'] == label).sum())
                counts[label] = value
                coverage[label] = round(value / denom, 4) if denom else 0.0
            counts_by_modality_action[modality] = counts
            coverage_by_modality_action[modality] = coverage
    else:
        for modality in MODALITIES:
            counts_by_modality_action[modality] = {label: 0 for label in ACTION_LABELS}
            coverage_by_modality_action[modality] = {label: 0.0 for label in ACTION_LABELS}

    missing_by_modality = {modality: _missing_combinations(df, modality) for modality in MODALITIES}

    subjects_by_class: dict[str, dict[str, int]] = {}
    if not df.empty:
        for label in ACTION_LABELS:
            class_df = df[df['action_label'] == label]
            actors = class_df[['actor', 'skill_level']].drop_duplicates()
            beginners = int((actors['skill_level'] == 'beginner').sum())
            experts = int((actors['skill_level'] == 'expert').sum())
            subjects_by_class[label] = {
                'beginner': beginners,
                'expert': experts,
                'total': beginners + experts,
            }
    else:
        for label in ACTION_LABELS:
            subjects_by_class[label] = {'beginner': 0, 'expert': 0, 'total': 0}

    actors_by_skill = {'beginner': 0, 'expert': 0}
    if not df.empty:
        actors = df[['actor', 'skill_level']].drop_duplicates()
        actors_by_skill = {
            'beginner': int((actors['skill_level'] == 'beginner').sum()),
            'expert': int((actors['skill_level'] == 'expert').sum()),
        }

    rng = np.random.default_rng(seed)
    meta_check = video_meta_check(df, dataset_root, rng=rng, full=full_integrity)
    alignment = (
        cross_modality_alignment_check(df, dataset_root)
        if full_integrity and not df.empty
        else {'checked_rows': 0, 'mismatched': [], 'skipped_reason': 'run with --full-integrity'}
    )
    degenerate = (
        degenerate_clips_check(df, dataset_root)
        if full_integrity and not df.empty
        else {'checked_clips': 0, 'blank_clips': [], 'skipped_reason': 'run with --full-integrity'}
    )
    duplicates = (
        duplicate_files_check(df, dataset_root)
        if full_integrity and not df.empty
        else {
            'files_hashed': 0,
            'duplicate_groups': [],
            'duplicate_file_count': 0,
            'skipped_reason': 'run with --full-integrity',
        }
    )

    return {
        'schema_version': '1.4',  # +key_collisions, alignment, duplicate_files, degenerate_clips  
        'generated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'dataset_root': str(dataset_root),
        'seed': int(seed),
        'totals': {
            'rows_in_manifest': int(len(df)),
            'expected_max_rows': 1980,
            'unique_actors': int(df['actor'].nunique()) if not df.empty else 0,
            'unique_actions': int(df['action_label'].nunique()) if not df.empty else 0,
            'unique_sequences': sorted(int(s) for s in df['sequence_idx'].unique().tolist()) if not df.empty else [],
        },
        'counts_by_modality': counts_by_modality,
        'counts_by_modality_action': counts_by_modality_action,
        'coverage_by_modality_action_pct': coverage_by_modality_action,
        'missing_by_modality': missing_by_modality,
        'subjects_by_class': subjects_by_class,
        'actors_by_skill': actors_by_skill,
        'video_meta_check': meta_check,
        'orphans': diagnostics['orphans'],
        'action_mismatches': diagnostics['action_mismatches'],
        'missing_modality_dirs': diagnostics['missing_modality_dirs'],
        'unknown_class_dirs': diagnostics['unknown_class_dirs'],
        'key_collisions': diagnostics.get('key_collisions', []),
        'cross_modality_alignment': alignment,
        'duplicate_files': duplicates,
        'degenerate_clips': degenerate,
    }


def write_label_index(path: Path) -> None:
    '''Persist the canonical ``action_label → action_index`` mapping.'''
    payload = {
        'labels': list(ACTION_LABELS),
        'label_to_index': dict(ACTION_INDEX),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def manifest_sha1(df: pd.DataFrame) -> str:
    '''Hash of the ordered ``sample_id`` column — used for episode reproducibility.'''
    if df.empty:
        return hashlib.sha1(b'').hexdigest()
    joined = '\n'.join(df['sample_id'].tolist()).encode('utf-8')
    return hashlib.sha1(joined).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build the THETIS manifest + integrity report.')
    parser.add_argument('--input', type=Path, required=True, help='Path to dataset/ root')
    parser.add_argument('--output', type=Path, required=True, help='Path to data/ output root')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--full-integrity',
        action='store_true',
        help='Open ALL videos for the integrity check (slow). Default: stratified sample of ~200.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset_root = args.input.resolve()
    if not dataset_root.exists():
        parser.error(f'--input does not exist: {dataset_root}')

    out_root = args.output.resolve() / 'processed'
    out_root.mkdir(parents=True, exist_ok=True)

    df, diagnostics = collect_records_wide(dataset_root)
    report = build_integrity_report(df, diagnostics, dataset_root, args.seed, full_integrity=args.full_integrity)
    counts = build_counts_table(df)

    manifest_path = out_root / 'manifest.csv'
    counts_path = out_root / 'counts_by_modality_action.csv'
    report_path = out_root / 'integrity_report.json'
    label_path = out_root / 'label_to_index.json'
    exclusions_path = out_root / 'excluded_clips.json'

    exclusions = build_exclusions(report, df)

    df.to_csv(manifest_path, index=False)
    counts.to_csv(counts_path, index=False)
    report_path.write_text(json.dumps(report, indent=2))
    write_label_index(label_path)
    exclusions_path.write_text(json.dumps(exclusions, indent=2))

    print(f'manifest:           {manifest_path}  ({len(df)} rows)')
    print(f'counts:             {counts_path}')
    print(f'integrity_report:   {report_path}')
    print(f'label_to_index:     {label_path}')
    print(f'excluded_clips:     {exclusions_path}  ({exclusions["counts"]})')
    if not args.full_integrity:
        print('  (empty — the checks that feed it need --full-integrity)')

    collisions = report['key_collisions']
    if collisions:
        print(
            f'\nWARNING: {len(collisions)} key collision(s) — two files parsed to the same '
            f'(actor, action, sequence). The dropped clip is missing from the manifest, and '
            f'the kept one may be a different take than the sibling modalities at that key:'
        )
        for c in collisions:
            print(f"  [{c['modality']}] {c['actor']}/{c['action_label']}/s{c['sequence_idx']}")
            print(f"      kept:    {c['kept']}")
            print(f"      dropped: {c['dropped']}")
        print('  Fix by renaming the files to their true sequence index, then re-run.')

    misaligned = report['cross_modality_alignment']['mismatched']
    if misaligned:
        print(
            f'\nNOTE: {len(misaligned)} row(s) whose modalities disagree on frame count. '
            f'Some are harmless trim differences between rendering pipelines; others mark a row '
            f'whose modalities come from different takes — cross-check duplicate_files below and '
            f'see integrity_report.json > cross_modality_alignment.'
        )
        for m in sorted(misaligned, key=lambda x: -x['spread'])[:5]:
            print(f"  {m['sample_id']} (spread {m['spread']}): {m['frames_by_modality']}")
        if len(misaligned) > 5:
            print(f'  ... and {len(misaligned) - 5} more')

    dup = report['duplicate_files']['duplicate_groups']
    if dup:
        n_files = report['duplicate_files']['duplicate_file_count']
        print(
            f'\nNOTE: {len(dup)} group(s) of byte-identical clips ({n_files} files). THETIS pads '
            f'missing repetitions with copies of a sibling take; when the copies come from '
            f'different takes per modality, that row is internally inconsistent:'
        )
        for g in dup[:5]:
            print('  ' + '  ==  '.join(Path(x).name for x in g['paths']))
        if len(dup) > 5:
            print(f'  ... and {len(dup) - 5} more — see integrity_report.json > duplicate_files')
    return 0


# ---------------------------------------------------------------------------
# PyTorch-style Dataset (lazy torch / cv2 imports)
# ---------------------------------------------------------------------------


# Output keys per modality (skeletons get the explicit ``_video`` suffix)
MODALITY_KEY: dict[str, str] = {
    'rgb': 'rgb',
    'depth': 'depth',
    'mask': 'mask',
    'skeleton_2d': 'skeleton_2d_video',
    'skeleton_3d': 'skeleton_3d_video',
}


def _read_video_cv2(path: Path, frame_count: int | None) -> np.ndarray:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f'cv2 cannot open video: {path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count is not None and total > 0:
        indices = np.linspace(0, total - 1, num=frame_count).round().astype(int)
        wanted: set[int] | None = {int(i) for i in indices}
    else:
        wanted = None

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if wanted is None or idx in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    if not frames:
        raise IOError(f'no frames decoded: {path}')
    return np.stack(frames, axis=0).astype(np.uint8)


def _read_video_decord(path: Path, frame_count: int | None) -> np.ndarray:
    import decord  # type: ignore  # noqa: PLC0415

    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(str(path))
    total = len(vr)
    if frame_count is not None and total > 0:
        indices = np.linspace(0, total - 1, num=frame_count).round().astype(int).tolist()
    else:
        indices = list(range(total))
    arr = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) RGB
    return arr.astype(np.uint8)


def _read_video(path: Path, frame_count: int | None) -> np.ndarray:
    '''Read a video into ``(T,H,W,3)`` uint8. Tries decord, falls back to cv2.'''
    try:
        return _read_video_decord(path, frame_count)
    except ImportError:
        return _read_video_cv2(path, frame_count)


def _resize_clip(arr: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    '''Bilinear-resize every frame of a ``(T, H, W, C)`` clip to ``size``.

    Mirrors :class:`src.data.augment.ResizeVideo` exactly (per-frame
    ``cv2.resize`` with ``INTER_LINEAR``), so pre-resizing a clip here and
    letting a downstream ``ResizeVideo(size)`` become a no-op yields
    byte-identical results. Used by :class:`ThetisDataset`'s decode cache to
    store clips at a small working resolution (e.g. 128x128 ~ 1.5 MB / 32
    frames) instead of native (480x640 ~ 29 MB).
    '''
    import cv2  # type: ignore  # noqa: PLC0415

    out_h, out_w = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
    t, h, w, c = arr.shape
    if h == out_h and w == out_w:
        return arr
    out = np.empty((t, out_h, out_w, c), dtype=arr.dtype)
    for i in range(t):
        out[i] = cv2.resize(arr[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return out


class ThetisDataset:
    '''PyTorch-style dataset over the THETIS manifest.

    Args:
        manifest_path: Path to ``data/processed/manifest.csv``.
        modalities: subset of :data:`MODALITIES` to load. Rows missing ANY of
            these modalities are dropped at ``__init__`` time.
        dataset_root: Base directory for the relative ``path_*`` columns
            (typically the same path passed as ``--input`` to the CLI).
        transform: Optional callable that receives and returns a sample dict.
        frame_count: If given, uniformly samples this many frames per video.
            If ``None``, returns every frame.
        return_tensors: If True, returns ``torch.Tensor`` objects (lazy import);
            else ``np.ndarray``.
        cache: If True, memoise each decoded clip in RAM keyed by
            ``(index, modality)``. Episodic training re-samples the same small
            class pool thousands of times, so this collapses millions of disk
            decodes into one-per-clip. Pair with ``cache_resize`` to bound
            memory. Process-local; defaults to off to keep other callers
            (and tests) byte-for-byte unchanged.
        cache_resize: If set, resize decoded clips to this ``(H, W)`` (int →
            square) *before* caching, matching a downstream
            :class:`~src.data.augment.ResizeVideo` so results are unchanged
            while cached clips stay small.

    ``__getitem__`` returns a dict with keys
    ``sample_id, label, action_label, action_index, actor, actor_index,
    skill_level, sequence_idx`` and one tensor per requested modality:

      rgb / depth / mask          → (T, H, W, 3) uint8
      skeleton_2d_video           → (T, H, W, 3) uint8
      skeleton_3d_video           → (T, H, W, 3) uint8

    Skeletons in THETIS are visualization videos (skeleton rendered on a black
    background), NOT raw (T, J, C) joint coordinates. ``src/features/pose.py``
    will eventually materialise coordinate arrays under ``data/processed/pose/``.
    '''

    def __init__(
        self,
        manifest_path: str | Path,
        modalities: Sequence[str],
        dataset_root: str | Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        frame_count: int | None = None,
        return_tensors: bool = True,
        cache: bool = False,
        cache_resize: int | tuple[int, int] | None = None,
    ) -> None:
        unknown = [m for m in modalities if m not in PATH_COLUMNS]
        if unknown:
            raise ValueError(f'unknown modalities: {unknown}; valid: {list(MODALITIES)}')
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root).resolve()
        self.modalities: tuple[str, ...] = tuple(modalities)
        self.transform = transform
        self.frame_count = frame_count
        self.return_tensors = return_tensors
        self.cache_enabled = bool(cache)
        self.cache_resize = cache_resize
        # (idx, modality) → decoded uint8 clip; lazily filled so each video is
        # read from disk at most once for this dataset's lifetime.
        self._decode_cache: dict[tuple[int, str], np.ndarray] = {}

        df = pd.read_csv(
            self.manifest_path,
            dtype={'actor': str, 'action_code': str},
            keep_default_na=False,
        )
        for modality in self.modalities:
            column = PATH_COLUMNS[modality]
            df = df[df[column].astype(str) != '']
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _to_tensor(self, arr: np.ndarray) -> Any:
        if not self.return_tensors:
            return arr
        import torch  # noqa: PLC0415

        return torch.from_numpy(arr)

    def _load_video_arr(self, idx: int, modality: str) -> np.ndarray:
        '''Return the decoded ``(T, H, W, 3)`` uint8 clip for ``(idx, modality)``.

        Reads (and optionally resizes) from disk on first request; on later
        requests returns the cached buffer when ``cache`` is enabled. The
        returned array is the shared cache buffer — callers that mutate it must
        copy first (``__getitem__`` does).
        '''
        if self.cache_enabled:
            hit = self._decode_cache.get((idx, modality))
            if hit is not None:
                return hit
        relpath = str(self.df.iloc[idx][PATH_COLUMNS[modality]])
        arr = _read_video(self.dataset_root / relpath, self.frame_count)
        if self.cache_resize is not None:
            arr = _resize_clip(arr, self.cache_resize)
        if self.cache_enabled:
            self._decode_cache[(idx, modality)] = arr
        return arr

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sample: dict[str, Any] = {
            'sample_id': str(row['sample_id']),
            'label': int(row['action_index']),
            'action_label': str(row['action_label']),
            'action_index': int(row['action_index']),
            'actor': str(row['actor']),
            'actor_index': int(row['actor_index']),
            'skill_level': str(row['skill_level']),
            'sequence_idx': int(row['sequence_idx']),
        }
        for modality in self.modalities:
            arr = self._load_video_arr(idx, modality)
            if self.cache_enabled:
                # Hand out a private copy so downstream in-place transforms
                # never corrupt the shared cache buffer.
                arr = arr.copy()
            sample[MODALITY_KEY[modality]] = self._to_tensor(arr)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


__all__ = [
    'ACTION_ALIASES',
    'ACTION_INDEX',
    'ACTION_LABELS',
    'ACTION_LABEL_TO_CODE',
    'MODALITIES',
    'MODALITY_DIRS',
    'MODALITY_KEY',
    'PATH_COLUMNS',
    'REQUIRED_COLUMNS',
    'ThetisDataset',
    'build_counts_table',
    'build_integrity_report',
    'canonical_action',
    'collect_records_wide',
    'infer_action_from_token',
    'infer_skill_level',
    'main',
    'manifest_sha1',
    'parse_actor_and_sequence',
    'video_meta_check',
    'write_label_index',
]


if __name__ == '__main__':
    raise SystemExit(main())
