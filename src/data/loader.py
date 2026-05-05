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
    orphan files (regex non-match) and missing modality directories.
    '''
    records: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(dict)
    parse_failures: list[dict[str, str]] = []
    action_mismatches: list[dict[str, str]] = []
    orphans: list[dict[str, str]] = []
    missing_modality_dirs: list[str] = []
    unknown_class_dirs: list[dict[str, str]] = []

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

    return {
        'schema_version': '1.0',
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

    df.to_csv(manifest_path, index=False)
    counts.to_csv(counts_path, index=False)
    report_path.write_text(json.dumps(report, indent=2))
    write_label_index(label_path)

    print(f'manifest:           {manifest_path}  ({len(df)} rows)')
    print(f'counts:             {counts_path}')
    print(f'integrity_report:   {report_path}')
    print(f'label_to_index:     {label_path}')
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
            relpath = str(row[PATH_COLUMNS[modality]])
            arr = _read_video(self.dataset_root / relpath, self.frame_count)
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
