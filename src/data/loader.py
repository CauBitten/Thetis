from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODALITY_DIRS: dict[str, str] = {
    'VIDEO_RGB': 'rgb',
    'VIDEO_Depth': 'depth',
    'VIDEO_Mask': 'mask',
    'VIDEO_Skelet2D': 'skeleton_2d',
    'VIDEO_Skelet3D': 'skeleton_3d',
}

ACTION_ALIASES: dict[str, str] = {
    'backhand': 'backhand',
    'backhand_slice': 'backhand_slice',
    'backhand_volley': 'backhand_volley',
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
    'bvolley': 'backhand_volley',
}

ACTION_LABELS: dict[str, str] = {
    'backhand': 'Backhand',
    'backhand2hands': 'Backhand 2 Hands',
    'backhand_slice': 'Backhand Slice',
    'backhand_volley': 'Backhand Volley',
    'flat_service': 'Flat Service',
    'forehand_flat': 'Forehand Flat',
    'forehand_openstands': 'Forehand Open Stance',
    'forehand_slice': 'Forehand Slice',
    'forehand_volley': 'Forehand Volley',
    'kick_service': 'Kick Service',
    'slice_service': 'Slice Service',
    'smash': 'Smash',
}

FILE_EXTENSIONS = {'.avi'}


def canonical_action(raw_value: str) -> str:
    token = raw_value.strip().lower().replace('-', '_').replace(' ', '_')
    token = re.sub(r'_+', '_', token)
    return ACTION_ALIASES.get(token, token)


def parse_actor_and_sequence(stem: str) -> tuple[str | None, int | None, str | None]:
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
    if not action_token:
        return fallback_action_id

    normalized = action_token.strip().lower().replace('-', '_').replace(' ', '_')
    normalized = re.sub(r'_+', '_', normalized)

    canonical = canonical_action(normalized)
    if canonical in ACTION_LABELS:
        return canonical

    for alias in sorted(ACTION_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf'(^|_){re.escape(alias)}($|_)', normalized):
            return ACTION_ALIASES[alias]

    return fallback_action_id


def infer_skill_level(actor_id: str) -> str:
    actor_index = int(actor_id[1:])
    return 'beginner' if actor_index <= 31 else 'expert'


def collect_records(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    missing_modality_dirs: list[str] = []

    for modality_dir_name, modality in MODALITY_DIRS.items():
        modality_root = dataset_root / modality_dir_name
        if not modality_root.exists():
            missing_modality_dirs.append(modality_dir_name)
            continue

        for action_dir in sorted([p for p in modality_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            action_id = canonical_action(action_dir.name)
            action_label = ACTION_LABELS.get(action_id, action_id.replace('_', ' ').title())

            for file_path in sorted(action_dir.iterdir(), key=lambda p: p.name):
                if not file_path.is_file() or file_path.suffix.lower() not in FILE_EXTENSIONS:
                    continue

                actor_id, sequence_index, action_token = parse_actor_and_sequence(file_path.stem)
                if actor_id is None or sequence_index is None:
                    parse_failures.append(
                        {
                            'path': file_path.as_posix(),
                            'reason': 'could_not_parse_actor_or_sequence',
                        }
                    )
                    continue

                actor_index = int(actor_id[1:])
                inferred_action = infer_action_from_token(action_token, fallback_action_id=action_id)
                action_mismatch = inferred_action != action_id

                relpath = file_path.relative_to(dataset_root).as_posix()
                digest = hashlib.sha1(relpath.encode('utf-8')).hexdigest()[:10]
                sample_id = f'{actor_id}_{action_id}_{sequence_index}_{modality}_{digest}'

                records.append(
                    {
                        'sample_id': sample_id,
                        'actor_id': actor_id,
                        'actor_index': actor_index,
                        'skill_level': infer_skill_level(actor_id),
                        'action_id': action_id,
                        'action_label': action_label,
                        'modality': modality,
                        'modality_dir': modality_dir_name,
                        'sequence_index': sequence_index,
                        'action_from_filename': inferred_action,
                        'action_mismatch': action_mismatch,
                        'relative_path': relpath,
                    }
                )

    records.sort(key=lambda row: (row['modality'], row['action_id'], row['actor_id'], row['sequence_index'], row['relative_path']))

    diagnostics = {
        'missing_modality_dirs': sorted(missing_modality_dirs),
        'parse_failures': parse_failures,
    }
    return records, diagnostics


def _partition_values(values: list[str], seed: int, train_frac: float, val_frac: float) -> tuple[set[str], set[str], set[str]]:
    if not values:
        return set(), set(), set()

    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError('train_frac and val_frac must be > 0 and train_frac + val_frac < 1')

    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)

    n_values = len(shuffled)
    n_train = int(round(n_values * train_frac))
    n_val = int(round(n_values * val_frac))

    if n_values >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n_values - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
    else:
        n_train = 1
        n_val = 0

    n_test = n_values - n_train - n_val

    train_values = set(shuffled[:n_train])
    val_values = set(shuffled[n_train : n_train + n_val])
    test_values = set(shuffled[n_train + n_val : n_train + n_val + n_test])

    return train_values, val_values, test_values


def build_cross_subject_split(
    records: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    actor_ids = sorted({str(row['actor_id']) for row in records}, key=lambda x: int(x[1:]))
    train_actors, val_actors, test_actors = _partition_values(actor_ids, seed, train_frac, val_frac)

    split_rows: list[dict[str, Any]] = []
    for row in records:
        actor_id = str(row['actor_id'])
        if actor_id in train_actors:
            split_name = 'train'
        elif actor_id in val_actors:
            split_name = 'val'
        else:
            split_name = 'test'

        split_rows.append(
            {
                'sample_id': row['sample_id'],
                'split': split_name,
                'actor_id': actor_id,
                'action_id': row['action_id'],
                'modality': row['modality'],
            }
        )

    metadata = {
        'train': sorted(train_actors, key=lambda x: int(x[1:])),
        'val': sorted(val_actors, key=lambda x: int(x[1:])),
        'test': sorted(test_actors, key=lambda x: int(x[1:])),
    }
    return split_rows, metadata


def build_cross_action_split(
    records: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    action_ids = sorted({str(row['action_id']) for row in records})
    train_actions, val_actions, test_actions = _partition_values(action_ids, seed, train_frac, val_frac)

    split_rows: list[dict[str, Any]] = []
    for row in records:
        action_id = str(row['action_id'])
        if action_id in train_actions:
            split_name = 'train'
        elif action_id in val_actions:
            split_name = 'val'
        else:
            split_name = 'test'

        split_rows.append(
            {
                'sample_id': row['sample_id'],
                'split': split_name,
                'actor_id': row['actor_id'],
                'action_id': action_id,
                'modality': row['modality'],
            }
        )

    metadata = {
        'train': sorted(train_actions),
        'val': sorted(val_actions),
        'test': sorted(test_actions),
    }
    return split_rows, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f'Cannot write empty CSV: {path}')

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_integrity_report(records: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    modality_counts = Counter(str(row['modality']) for row in records)
    action_counts = Counter(str(row['action_id']) for row in records)
    actor_counts = Counter(str(row['actor_id']) for row in records)

    modality_action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    modality_actor_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in records:
        modality = str(row['modality'])
        action_id = str(row['action_id'])
        actor_id = str(row['actor_id'])
        modality_action_counts[modality][action_id] += 1
        modality_actor_counts[modality][actor_id] += 1

    missing_action_by_modality: dict[str, list[str]] = {}
    known_actions = sorted(action_counts.keys())
    for modality, counts in modality_action_counts.items():
        missing_action_by_modality[modality] = [action for action in known_actions if action not in counts]

    action_mismatch_count = sum(1 for row in records if bool(row['action_mismatch']))

    rgb_count = modality_counts.get('rgb', 0)
    skeleton2d_count = modality_counts.get('skeleton_2d', 0)
    skeleton3d_count = modality_counts.get('skeleton_3d', 0)

    report: dict[str, Any] = {
        'total_samples': len(records),
        'unique_actors': len(actor_counts),
        'unique_actions': len(action_counts),
        'modalities_present': sorted(modality_counts.keys()),
        'counts_by_modality': dict(sorted(modality_counts.items())),
        'counts_by_action': dict(sorted(action_counts.items())),
        'counts_by_modality_action': {
            modality: dict(sorted(action_map.items()))
            for modality, action_map in sorted(modality_action_counts.items())
        },
        'actors_per_modality': {
            modality: len(actor_map)
            for modality, actor_map in sorted(modality_actor_counts.items())
        },
        'missing_actions_by_modality': missing_action_by_modality,
        'action_mismatch_count': action_mismatch_count,
        'skeleton_to_rgb_ratio': {
            'skeleton_2d_over_rgb': (round(skeleton2d_count / rgb_count, 4) if rgb_count else None),
            'skeleton_3d_over_rgb': (round(skeleton3d_count / rgb_count, 4) if rgb_count else None),
        },
        'missing_modality_dirs': diagnostics.get('missing_modality_dirs', []),
        'parse_failure_count': len(diagnostics.get('parse_failures', [])),
        'parse_failures': diagnostics.get('parse_failures', []),
    }
    return report


def write_integrity_table(path: Path, report: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    counts = report.get('counts_by_modality_action', {})

    for modality, action_map in counts.items():
        for action_id, count in action_map.items():
            rows.append(
                {
                    'modality': modality,
                    'action_id': action_id,
                    'count': count,
                }
            )

    rows.sort(key=lambda row: (row['modality'], row['action_id']))
    write_csv(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build THETIS manifest, integrity report, and reproducible splits.'
    )
    parser.add_argument('--input', type=Path, required=True, help='Path to the raw THETIS dataset folder')
    parser.add_argument('--output', type=Path, required=True, help='Path to output data folder')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split generation')
    parser.add_argument(
        '--train-frac',
        type=float,
        default=0.70,
        help='Train fraction used by cross-subject and cross-action splits',
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=0.15,
        help='Validation fraction used by cross-subject and cross-action splits',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset path does not exist: {dataset_root}')

    records, diagnostics = collect_records(dataset_root)
    if not records:
        raise RuntimeError('No records were found in the dataset. Check folder structure and extensions.')

    processed_root = output_root / 'processed'
    splits_root = output_root / 'splits'
    processed_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_root / 'manifest.csv'
    integrity_report_path = processed_root / 'integrity_report.json'
    integrity_table_path = processed_root / 'counts_by_modality_action.csv'

    cross_subject_path = splits_root / 'cross_subject.csv'
    cross_action_path = splits_root / 'cross_action.csv'
    split_metadata_path = splits_root / 'split_metadata.json'

    write_csv(manifest_path, records)

    integrity_report = build_integrity_report(records, diagnostics)
    with integrity_report_path.open('w', encoding='utf-8') as report_file:
        json.dump(integrity_report, report_file, indent=2)

    write_integrity_table(integrity_table_path, integrity_report)

    cross_subject_rows, cross_subject_meta = build_cross_subject_split(
        records=records,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    write_csv(cross_subject_path, cross_subject_rows)

    cross_action_rows, cross_action_meta = build_cross_action_split(
        records=records,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    write_csv(cross_action_path, cross_action_rows)

    split_metadata = {
        'seed': args.seed,
        'train_frac': args.train_frac,
        'val_frac': args.val_frac,
        'test_frac': round(1.0 - args.train_frac - args.val_frac, 6),
        'cross_subject': cross_subject_meta,
        'cross_action': cross_action_meta,
    }
    with split_metadata_path.open('w', encoding='utf-8') as metadata_file:
        json.dump(split_metadata, metadata_file, indent=2)

    print('Preprocessing completed successfully.')
    print(f'Manifest: {manifest_path}')
    print(f'Integrity report: {integrity_report_path}')
    print(f'Integrity table: {integrity_table_path}')
    print(f'Cross-subject split: {cross_subject_path}')
    print(f'Cross-action split: {cross_action_path}')
    print(f'Split metadata: {split_metadata_path}')


if __name__ == '__main__':
    main()
