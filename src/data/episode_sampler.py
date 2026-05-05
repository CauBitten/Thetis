'''Episode sampler for N-way K-shot Few-Shot Action Recognition on THETIS.

The CLI consumes the manifest produced by :mod:`src.data.loader` and writes
JSONL episodes per split under ``data/episodes/{meta_train,meta_val,meta_test}/``.

The :class:`EpisodeSampler` class is reusable at training time for sampling
episodes on-the-fly (avoid pre-serialising) — the same code path is used by
the CLI when writing pre-computed episodes.

Speed-robustness modes:
    ``beginner_to_expert``  support drawn only from p1–p31, query from p32–p55.
    ``expert_to_beginner``  support from p32–p55, query from p1–p31.
    ``both``                writes both episodes_speed_b2e.jsonl and
                            episodes_speed_e2b.jsonl per split alongside the
                            unrestricted episodes.jsonl.
'''
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# When invoked as ``python src/data/episode_sampler.py`` (no package context),
# ensure the repository root is on sys.path so ``from src.data...`` resolves.
if __package__ in (None, ''):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import ACTION_LABELS, manifest_sha1  # noqa: E402

SPEED_MODES: tuple[str, ...] = ('none', 'beginner_to_expert', 'expert_to_beginner', 'both')
SPLIT_NAMES: tuple[str, ...] = ('meta_train', 'meta_val', 'meta_test')


def split_classes(
    all_classes: Sequence[str],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, list[str]]:
    '''Deterministically partition classes into meta-train/val/test (no overlap).'''
    if train_n + val_n + test_n != len(all_classes):
        raise ValueError(
            f'train_n+val_n+test_n ({train_n}+{val_n}+{test_n}) must equal '
            f'len(all_classes) ({len(all_classes)})'
        )
    rng = np.random.default_rng(seed)
    classes = sorted(all_classes)  # stable starting order
    rng.shuffle(classes)
    return {
        'meta_train': sorted(classes[:train_n]),
        'meta_val': sorted(classes[train_n : train_n + val_n]),
        'meta_test': sorted(classes[train_n + val_n :]),
    }


def _seed_for_episode(base_seed: int, split: str, episode_idx: int) -> int:
    '''Hash split name + index to produce a deterministic 32-bit seed.'''
    payload = f'{base_seed}|{split}|{episode_idx}'.encode('utf-8')
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:4], 'big')


class EpisodeSampler:
    '''Sample N-way K-shot episodes deterministically from the THETIS manifest.

    Args:
        manifest_path: ``data/processed/manifest.csv``.
        n_way:         classes per episode.
        k_shot:        support samples per class.
        q_query:       query samples per class.
        splits:        mapping ``{'meta_train': [classes], 'meta_val': [...], 'meta_test': [...]}``
                       — disjoint class lists.
        seed:          base seed for reproducibility (per-episode RNG is derived).
        speed_split:   one of :data:`SPEED_MODES`.
        modality:      optional modality filter; when set, candidates are
                       restricted to manifest rows where the corresponding
                       ``path_<modality>`` column is non-empty.
        strict:        if True, raise when a class can't supply enough samples
                       for an episode; else skip the class and resample.
    '''

    def __init__(
        self,
        manifest_path: str | Path,
        n_way: int,
        k_shot: int,
        q_query: int,
        splits: dict[str, list[str]],
        seed: int,
        speed_split: str = 'none',
        modality: str | None = None,
        strict: bool = False,
    ) -> None:
        if speed_split not in SPEED_MODES:
            raise ValueError(f'speed_split must be one of {SPEED_MODES}, got {speed_split!r}')
        if n_way <= 0 or k_shot <= 0 or q_query <= 0:
            raise ValueError('n_way, k_shot, q_query must all be positive')
        seen: set[str] = set()
        for split_name, class_list in splits.items():
            duplicates = [c for c in class_list if c in seen]
            if duplicates:
                raise ValueError(f'class overlap between splits: {duplicates}')
            seen.update(class_list)

        self.manifest_path = Path(manifest_path)
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)
        self.splits = {k: list(v) for k, v in splits.items()}
        self.seed = int(seed)
        self.speed_split = speed_split
        self.modality = modality
        self.strict = bool(strict)

        df = pd.read_csv(
            self.manifest_path,
            dtype={'actor': str, 'action_code': str},
            keep_default_na=False,
        )
        if modality is not None:
            from src.data.loader import PATH_COLUMNS  # noqa: PLC0415

            if modality not in PATH_COLUMNS:
                raise ValueError(f'unknown modality: {modality}')
            column = PATH_COLUMNS[modality]
            df = df[df[column].astype(str) != ''].reset_index(drop=True)
        self.df = df
        self._by_class: dict[str, pd.DataFrame] = {
            c: df[df['action_label'] == c].reset_index(drop=True) for c in df['action_label'].unique()
        }
        self.manifest_sha1 = manifest_sha1(df)

    # ------------------------------------------------------------------ helpers

    def _candidates_for_class(self, cls: str) -> pd.DataFrame:
        if cls not in self._by_class:
            return pd.DataFrame(columns=self.df.columns)
        return self._by_class[cls]

    def _sample_class(
        self,
        cls: str,
        rng: np.random.Generator,
        speed_mode: str,
    ) -> tuple[list[str], list[str]] | None:
        '''Return ``(support_ids, query_ids)`` or ``None`` if the class can't satisfy K+Q.'''
        candidates = self._candidates_for_class(cls)
        if speed_mode in ('none',):
            ids = candidates['sample_id'].tolist()
            if len(ids) < self.k_shot + self.q_query:
                return None
            chosen = rng.choice(ids, size=self.k_shot + self.q_query, replace=False).tolist()
            return chosen[: self.k_shot], chosen[self.k_shot :]

        if speed_mode == 'beginner_to_expert':
            support_pool = candidates[candidates['skill_level'] == 'beginner']['sample_id'].tolist()
            query_pool = candidates[candidates['skill_level'] == 'expert']['sample_id'].tolist()
        elif speed_mode == 'expert_to_beginner':
            support_pool = candidates[candidates['skill_level'] == 'expert']['sample_id'].tolist()
            query_pool = candidates[candidates['skill_level'] == 'beginner']['sample_id'].tolist()
        else:
            raise ValueError(f'unsupported speed_mode in _sample_class: {speed_mode}')

        if len(support_pool) < self.k_shot or len(query_pool) < self.q_query:
            return None
        support = rng.choice(support_pool, size=self.k_shot, replace=False).tolist()
        query = rng.choice(query_pool, size=self.q_query, replace=False).tolist()
        return support, query

    # ------------------------------------------------------------------ public API

    def sample_episode(
        self,
        split: str,
        episode_idx: int,
        speed_mode: str | None = None,
    ) -> dict[str, Any]:
        '''Sample one episode from ``split``. Deterministic given ``(seed, split, episode_idx)``.'''
        if split not in self.splits:
            raise KeyError(f'unknown split {split!r}; valid: {list(self.splits)}')
        mode = speed_mode or (self.speed_split if self.speed_split != 'both' else 'none')
        episode_seed = _seed_for_episode(self.seed, f'{split}:{mode}', episode_idx)
        rng = np.random.default_rng(episode_seed)

        available_classes = [c for c in self.splits[split] if c in self._by_class]
        if len(available_classes) < self.n_way:
            raise ValueError(
                f'split {split!r} has {len(available_classes)} usable classes, need n_way={self.n_way}'
            )

        chosen_classes: list[str] = []
        support: dict[str, list[str]] = {}
        query: dict[str, list[str]] = {}
        attempts = 0
        max_attempts = max(20, 4 * self.n_way)
        candidate_pool = list(available_classes)
        while len(chosen_classes) < self.n_way and attempts < max_attempts:
            attempts += 1
            remaining = [c for c in candidate_pool if c not in chosen_classes]
            if len(remaining) < self.n_way - len(chosen_classes):
                break
            cls = str(rng.choice(remaining))
            sampled = self._sample_class(cls, rng, mode)
            if sampled is None:
                if self.strict:
                    raise ValueError(
                        f'class {cls!r} in split {split!r} has insufficient samples for '
                        f'K={self.k_shot} Q={self.q_query} (mode={mode})'
                    )
                candidate_pool.remove(cls)
                continue
            support[cls], query[cls] = sampled
            chosen_classes.append(cls)

        if len(chosen_classes) < self.n_way:
            raise ValueError(
                f'could not assemble {self.n_way}-way episode for split {split!r} (mode={mode}); '
                f'only {len(chosen_classes)} classes have enough samples'
            )

        return {
            'episode_id': f'{split}_{mode}_{episode_idx:06d}' if mode != 'none' else f'{split}_{episode_idx:06d}',
            'split': split,
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'q_query': self.q_query,
            'classes': sorted(chosen_classes),
            'support': {cls: support[cls] for cls in sorted(chosen_classes)},
            'query': {cls: query[cls] for cls in sorted(chosen_classes)},
            'speed_split_mode': mode,
            'seed_used': int(episode_seed),
        }

    def iter_episodes(
        self,
        split: str,
        n_episodes: int,
        speed_mode: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        '''Yield ``n_episodes`` deterministic episodes for ``split``.'''
        for i in range(n_episodes):
            yield self.sample_episode(split, i, speed_mode=speed_mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sample N-way K-shot episodes for THETIS FSAR.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-way', type=int, default=5)
    parser.add_argument('--k-shot', type=int, default=5)
    parser.add_argument('--q-query', type=int, default=15)
    parser.add_argument('--episodes-per-split', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-classes', type=int, default=7)
    parser.add_argument('--val-classes', type=int, default=2)
    parser.add_argument('--test-classes', type=int, default=3)
    parser.add_argument('--speed-split', choices=list(SPEED_MODES), default='none')
    parser.add_argument(
        '--modality',
        type=str,
        default=None,
        help="Optional modality filter (e.g. 'rgb'); restrict sampling to rows where this modality is present.",
    )
    parser.add_argument('--strict', action='store_true')
    return parser


def _write_jsonl(path: Path, episodes: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, sort_keys=True))
            fh.write('\n')
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    classes_in_manifest = sorted(pd.read_csv(args.manifest, usecols=['action_label'])['action_label'].unique())
    universe = sorted(set(ACTION_LABELS) | set(classes_in_manifest))
    if set(classes_in_manifest) != set(ACTION_LABELS):
        # Trust the manifest — but warn via stderr-equivalent log.
        print(
            f'warning: manifest classes {classes_in_manifest} differ from ACTION_LABELS {list(ACTION_LABELS)}; '
            'using union for split partitioning.'
        )
    splits = split_classes(universe, args.train_classes, args.val_classes, args.test_classes, args.seed)

    sampler_unrestricted = EpisodeSampler(
        manifest_path=args.manifest,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        splits=splits,
        seed=args.seed,
        speed_split='none',
        modality=args.modality,
        strict=args.strict,
    )

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Pre-flight: any split with fewer classes than n_way is unviable.
    eligible_splits: list[str] = []
    for split in SPLIT_NAMES:
        n_classes = len(splits[split])
        if n_classes < args.n_way:
            print(
                f'warning: skipping split {split!r} — has {n_classes} classes, '
                f'need n_way={args.n_way}. Use --train/--val/--test-classes or lower --n-way.'
            )
            continue
        eligible_splits.append(split)

    counts: dict[str, dict[str, int]] = {split: {} for split in SPLIT_NAMES}

    def _emit(sampler: EpisodeSampler, filename: str) -> None:
        for split in eligible_splits:
            target = output_root / split / filename
            written = _write_jsonl(target, sampler.iter_episodes(split, args.episodes_per_split))
            counts[split][filename] = written

    if args.speed_split in ('none', 'both'):
        _emit(sampler_unrestricted, 'episodes.jsonl')

    if args.speed_split in ('beginner_to_expert', 'both'):
        sampler_b2e = EpisodeSampler(
            manifest_path=args.manifest,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            splits=splits,
            seed=args.seed,
            speed_split='beginner_to_expert',
            modality=args.modality,
            strict=args.strict,
        )
        _emit(sampler_b2e, 'episodes_speed_b2e.jsonl')

    if args.speed_split in ('expert_to_beginner', 'both'):
        sampler_e2b = EpisodeSampler(
            manifest_path=args.manifest,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            splits=splits,
            seed=args.seed,
            speed_split='expert_to_beginner',
            modality=args.modality,
            strict=args.strict,
        )
        _emit(sampler_e2b, 'episodes_speed_e2b.jsonl')

    metadata = {
        'schema_version': '1.0',
        'generated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'manifest_sha1': sampler_unrestricted.manifest_sha1,
        'manifest_path': str(Path(args.manifest).resolve()),
        'seed': int(args.seed),
        'n_way': int(args.n_way),
        'k_shot': int(args.k_shot),
        'q_query': int(args.q_query),
        'episodes_per_split': int(args.episodes_per_split),
        'speed_split_mode': args.speed_split,
        'modality_filter': args.modality,
        'splits': {
            split: {
                'n_classes': len(splits[split]),
                'classes': splits[split],
                'files': counts[split],
                'eligible': split in eligible_splits,
            }
            for split in SPLIT_NAMES
        },
        'split_strategy': 'deterministic_seeded_shuffle',
    }
    (output_root / 'split_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'wrote split_metadata.json under {output_root}')
    return 0


__all__ = [
    'EpisodeSampler',
    'SPEED_MODES',
    'SPLIT_NAMES',
    'main',
    'split_classes',
]


if __name__ == '__main__':
    raise SystemExit(main())
