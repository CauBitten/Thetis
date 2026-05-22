'''Tests for src/models/{protonet, encoders}.py.

The ProtoNet head is tested with a tiny synthetic encoder so we don't depend on
torchvision weights. Encoder tests are guarded behind ``torchvision`` import
availability and ``--run-slow`` (off by default) to keep CI cheap.
'''
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.models.encoders import KINETICS_MEAN, KINETICS_STD, preprocess_video_batch
from src.models.protonet import ProtoNet


# ---------------------------------------------------------------------------
# preprocess_video_batch
# ---------------------------------------------------------------------------


def test_preprocess_uint8_to_normalized() -> None:
    x = torch.zeros((2, 4, 8, 8, 3), dtype=torch.uint8)
    y = preprocess_video_batch(x)
    assert y.shape == (2, 3, 4, 8, 8)
    assert y.dtype == torch.float32
    # All zeros input → output is the negative normalised mean / std (per channel).
    expected = torch.tensor([-m / s for m, s in zip(KINETICS_MEAN, KINETICS_STD)]).view(1, 3, 1, 1, 1)
    assert torch.allclose(y, expected.expand_as(y), atol=1e-6)


def test_preprocess_float_in_zero_one() -> None:
    x = torch.ones((1, 2, 4, 4, 3), dtype=torch.float32)
    y = preprocess_video_batch(x)
    expected = torch.tensor([(1.0 - m) / s for m, s in zip(KINETICS_MEAN, KINETICS_STD)]).view(1, 3, 1, 1, 1)
    assert torch.allclose(y, expected.expand_as(y), atol=1e-6)


def test_preprocess_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match='expected'):
        preprocess_video_batch(torch.zeros((4, 8, 8, 3)))  # missing batch axis


# ---------------------------------------------------------------------------
# ProtoNet — synthetic encoder
# ---------------------------------------------------------------------------


class _IdentityVideoEncoder(nn.Module):
    '''Encoder that maps ``(B, T, H, W, 3)`` to a deterministic D-dim hash.

    Output is the per-channel mean of the video (after permute), giving a
    `D=3` embedding. Good enough to test the head's prototype / distance math.
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x.to(torch.float32).mean(dim=(1, 2, 3))  # (B, 3)


@pytest.fixture
def tiny_protonet() -> ProtoNet:
    torch.manual_seed(0)
    return ProtoNet(_IdentityVideoEncoder())


def _episode_tensors(
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    # One distinct color per class so the embedding (mean RGB) separates them perfectly.
    class_colors = (rng.integers(20, 235, size=(n_way, 3))).astype(np.uint8)
    support = np.empty((n_way * k_shot, 4, 8, 8, 3), dtype=np.uint8)
    support_labels = np.empty((n_way * k_shot,), dtype=np.int64)
    for c in range(n_way):
        for k in range(k_shot):
            support[c * k_shot + k] = class_colors[c]
            support_labels[c * k_shot + k] = c
    query = np.empty((n_way * q_query, 4, 8, 8, 3), dtype=np.uint8)
    query_labels = np.empty((n_way * q_query,), dtype=np.int64)
    for c in range(n_way):
        for q in range(q_query):
            jitter = rng.integers(-3, 4, size=3).astype(np.int16)
            query[c * q_query + q] = np.clip(class_colors[c].astype(np.int16) + jitter, 0, 255).astype(np.uint8)
            query_labels[c * q_query + q] = c
    return (
        torch.from_numpy(support),
        torch.from_numpy(query),
        torch.from_numpy(support_labels),
        torch.from_numpy(query_labels),
    )


def test_protonet_perfect_accuracy_on_separable_episode(tiny_protonet: ProtoNet) -> None:
    support, query, support_labels, query_labels = _episode_tensors(n_way=5, k_shot=5, q_query=15)
    out = tiny_protonet(support, query, support_labels, query_labels, n_way=5)
    assert out['logits'].shape == (5 * 15, 5)
    assert out['preds'].shape == (5 * 15,)
    assert out['prototypes'].shape == (5, 3)
    assert out['accuracy'] == pytest.approx(1.0, abs=1e-6)
    assert torch.isfinite(out['loss']).item()


def test_protonet_loss_has_gradient(tiny_protonet: ProtoNet) -> None:
    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.to(torch.float32).mean(dim=(1, 2, 3)))

    model = ProtoNet(_Tiny())
    support, query, support_labels, query_labels = _episode_tensors(n_way=3, k_shot=2, q_query=3)
    out = model(support, query, support_labels, query_labels, n_way=3)
    out['loss'].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_protonet_rejects_mismatched_shapes(tiny_protonet: ProtoNet) -> None:
    support = torch.zeros((4, 2, 4, 4, 3), dtype=torch.uint8)
    query = torch.zeros((4, 2, 4, 4, 3), dtype=torch.uint8)
    bad_labels = torch.zeros((3,), dtype=torch.long)
    good_labels = torch.zeros((4,), dtype=torch.long)
    with pytest.raises(ValueError, match='support_labels'):
        tiny_protonet(support, query, bad_labels, good_labels, n_way=2)
    with pytest.raises(ValueError, match='query_labels'):
        tiny_protonet(support, query, good_labels, bad_labels, n_way=2)


def test_protonet_class_with_no_support_raises(tiny_protonet: ProtoNet) -> None:
    # n_way=3 but support_labels only cover classes 0, 1 — class 2 has no prototype.
    support = torch.zeros((4, 2, 4, 4, 3), dtype=torch.uint8)
    query = torch.zeros((6, 2, 4, 4, 3), dtype=torch.uint8)
    support_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    query_labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    with pytest.raises(ValueError, match='no support samples'):
        tiny_protonet(support, query, support_labels, query_labels, n_way=3)


def test_protonet_determinism_same_seed() -> None:
    s, q, sl, ql = _episode_tensors(n_way=3, k_shot=2, q_query=5, seed=7)
    model_a = ProtoNet(_IdentityVideoEncoder())
    model_b = ProtoNet(_IdentityVideoEncoder())
    out_a = model_a(s, q, sl, ql, n_way=3)
    out_b = model_b(s, q, sl, ql, n_way=3)
    assert torch.equal(out_a['preds'], out_b['preds'])
    assert torch.allclose(out_a['logits'], out_b['logits'])
