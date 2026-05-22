'''Prototypical Networks (Snell et al. 2017) for few-shot action recognition.

Given a single episode with ``N`` classes, ``K`` support samples per class and
``Q`` query samples per class, the head:

1. encodes support and query in one forward pass (``encoder(...)`` → ``(N*(K+Q), D)``);
2. averages each class's K support embeddings → prototypes ``(N, D)``;
3. classifies each query by ``-‖q - c_i‖²`` as logits (negative squared L2);
4. cross-entropy against ``[0, N)`` labels.

The encoder is decoupled — pass any ``nn.Module`` that maps
``(B, T, H, W, 3)`` videos to ``(B, D)`` embeddings (e.g.
:class:`src.models.encoders.VideoEncoder`). Tests inject a tiny dummy.
'''
from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F


class ProtoNetOutput(TypedDict):
    logits: torch.Tensor   # (N*Q, N)
    loss: torch.Tensor     # scalar
    accuracy: float        # scalar in [0, 1]
    preds: torch.Tensor    # (N*Q,) int64
    prototypes: torch.Tensor  # (N, D)


class ProtoNet(nn.Module):
    '''ProtoNet head wrapping an arbitrary video encoder.

    Args:
        encoder: ``nn.Module`` with ``forward(x: (B, T, H, W, 3)) -> (B, D)``.

    Call signature:
        ``forward(support, query, support_labels, query_labels, n_way)`` where
        ``support`` is ``(N*K, T, H, W, 3)``, ``query`` is ``(N*Q, T, H, W, 3)``,
        and labels are int tensors in ``[0, N)``. Returns a :class:`ProtoNetOutput`
        dict so callers can choose to backprop on ``loss`` and log ``accuracy``.
    '''

    def __init__(self, encoder: nn.Module, encoder_batch_size: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_batch_size = encoder_batch_size

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: torch.Tensor,
        query_labels: torch.Tensor,
        n_way: int,
        encoder_batch_size: int | None = None,
    ) -> ProtoNetOutput:
        if support.ndim != 5 or query.ndim != 5:
            raise ValueError(
                f'expected (B,T,H,W,3) tensors; got support={tuple(support.shape)}, query={tuple(query.shape)}'
            )
        n_support = support.shape[0]
        n_query = query.shape[0]
        if support_labels.shape != (n_support,):
            raise ValueError(f'support_labels shape {tuple(support_labels.shape)} != ({n_support},)')
        if query_labels.shape != (n_query,):
            raise ValueError(f'query_labels shape {tuple(query_labels.shape)} != ({n_query},)')

        combined = torch.cat([support, query], dim=0)
        batch_size = encoder_batch_size if encoder_batch_size is not None else self.encoder_batch_size
        if batch_size is None or batch_size >= combined.shape[0]:
            embeddings = self.encoder(combined)
        else:
            # Chunked encode keeps activations bounded — critical on CPU / small GPUs
            # where R(2+1)D-18 with batch=100 explodes RAM.
            chunks: list[torch.Tensor] = []
            for start in range(0, combined.shape[0], batch_size):
                chunks.append(self.encoder(combined[start : start + batch_size]))
            embeddings = torch.cat(chunks, dim=0)
        if embeddings.ndim != 2:
            raise ValueError(f'encoder must return (B, D); got shape {tuple(embeddings.shape)}')

        support_emb = embeddings[:n_support]
        query_emb = embeddings[n_support:]

        embed_dim = support_emb.shape[1]
        prototypes = torch.zeros(n_way, embed_dim, device=support_emb.device, dtype=support_emb.dtype)
        for c in range(n_way):
            mask = support_labels == c
            count = int(mask.sum().item())
            if count == 0:
                raise ValueError(f'class {c} has no support samples in this episode')
            prototypes[c] = support_emb[mask].mean(dim=0)

        # Negative squared L2 distance: (N*Q, N)
        diffs = query_emb.unsqueeze(1) - prototypes.unsqueeze(0)
        logits = -(diffs * diffs).sum(dim=-1)
        loss = F.cross_entropy(logits, query_labels)
        preds = logits.argmax(dim=-1)
        accuracy = float((preds == query_labels).float().mean().item())

        return {
            'logits': logits,
            'loss': loss,
            'accuracy': accuracy,
            'preds': preds,
            'prototypes': prototypes,
        }


__all__ = ['ProtoNet', 'ProtoNetOutput']
