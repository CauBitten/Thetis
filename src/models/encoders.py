'''Video encoders used as backbones for FSAR methods.

The Phase-2 baseline uses :class:`VideoEncoder` wrapping
``torchvision.models.video.r2plus1d_18`` pre-trained on Kinetics-400, with
its classifier head replaced by an identity (so the wrapper returns
``(B, embed_dim)`` features). All FSAR heads (ProtoNet, TRX, ...) consume
this same embedding shape.

The wrapper accepts ``(B, T, H, W, 3)`` uint8 input — the same layout
returned by :class:`src.data.loader.ThetisDataset` — and handles the
permute + Kinetics normalisation internally via
:func:`preprocess_video_batch`.

Gradient checkpointing (``use_checkpointing=True``) trades ~30% extra
compute for ~70% less activation memory by re-running each ResNet stage
during the backward pass. Essential on ≤6 GB GPUs.
'''
from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

# Kinetics-400 normalisation statistics (torchvision R(2+1)D / R3D defaults).
KINETICS_MEAN: tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
KINETICS_STD: tuple[float, float, float] = (0.22803, 0.22145, 0.216989)


def preprocess_video_batch(x: torch.Tensor) -> torch.Tensor:
    '''Convert ``(B, T, H, W, 3)`` uint8/float into ``(B, 3, T, H, W)`` float32, normalized.

    Accepts uint8 (divides by 255) or float (assumed in [0, 1] or [0, 255] —
    inferred from max). Normalises with Kinetics statistics.
    '''
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError(f'expected (B, T, H, W, 3) tensor, got shape {tuple(x.shape)}')
    # Target dtype honours an enclosing autocast: under AMP the clip enters the
    # backbone in fp16 (half the bytes) instead of a full fp32 copy the conv
    # would immediately re-cast anyway. Falls back to fp32 (no autocast / CPU).
    out_dtype = torch.float32
    if x.is_cuda and torch.is_autocast_enabled():
        out_dtype = (
            torch.get_autocast_dtype('cuda')
            if hasattr(torch, 'get_autocast_dtype')
            else torch.get_autocast_gpu_dtype()
        )
    # uint8 always scales by 255; float is assumed in [0, 1] unless it looks like
    # 0-255 (short-circuits so the uint8 path never pays the max() device sync).
    divide = x.dtype == torch.uint8 or (x.numel() > 0 and float(x.max()) > 1.5)
    # Single cast+copy straight into channel-first contiguous layout (B, 3, T, H, W),
    # then normalise in place — no extra full-size temporaries.
    y = x.permute(0, 4, 1, 2, 3).to(out_dtype, memory_format=torch.contiguous_format)
    if divide:
        y = y.div_(255.0)
    mean = torch.tensor(KINETICS_MEAN, device=y.device, dtype=y.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, device=y.device, dtype=y.dtype).view(1, 3, 1, 1, 1)
    return y.sub_(mean).div_(std)


class VideoEncoder(nn.Module):
    '''Wrapper around a torchvision video backbone returning ``(B, embed_dim)``.

    Args:
        name: backbone name. Currently ``'r2plus1d_18'`` (default) and
            ``'r3d_18'`` (fallback). Both expose a 512-dim feature.
        pretrained: load Kinetics-400 weights via torchvision's enum API.
            If the download fails, prints a warning and continues with
            random init.
        use_checkpointing: if True, applies gradient checkpointing on
            ``layer1..layer4`` of the ResNet to slash activation memory.
            Off by default (turn on for small GPUs / large batches).
    '''

    def __init__(
        self,
        name: str = 'r2plus1d_18',
        pretrained: bool = True,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        backbone, embed_dim = _load_backbone(name, pretrained=pretrained)
        self.name = name
        self.backbone = backbone
        self.embed_dim: int = embed_dim
        self.use_checkpointing = bool(use_checkpointing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''``(B, T, H, W, 3)`` uint8/float in → ``(B, embed_dim)`` float32 out.'''
        x = preprocess_video_batch(x)
        if not self.use_checkpointing or not torch.is_grad_enabled():
            return self.backbone(x)
        return self._forward_checkpointed(x)

    def _forward_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        '''Per-stage checkpoint: stem → layer1 → ... → layer4 → avgpool → fc.

        Each block's activations are re-computed during backward instead of
        kept in memory. Cuts activation memory ~70% at the cost of one extra
        forward per stage.
        '''
        b = self.backbone
        x = _grad_checkpoint(b.stem, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer1, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer2, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer3, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer4, x, use_reentrant=False)
        x = b.avgpool(x)
        x = x.flatten(1)
        return b.fc(x)


def _load_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import video as tv_video  # noqa: PLC0415

    name = name.lower()
    if name == 'r2plus1d_18':
        weights_enum = tv_video.R2Plus1D_18_Weights
        ctor = tv_video.r2plus1d_18
    elif name == 'r3d_18':
        weights_enum = tv_video.R3D_18_Weights
        ctor = tv_video.r3d_18
    else:
        raise ValueError(f'unknown video backbone: {name!r}')

    weights = weights_enum.KINETICS400_V1 if pretrained else None
    try:
        model = ctor(weights=weights)
    except Exception as exc:  # noqa: BLE001 — pretrained download can fail in many ways
        if pretrained:
            warnings.warn(f'failed to load pretrained {name}: {exc}; falling back to random init', stacklevel=2)
            model = ctor(weights=None)
        else:
            raise

    embed_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, embed_dim


__all__ = ['KINETICS_MEAN', 'KINETICS_STD', 'VideoEncoder', 'preprocess_video_batch']
