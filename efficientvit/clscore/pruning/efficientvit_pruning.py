"""EfficientViT classification 모델용 Soft Pruning 구현.

PRUNING_METHODOLOGY.md 의 방법론에 따라, 매 optimizer.step() 직후
호출되어 L2 norm 하위 X% 의 weight 를 0 으로 마스킹한다.

Prunable 그룹:
  - MBConv 의 mid_channels (inverted_conv → depth_conv → point_conv 의 hidden dim)
  - FusedMBConv 의 mid_channels (spatial_conv → point_conv 의 hidden dim)
  - **Input Stem chain** (stage0 채널 = width_list[0])
      B-series: input_stem (Conv + DSConv ResidualBlock × n)
      L-series: stages[0]   (Conv + ResBlock ResidualBlock × n)
      Stem 의 모든 레이어가 동일한 채널 수(width_list[0]) 를 공유하므로
      single index 를 chain 전체 + 다음 stage 첫 conv 의 입력 컬럼에 동기화.

Non-prunable (외부 채널과 직결되어 건드리지 않음):
  - LiteMLA (multi-scale aggreg 와 결합도가 높아 제외)
  - ClsHead (정확도 보존을 위해 제외)
  - Stage 사이 채널 경계 (width_list[1..4])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    FusedMBConv,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)

__all__ = ["EfficientViTPruner"]


# 너무 작은 hidden dim 이 0 이 되어 정보가 완전히 끊기지 않도록 최소 생존 채널 수.
MIN_SURVIVE = 4


def _bn_param_count(bn: nn.Module | None) -> int:
    """BN(혹은 LN) 의 학습 파라미터 수 (running stat 제외, weight + bias)."""
    if bn is None:
        return 0
    n = 0
    if getattr(bn, "weight", None) is not None:
        n += bn.weight.numel()
    if getattr(bn, "bias", None) is not None:
        n += bn.bias.numel()
    return n


def _conv_param_count(conv: nn.Conv2d) -> int:
    n = conv.weight.numel()
    if conv.bias is not None:
        n += conv.bias.numel()
    return n


def _calc_n_prune(n_total: int, sparsity: float) -> int:
    """sparsity 비율에 따라 제거할 채널 수. MIN_SURVIVE 만큼은 반드시 살린다."""
    n_prune = round(n_total * sparsity)
    n_prune = min(n_prune, n_total - MIN_SURVIVE)
    return max(n_prune, 0)


def _zero_bn_(bn: nn.Module | None, idx: torch.Tensor) -> None:
    """BN 의 idx 채널들을 안전 마스킹 (running_var 는 1.0 로 — 분모 0 방지)."""
    if bn is None:
        return
    with torch.no_grad():
        if getattr(bn, "weight", None) is not None:
            bn.weight.data[idx] = 0.0
        if getattr(bn, "bias", None) is not None:
            bn.bias.data[idx] = 0.0
        if isinstance(bn, _BatchNorm):
            bn.running_mean.data[idx] = 0.0
            bn.running_var.data[idx] = 1.0


def _topk_smallest_l2_idx(weight: torch.Tensor, k: int) -> torch.Tensor:
    """첫 차원(필터 축) L2 norm 이 가장 작은 k 개의 인덱스."""
    n = weight.shape[0]
    norms = torch.norm(weight.detach().reshape(n, -1), dim=1)
    _, idx = torch.topk(norms, k, largest=False)
    return idx


# ---------------------------------------------------------------------------
# Prunable 그룹 단위 1) 마스킹  2) 제거 파라미터 추정 함수.
# 두 함수는 항상 동일한 sparsity 가 같은 인덱스 / 같은 제거량을 만들도록 짝지어 둔다.
# ---------------------------------------------------------------------------


def _prune_mbconv(mb: MBConv, sparsity: float) -> None:
    """MBConv mid_channels(=inverted_conv 출력) 을 sparsity 비율로 마스킹."""
    weight = mb.inverted_conv.conv.weight  # (mid, in, 1, 1)
    mid = weight.shape[0]
    n_prune = _calc_n_prune(mid, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)

    with torch.no_grad():
        # inverted_conv: 출력 필터 마스킹.
        mb.inverted_conv.conv.weight.data[idx] = 0.0
        if mb.inverted_conv.conv.bias is not None:
            mb.inverted_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(mb.inverted_conv.norm, idx)

        # depth_conv: groups=mid 이므로 동일 인덱스가 그대로 채널/그룹 인덱스.
        mb.depth_conv.conv.weight.data[idx] = 0.0
        if mb.depth_conv.conv.bias is not None:
            mb.depth_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(mb.depth_conv.norm, idx)

        # point_conv: 입력 컬럼(=mid 축) 마스킹. 출력은 외부 채널이라 건드리지 않음.
        mb.point_conv.conv.weight.data[:, idx] = 0.0


def _prune_fusedmbconv(fmb: FusedMBConv, sparsity: float) -> None:
    """FusedMBConv mid_channels(=spatial_conv 출력) 을 sparsity 비율로 마스킹."""
    weight = fmb.spatial_conv.conv.weight  # (mid, in/groups, k, k)
    mid = weight.shape[0]
    n_prune = _calc_n_prune(mid, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)

    with torch.no_grad():
        fmb.spatial_conv.conv.weight.data[idx] = 0.0
        if fmb.spatial_conv.conv.bias is not None:
            fmb.spatial_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(fmb.spatial_conv.norm, idx)

        # point_conv 입력 컬럼.
        fmb.point_conv.conv.weight.data[:, idx] = 0.0


def _estimate_removed_mbconv(mb: MBConv, sparsity: float) -> int:
    """sparsity 적용 시 MBConv 에서 제거되는 파라미터 수 (이차 효과 반영)."""
    weight = mb.inverted_conv.conv.weight
    mid, in_ch = weight.shape[0], weight.shape[1]
    out_ch = mb.point_conv.conv.weight.shape[0]
    k = mb.depth_conv.conv.weight.shape[-1]

    n_prune = _calc_n_prune(mid, sparsity)
    if n_prune <= 0:
        return 0

    removed = 0
    # inverted_conv: 출력 필터 → (in_ch * 1 * 1) 만큼씩 사라짐 + (옵션) bias.
    removed += n_prune * in_ch
    if mb.inverted_conv.conv.bias is not None:
        removed += n_prune
    removed += n_prune * (1 if isinstance(mb.inverted_conv.norm, _BatchNorm) else 0) * 2

    # depth_conv: groups=mid 이므로 채널 1개 = (1 * k * k) 만큼.
    removed += n_prune * k * k
    if mb.depth_conv.conv.bias is not None:
        removed += n_prune
    removed += n_prune * (1 if isinstance(mb.depth_conv.norm, _BatchNorm) else 0) * 2

    # point_conv: 입력 컬럼 → out_ch 개의 슬롯을 잃음.
    removed += out_ch * n_prune
    return removed


def _estimate_removed_fusedmbconv(fmb: FusedMBConv, sparsity: float) -> int:
    weight = fmb.spatial_conv.conv.weight
    mid, in_per_group = weight.shape[0], weight.shape[1]
    k = weight.shape[-1]
    out_ch = fmb.point_conv.conv.weight.shape[0]

    n_prune = _calc_n_prune(mid, sparsity)
    if n_prune <= 0:
        return 0

    removed = 0
    removed += n_prune * in_per_group * k * k
    if fmb.spatial_conv.conv.bias is not None:
        removed += n_prune
    removed += n_prune * (1 if isinstance(fmb.spatial_conv.norm, _BatchNorm) else 0) * 2

    removed += out_ch * n_prune
    return removed


# ---------------------------------------------------------------------------
# Input Stem (stage0) — chain pruning
# ---------------------------------------------------------------------------


def _zero_conv_out_filters_(cl: ConvLayer, idx: torch.Tensor) -> None:
    """ConvLayer 출력 필터 + BN 채널을 한 번에 마스킹."""
    with torch.no_grad():
        cl.conv.weight.data[idx] = 0.0
        if cl.conv.bias is not None:
            cl.conv.bias.data[idx] = 0.0
    _zero_bn_(cl.norm, idx)


def _zero_conv_in_cols_(cl: ConvLayer, idx: torch.Tensor) -> None:
    """ConvLayer 입력 컬럼만 마스킹 (출력 채널/BN 은 손대지 않음)."""
    with torch.no_grad():
        cl.conv.weight.data[:, idx] = 0.0


def _get_stem_op_seq(model: nn.Module) -> OpSequential | None:
    """B-series 의 input_stem 또는 L-series 의 stages[0] 을 반환. 없으면 None."""
    bb = getattr(model, "backbone", None)
    if bb is None:
        return None
    if hasattr(bb, "input_stem"):
        return bb.input_stem
    if hasattr(bb, "stages") and len(bb.stages) > 0:
        return bb.stages[0]
    return None


def _get_post_stem_first_conv(model: nn.Module) -> ConvLayer | None:
    """Stem 직후 첫 stage 의 첫 down-sampling 블록의 첫 ConvLayer 를 반환.

    이 ConvLayer 의 입력 컬럼이 stem 출력 채널과 직결되므로,
    stem chain 마스킹 시 함께 입력 컬럼을 마스킹해야 한다.
    """
    bb = getattr(model, "backbone", None)
    if bb is None or not hasattr(bb, "stages") or len(bb.stages) == 0:
        return None
    # B-series: input_stem 이 따로 있으므로 stages[0] 가 stage1.
    # L-series: stages[0] 이 stem 이므로 stages[1] 가 stage1.
    stage1_idx = 0 if hasattr(bb, "input_stem") else 1
    if stage1_idx >= len(bb.stages):
        return None
    stage1 = bb.stages[stage1_idx]
    if not hasattr(stage1, "op_list") or len(stage1.op_list) == 0:
        return None
    first = stage1.op_list[0]  # ResidualBlock(MBConv|FusedMBConv, shortcut=None)
    main = getattr(first, "main", first)
    if isinstance(main, MBConv):
        return main.inverted_conv
    if isinstance(main, FusedMBConv):
        return main.spatial_conv
    return None


def _iter_stem_inner_blocks(stem: OpSequential):
    """Stem 의 op_list[1..] 가 ResidualBlock(DSConv|ResBlock, Identity) 일 때
    내부의 (DSConv | ResBlock) 만 순회한다.
    """
    for op in stem.op_list[1:]:
        main = getattr(op, "main", op)
        if isinstance(main, (DSConv, ResBlock)):
            yield main


def _prune_input_stem(model: nn.Module, sparsity: float) -> None:
    """Stem chain (input_stem 또는 stages[0]) 을 single-index 로 마스킹.

    구조:
      stem.op_list[0] = ConvLayer(3 → C0)                 ← 출력 필터 = C0
      stem.op_list[1..] = ResidualBlock(DSConv|ResBlock, Identity)
        DSConv:    .depth_conv (groups=C0) + .point_conv (in=out=C0)
        ResBlock:  .conv1 (in=out=C0)     + .conv2 (in=out=C0)
      이후 stage1 의 첫 down-sampling Conv 입력 컬럼 (C0) 도 동기화.
    """
    stem = _get_stem_op_seq(model)
    if stem is None or not hasattr(stem, "op_list") or len(stem.op_list) == 0:
        return
    first_cl = stem.op_list[0]
    if not isinstance(first_cl, ConvLayer):
        return

    weight = first_cl.conv.weight  # (C0, 3, k, k)
    n_total = weight.shape[0]
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)

    # stem[0]: 출력 필터.
    _zero_conv_out_filters_(first_cl, idx)

    # stem[1..]: 각 inner DSConv/ResBlock 은 in==out==C0 라 양방향 동기화.
    for inner in _iter_stem_inner_blocks(stem):
        if isinstance(inner, DSConv):
            # depth_conv: groups=C0 → 같은 idx 가 채널 = 그룹 인덱스.
            _zero_conv_out_filters_(inner.depth_conv, idx)
            # point_conv: in / out 둘 다 C0.
            _zero_conv_out_filters_(inner.point_conv, idx)
            _zero_conv_in_cols_(inner.point_conv, idx)
        else:  # ResBlock
            _zero_conv_out_filters_(inner.conv1, idx)
            _zero_conv_in_cols_(inner.conv1, idx)
            _zero_conv_out_filters_(inner.conv2, idx)
            _zero_conv_in_cols_(inner.conv2, idx)

    # stem 출력 채널 → 다음 stage 첫 down-sampling Conv 입력 컬럼 동기화.
    post = _get_post_stem_first_conv(model)
    if post is not None:
        _zero_conv_in_cols_(post, idx)


def _estimate_removed_input_stem(model: nn.Module, sparsity: float) -> int:
    """Stem chain pruning 시 제거되는 파라미터 수 (이차 효과 포함).

    포함 항목:
      - stem[0]: out filter * (3 * k * k) + BN
      - 각 DSConv/ResBlock:
          DSConv:   depth_conv (k×k, groups=C0) + point_conv (in=out=C0, 양방향)
          ResBlock: conv1 (in=out=C0, 양방향)   + conv2 (in=out=C0, 양방향)
      - 다음 stage 첫 conv 의 입력 컬럼: n_C0 * out_ch_first
        (n_C0 * n_mid 만큼의 cross-term 은 무시 → 약간의 over-estimate, 안전 방향)
    """
    stem = _get_stem_op_seq(model)
    if stem is None or not hasattr(stem, "op_list") or len(stem.op_list) == 0:
        return 0
    first_cl = stem.op_list[0]
    if not isinstance(first_cl, ConvLayer):
        return 0

    weight = first_cl.conv.weight
    n_total, in_ch_image = weight.shape[0], weight.shape[1]
    k0 = weight.shape[-1]
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return 0
    n_surv = n_total - n_prune

    removed = 0
    # stem[0] out filter.
    removed += n_prune * in_ch_image * k0 * k0
    if first_cl.conv.bias is not None:
        removed += n_prune
    if isinstance(first_cl.norm, _BatchNorm):
        removed += n_prune * 2

    # Inner blocks.
    for inner in _iter_stem_inner_blocks(stem):
        if isinstance(inner, DSConv):
            kd = inner.depth_conv.conv.weight.shape[-1]
            # depth_conv: (C0, 1, k, k) groups=C0
            removed += n_prune * kd * kd
            if inner.depth_conv.conv.bias is not None:
                removed += n_prune
            if isinstance(inner.depth_conv.norm, _BatchNorm):
                removed += n_prune * 2
            # point_conv: (C0, C0, 1, 1) — 양방향 축소
            removed += n_total * n_total - n_surv * n_surv
            if inner.point_conv.conv.bias is not None:
                removed += n_prune  # bias 는 출력 채널 기준
            if isinstance(inner.point_conv.norm, _BatchNorm):
                removed += n_prune * 2
        else:  # ResBlock
            kr1 = inner.conv1.conv.weight.shape[-1]
            kr2 = inner.conv2.conv.weight.shape[-1]
            # conv1, conv2 둘 다 (C0, C0, k, k) — 양방향
            removed += (n_total * n_total - n_surv * n_surv) * kr1 * kr1
            if inner.conv1.conv.bias is not None:
                removed += n_prune
            if isinstance(inner.conv1.norm, _BatchNorm):
                removed += n_prune * 2
            removed += (n_total * n_total - n_surv * n_surv) * kr2 * kr2
            if inner.conv2.conv.bias is not None:
                removed += n_prune
            if isinstance(inner.conv2.norm, _BatchNorm):
                removed += n_prune * 2

    # 다음 stage 첫 conv 의 입력 컬럼 (n_C0 × out_ch_first).
    post = _get_post_stem_first_conv(model)
    if post is not None:
        post_w = post.conv.weight  # (out_first, C0, k, k)
        out_first = post_w.shape[0]
        kp = post_w.shape[-1]
        removed += n_prune * out_first * kp * kp

    return removed


# ---------------------------------------------------------------------------
# 모델 전체 enumerate / 추정 / sparsity 이진탐색
# ---------------------------------------------------------------------------


def _iter_prunable_modules(model: nn.Module):
    """(kind, module) 튜플을 yield. kind ∈ {'mbconv', 'fmbconv'}."""
    for module in model.modules():
        if isinstance(module, MBConv):
            yield ("mbconv", module)
        elif isinstance(module, FusedMBConv):
            yield ("fmbconv", module)


def _count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_total_removed(model: nn.Module, sparsity: float) -> int:
    total = 0
    for kind, mod in _iter_prunable_modules(model):
        if kind == "mbconv":
            total += _estimate_removed_mbconv(mod, sparsity)
        elif kind == "fmbconv":
            total += _estimate_removed_fusedmbconv(mod, sparsity)
    total += _estimate_removed_input_stem(model, sparsity)
    return total


def _find_sparsity_by_bisection(
    model: nn.Module,
    target_compression: float,
    max_sparsity: float = 0.95,
    iters: int = 64,
) -> float:
    """이진탐색으로 target_compression 비율을 만족시키는 per-group sparsity 를 찾는다."""
    if target_compression <= 0:
        return 0.0
    total = _count_total_params(model)
    target_remove = target_compression * total
    lo, hi = 0.0, max_sparsity
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Pruner 클래스
# ---------------------------------------------------------------------------


class EfficientViTPruner:
    """학습 중 매 step 적용되는 Soft Pruning 컨트롤러.

    사용 예:
        pruner = EfficientViTPruner(model, target_compression=0.30)
        # ... 학습 루프 안에서, optimizer.step() 직후:
        pruner.apply(model)
    """

    def __init__(
        self,
        model: nn.Module,
        target_compression: float,
        max_sparsity: float = 0.95,
        sparsity: float | None = None,
    ) -> None:
        self.target_compression = float(target_compression)
        self.max_sparsity = float(max_sparsity)
        if sparsity is not None:
            self.sparsity = float(sparsity)
        else:
            self.sparsity = _find_sparsity_by_bisection(
                model, self.target_compression, self.max_sparsity
            )

        # 간단한 통계 로그.
        n_groups = sum(1 for _ in _iter_prunable_modules(model))
        total = _count_total_params(model)
        est_removed = _estimate_total_removed(model, self.sparsity)
        rate = 100.0 * est_removed / max(total, 1)
        print(
            f"[EfficientViTPruner] target={self.target_compression*100:.1f}% "
            f"per-group sparsity={self.sparsity:.4f} "
            f"prunable_groups={n_groups} estimated_compression={rate:.2f}%"
        )

    def apply(self, model: nn.Module) -> None:
        """모든 prunable 그룹에 대해 한 번 마스킹."""
        if self.sparsity <= 0:
            return
        for kind, mod in _iter_prunable_modules(model):
            if kind == "mbconv":
                _prune_mbconv(mod, self.sparsity)
            elif kind == "fmbconv":
                _prune_fusedmbconv(mod, self.sparsity)
        _prune_input_stem(model, self.sparsity)

    @torch.no_grad()
    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """실제 마스킹된 zero 비율 (검증용). MBConv/FusedMBConv mid + Stem 출력 채널."""
        n_total = 0
        n_zero = 0
        for kind, mod in _iter_prunable_modules(model):
            if kind == "mbconv":
                w = mod.inverted_conv.conv.weight
            else:
                w = mod.spatial_conv.conv.weight
            n_filt = w.shape[0]
            norms = torch.norm(w.detach().reshape(n_filt, -1), dim=1)
            n_total += n_filt
            n_zero += int((norms == 0).sum().item())
        # Stem 첫 conv 의 출력 채널 zero 도 함께 집계.
        stem = _get_stem_op_seq(model)
        if stem is not None and len(stem.op_list) > 0 and isinstance(stem.op_list[0], ConvLayer):
            w = stem.op_list[0].conv.weight
            n_filt = w.shape[0]
            norms = torch.norm(w.detach().reshape(n_filt, -1), dim=1)
            n_total += n_filt
            n_zero += int((norms == 0).sum().item())
        return {
            "prunable_filters": n_total,
            "zero_filters": n_zero,
            "actual_sparsity": n_zero / max(n_total, 1),
            "target_sparsity": self.sparsity,
        }
