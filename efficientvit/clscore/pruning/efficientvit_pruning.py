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
  - Stage 사이 채널 경계 (width_list[1..4])

성능 최적화:
  - init 시 모든 prunable 텐서 레퍼런스를 _PruneGroup 으로 수집 (매 step model.modules() 순회 제거).
  - 인덱스 캐싱: 기본 100 step 마다만 topk 재계산 (index_refresh_steps).
  - weight *= mask 벡터화: fancy-index scatter 대신 단일 elementwise mul (CUDA 커널 수 감소).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

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


# ---------------------------------------------------------------------------
# 기본 수치 헬퍼
# ---------------------------------------------------------------------------


def _bn_param_count(bn: nn.Module | None) -> int:
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


def _topk_smallest_l2_idx(weight: torch.Tensor, k: int) -> torch.Tensor:
    """첫 차원(필터 축) L2 norm 이 가장 작은 k 개의 인덱스."""
    n = weight.shape[0]
    norms = torch.norm(weight.detach().reshape(n, -1), dim=1)
    _, idx = torch.topk(norms, k, largest=False)
    return idx


# ---------------------------------------------------------------------------
# _PruneGroup: 단일 prunable 그룹의 텐서 레퍼런스 + 마스크 캐시
#
# 설계 원칙:
#   - criterion : L2 norm ranking 에 쓰이는 기준 weight (inverted_conv 또는 spatial_conv).
#   - targets   : 마스킹 대상 (tensor, dim, fill_value) 튜플 리스트.
#                 fill_value=0.0 → 0 으로 덮어쓰기, =1.0 → pruned 채널만 1 로 (running_var 용).
#   - _mask     : (n,) float32 GPU tensor. 1=alive, 0=pruned. None 이면 아직 미계산.
#
# apply() 내부에서 tensor *= mask 로 처리하므로 Python-level 루프가 없고
# CUDA 커널 수가 대폭 줄어든다.
# ---------------------------------------------------------------------------


@dataclass
class _PruneGroup:
    criterion: torch.Tensor                    # (n_filters, ...) — ranking 기준
    sparsity: float
    targets: List[Tuple[torch.Tensor, int, float]]  # (tensor, dim, fill)
    _mask: torch.Tensor | None = field(default=None, repr=False)

    def refresh(self) -> None:
        """topk 로 인덱스를 재계산하고 마스크를 갱신한다."""
        n = self.criterion.shape[0]
        n_prune = _calc_n_prune(n, self.sparsity)
        mask = torch.ones(n, dtype=torch.float32, device=self.criterion.device)
        if n_prune > 0:
            norms = torch.norm(self.criterion.detach().reshape(n, -1), dim=1)
            _, idx = torch.topk(norms, n_prune, largest=False)
            mask[idx] = 0.0
        self._mask = mask

    @torch.no_grad()
    def apply(self) -> None:
        """캐시된 마스크로 모든 대상 텐서를 벡터화하여 마스킹한다."""
        if self._mask is None:
            return
        mask = self._mask  # (n,)
        for tensor, dim, fill in self.targets:
            if tensor is None:
                continue
            # mask 를 tensor 차원에 맞게 브로드캐스트.
            shape = [1] * tensor.dim()
            shape[dim] = -1
            m = mask.view(shape)
            if fill == 0.0:
                tensor.data.mul_(m)
            else:
                # pruned 채널 → fill, alive 채널 → 현재 값 유지.
                # tensor = tensor * mask + fill * (1 - mask)
                tensor.data.mul_(m).add_(fill * (1.0 - m))


# ---------------------------------------------------------------------------
# 그룹 수집 헬퍼 — init 시 1회만 실행
# ---------------------------------------------------------------------------


def _targets_from_bn(bn: nn.Module | None, fill_mean: float = 0.0) -> list:
    """BN/LN 의 weight, bias, running_mean, running_var 를 타겟 리스트로 반환."""
    out = []
    if bn is None:
        return out
    if getattr(bn, "weight", None) is not None:
        out.append((bn.weight, 0, 0.0))
    if getattr(bn, "bias", None) is not None:
        out.append((bn.bias, 0, 0.0))
    if isinstance(bn, _BatchNorm):
        out.append((bn.running_mean, 0, 0.0))
        out.append((bn.running_var, 0, 1.0))  # div-by-zero 방지: pruned → 1.0
    return out


def _collect_mbconv_group(mb: MBConv, sparsity: float) -> _PruneGroup:
    targets = []
    inv = mb.inverted_conv
    targets.append((inv.conv.weight, 0, 0.0))
    if inv.conv.bias is not None:
        targets.append((inv.conv.bias, 0, 0.0))
    targets.extend(_targets_from_bn(inv.norm))

    dw = mb.depth_conv
    targets.append((dw.conv.weight, 0, 0.0))
    if dw.conv.bias is not None:
        targets.append((dw.conv.bias, 0, 0.0))
    targets.extend(_targets_from_bn(dw.norm))

    # point_conv: 입력 컬럼 (dim=1)
    targets.append((mb.point_conv.conv.weight, 1, 0.0))

    return _PruneGroup(criterion=inv.conv.weight, sparsity=sparsity, targets=targets)


def _collect_fusedmbconv_group(fmb: FusedMBConv, sparsity: float) -> _PruneGroup:
    targets = []
    sp = fmb.spatial_conv
    targets.append((sp.conv.weight, 0, 0.0))
    if sp.conv.bias is not None:
        targets.append((sp.conv.bias, 0, 0.0))
    targets.extend(_targets_from_bn(sp.norm))

    targets.append((fmb.point_conv.conv.weight, 1, 0.0))

    return _PruneGroup(criterion=sp.conv.weight, sparsity=sparsity, targets=targets)


def _collect_stem_groups(model: nn.Module, sparsity: float) -> list[_PruneGroup]:
    """Stem chain 전체를 단일 criterion(stem[0].conv.weight) 으로 묶어 반환."""
    stem = _get_stem_op_seq(model)
    if stem is None or not hasattr(stem, "op_list") or len(stem.op_list) == 0:
        return []
    first_cl = stem.op_list[0]
    if not isinstance(first_cl, ConvLayer):
        return []

    criterion = first_cl.conv.weight  # (C0, 3, k, k)
    targets = []

    # stem[0]: 출력 필터 (dim=0)
    targets.append((first_cl.conv.weight, 0, 0.0))
    if first_cl.conv.bias is not None:
        targets.append((first_cl.conv.bias, 0, 0.0))
    targets.extend(_targets_from_bn(first_cl.norm))

    # stem[1..]: inner DSConv / ResBlock
    for inner in _iter_stem_inner_blocks(stem):
        if isinstance(inner, DSConv):
            dw = inner.depth_conv
            targets.append((dw.conv.weight, 0, 0.0))
            if dw.conv.bias is not None:
                targets.append((dw.conv.bias, 0, 0.0))
            targets.extend(_targets_from_bn(dw.norm))

            pw = inner.point_conv
            targets.append((pw.conv.weight, 0, 0.0))  # 출력 필터
            targets.append((pw.conv.weight, 1, 0.0))  # 입력 컬럼 (양방향)
            if pw.conv.bias is not None:
                targets.append((pw.conv.bias, 0, 0.0))
            targets.extend(_targets_from_bn(pw.norm))
        else:  # ResBlock
            for cl in (inner.conv1, inner.conv2):
                targets.append((cl.conv.weight, 0, 0.0))
                targets.append((cl.conv.weight, 1, 0.0))
                if cl.conv.bias is not None:
                    targets.append((cl.conv.bias, 0, 0.0))
                targets.extend(_targets_from_bn(cl.norm))

    # 다음 stage 첫 conv 입력 컬럼 (dim=1)
    post = _get_post_stem_first_conv(model)
    if post is not None:
        targets.append((post.conv.weight, 1, 0.0))

    return [_PruneGroup(criterion=criterion, sparsity=sparsity, targets=targets)]


def _collect_head_groups(model: nn.Module, head_sparsity: float) -> list[_PruneGroup]:
    """G_HEAD0, G_HEAD1 을 각각 _PruneGroup 으로 수집한다."""
    if not _has_prunable_head(model):
        return []
    head = model.head  # type: ignore[attr-defined]
    groups = []

    # G_HEAD0: op_list[0] Conv 출력 필터 → op_list[2] Linear 입력 컬럼
    conv0_w = head.op_list[0].conv.weight  # (w0, in_ch, 1, 1)
    n_prune0 = _calc_n_prune(conv0_w.shape[0], head_sparsity)
    if n_prune0 > 0:
        t0 = []
        t0.append((conv0_w, 0, 0.0))
        if head.op_list[0].conv.bias is not None:
            t0.append((head.op_list[0].conv.bias, 0, 0.0))
        t0.extend(_targets_from_bn(head.op_list[0].norm))
        t0.append((head.op_list[2].linear.weight, 1, 0.0))  # Linear 입력 컬럼
        groups.append(_PruneGroup(criterion=conv0_w, sparsity=head_sparsity, targets=t0))

    # G_HEAD1: op_list[2] Linear 출력 행 → op_list[3] Linear 입력 컬럼
    lin2_w = head.op_list[2].linear.weight  # (w1, w0)
    effective_sp = min(head_sparsity, _HEAD1_MAX_SPARSITY)
    n_prune1 = _calc_n_prune(lin2_w.shape[0], effective_sp)
    if n_prune1 > 0:
        t1 = []
        t1.append((lin2_w, 0, 0.0))
        if head.op_list[2].linear.bias is not None:
            t1.append((head.op_list[2].linear.bias, 0, 0.0))
        # LayerNorm (running stats 없음, weight/bias 만)
        ln = head.op_list[2].norm
        if ln is not None:
            if getattr(ln, "weight", None) is not None:
                t1.append((ln.weight, 0, 0.0))
            if getattr(ln, "bias", None) is not None:
                t1.append((ln.bias, 0, 0.0))
        t1.append((head.op_list[3].linear.weight, 1, 0.0))  # Linear 입력 컬럼
        groups.append(_PruneGroup(criterion=lin2_w, sparsity=effective_sp, targets=t1))

    return groups


# ---------------------------------------------------------------------------
# log_sparsity / 이진탐색에 여전히 필요한 구버전 인터페이스 (변경 없음)
# ---------------------------------------------------------------------------


def _zero_bn_(bn: nn.Module | None, idx: torch.Tensor) -> None:
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


def _prune_mbconv(mb: MBConv, sparsity: float) -> None:
    weight = mb.inverted_conv.conv.weight
    mid = weight.shape[0]
    n_prune = _calc_n_prune(mid, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)
    with torch.no_grad():
        mb.inverted_conv.conv.weight.data[idx] = 0.0
        if mb.inverted_conv.conv.bias is not None:
            mb.inverted_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(mb.inverted_conv.norm, idx)
        mb.depth_conv.conv.weight.data[idx] = 0.0
        if mb.depth_conv.conv.bias is not None:
            mb.depth_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(mb.depth_conv.norm, idx)
        mb.point_conv.conv.weight.data[:, idx] = 0.0


def _prune_fusedmbconv(fmb: FusedMBConv, sparsity: float) -> None:
    weight = fmb.spatial_conv.conv.weight
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
        fmb.point_conv.conv.weight.data[:, idx] = 0.0


def _estimate_removed_mbconv(mb: MBConv, sparsity: float) -> int:
    weight = mb.inverted_conv.conv.weight
    mid, in_ch = weight.shape[0], weight.shape[1]
    out_ch = mb.point_conv.conv.weight.shape[0]
    k = mb.depth_conv.conv.weight.shape[-1]
    n_prune = _calc_n_prune(mid, sparsity)
    if n_prune <= 0:
        return 0
    removed = 0
    removed += n_prune * in_ch
    if mb.inverted_conv.conv.bias is not None:
        removed += n_prune
    removed += n_prune * (1 if isinstance(mb.inverted_conv.norm, _BatchNorm) else 0) * 2
    removed += n_prune * k * k
    if mb.depth_conv.conv.bias is not None:
        removed += n_prune
    removed += n_prune * (1 if isinstance(mb.depth_conv.norm, _BatchNorm) else 0) * 2
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
# Input Stem 유틸 (이진탐색 + log_sparsity + collect 공용)
# ---------------------------------------------------------------------------


def _zero_conv_out_filters_(cl: ConvLayer, idx: torch.Tensor) -> None:
    with torch.no_grad():
        cl.conv.weight.data[idx] = 0.0
        if cl.conv.bias is not None:
            cl.conv.bias.data[idx] = 0.0
    _zero_bn_(cl.norm, idx)


def _zero_conv_in_cols_(cl: ConvLayer, idx: torch.Tensor) -> None:
    with torch.no_grad():
        cl.conv.weight.data[:, idx] = 0.0


def _get_stem_op_seq(model: nn.Module) -> OpSequential | None:
    bb = getattr(model, "backbone", None)
    if bb is None:
        return None
    if hasattr(bb, "input_stem"):
        return bb.input_stem
    if hasattr(bb, "stages") and len(bb.stages) > 0:
        return bb.stages[0]
    return None


def _get_post_stem_first_conv(model: nn.Module) -> ConvLayer | None:
    bb = getattr(model, "backbone", None)
    if bb is None or not hasattr(bb, "stages") or len(bb.stages) == 0:
        return None
    stage1_idx = 0 if hasattr(bb, "input_stem") else 1
    if stage1_idx >= len(bb.stages):
        return None
    stage1 = bb.stages[stage1_idx]
    if not hasattr(stage1, "op_list") or len(stage1.op_list) == 0:
        return None
    first = stage1.op_list[0]
    main = getattr(first, "main", first)
    if isinstance(main, MBConv):
        return main.inverted_conv
    if isinstance(main, FusedMBConv):
        return main.spatial_conv
    return None


def _iter_stem_inner_blocks(stem: OpSequential):
    for op in stem.op_list[1:]:
        main = getattr(op, "main", op)
        if isinstance(main, (DSConv, ResBlock)):
            yield main


def _prune_input_stem(model: nn.Module, sparsity: float) -> None:
    stem = _get_stem_op_seq(model)
    if stem is None or not hasattr(stem, "op_list") or len(stem.op_list) == 0:
        return
    first_cl = stem.op_list[0]
    if not isinstance(first_cl, ConvLayer):
        return
    weight = first_cl.conv.weight
    n_total = weight.shape[0]
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)
    _zero_conv_out_filters_(first_cl, idx)
    for inner in _iter_stem_inner_blocks(stem):
        if isinstance(inner, DSConv):
            _zero_conv_out_filters_(inner.depth_conv, idx)
            _zero_conv_out_filters_(inner.point_conv, idx)
            _zero_conv_in_cols_(inner.point_conv, idx)
        else:
            _zero_conv_out_filters_(inner.conv1, idx)
            _zero_conv_in_cols_(inner.conv1, idx)
            _zero_conv_out_filters_(inner.conv2, idx)
            _zero_conv_in_cols_(inner.conv2, idx)
    post = _get_post_stem_first_conv(model)
    if post is not None:
        _zero_conv_in_cols_(post, idx)


def _estimate_removed_input_stem(model: nn.Module, sparsity: float) -> int:
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
    removed += n_prune * in_ch_image * k0 * k0
    if first_cl.conv.bias is not None:
        removed += n_prune
    if isinstance(first_cl.norm, _BatchNorm):
        removed += n_prune * 2
    for inner in _iter_stem_inner_blocks(stem):
        if isinstance(inner, DSConv):
            kd = inner.depth_conv.conv.weight.shape[-1]
            removed += n_prune * kd * kd
            if inner.depth_conv.conv.bias is not None:
                removed += n_prune
            if isinstance(inner.depth_conv.norm, _BatchNorm):
                removed += n_prune * 2
            removed += n_total * n_total - n_surv * n_surv
            if inner.point_conv.conv.bias is not None:
                removed += n_prune
            if isinstance(inner.point_conv.norm, _BatchNorm):
                removed += n_prune * 2
        else:
            kr1 = inner.conv1.conv.weight.shape[-1]
            kr2 = inner.conv2.conv.weight.shape[-1]
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
    post = _get_post_stem_first_conv(model)
    if post is not None:
        post_w = post.conv.weight
        out_first = post_w.shape[0]
        kp = post_w.shape[-1]
        removed += n_prune * out_first * kp * kp
    return removed


# ---------------------------------------------------------------------------
# ClsHead (G_HEAD0 + G_HEAD1)
# ---------------------------------------------------------------------------

_HEAD1_MAX_SPARSITY = 0.40


def _prune_head0(head: nn.Module, sparsity: float) -> None:
    weight = head.op_list[0].conv.weight
    n_total = weight.shape[0]
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)
    with torch.no_grad():
        head.op_list[0].conv.weight.data[idx] = 0.0
        if head.op_list[0].conv.bias is not None:
            head.op_list[0].conv.bias.data[idx] = 0.0
        _zero_bn_(head.op_list[0].norm, idx)
        head.op_list[2].linear.weight.data[:, idx] = 0.0


def _prune_head1(head: nn.Module, sparsity: float) -> None:
    effective_sp = min(sparsity, _HEAD1_MAX_SPARSITY)
    weight = head.op_list[2].linear.weight
    n_total = weight.shape[0]
    n_prune = _calc_n_prune(n_total, effective_sp)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)
    with torch.no_grad():
        head.op_list[2].linear.weight.data[idx] = 0.0
        if head.op_list[2].linear.bias is not None:
            head.op_list[2].linear.bias.data[idx] = 0.0
        _zero_bn_(head.op_list[2].norm, idx)
        head.op_list[3].linear.weight.data[:, idx] = 0.0


def _estimate_removed_head0(head: nn.Module, sparsity: float) -> int:
    weight = head.op_list[0].conv.weight
    n_total, in_ch = weight.shape[0], weight.shape[1]
    out_features = head.op_list[2].linear.weight.shape[0]
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return 0
    removed = n_prune * in_ch
    if head.op_list[0].conv.bias is not None:
        removed += n_prune
    if getattr(head.op_list[0].norm, "weight", None) is not None:
        removed += n_prune * 2
    removed += n_prune * out_features
    return removed


def _estimate_removed_head1(head: nn.Module, sparsity: float) -> int:
    effective_sp = min(sparsity, _HEAD1_MAX_SPARSITY)
    weight = head.op_list[2].linear.weight
    n_total, in_features = weight.shape[0], weight.shape[1]
    out_classes = head.op_list[3].linear.weight.shape[0]
    n_prune = _calc_n_prune(n_total, effective_sp)
    if n_prune <= 0:
        return 0
    removed = n_prune * in_features
    if head.op_list[2].linear.bias is not None:
        removed += n_prune
    if getattr(head.op_list[2].norm, "weight", None) is not None:
        removed += n_prune * 2
    removed += n_prune * out_classes
    return removed


def _has_prunable_head(model: nn.Module) -> bool:
    head = getattr(model, "head", None)
    return (
        head is not None
        and hasattr(head, "op_list")
        and len(head.op_list) >= 4
        and hasattr(head.op_list[0], "conv")
        and hasattr(head.op_list[2], "linear")
    )


# ---------------------------------------------------------------------------
# 이진탐색 (변경 없음)
# ---------------------------------------------------------------------------


def _iter_prunable_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MBConv):
            yield ("mbconv", module)
        elif isinstance(module, FusedMBConv):
            yield ("fmbconv", module)


def _count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_total_removed(model: nn.Module, sparsity: float, head_sparsity_scale: float = 0.0) -> int:
    total = 0
    for kind, mod in _iter_prunable_modules(model):
        if kind == "mbconv":
            total += _estimate_removed_mbconv(mod, sparsity)
        elif kind == "fmbconv":
            total += _estimate_removed_fusedmbconv(mod, sparsity)
    total += _estimate_removed_input_stem(model, sparsity)
    if head_sparsity_scale > 0 and _has_prunable_head(model):
        head_sp = sparsity * head_sparsity_scale
        head = model.head  # type: ignore[attr-defined]
        total += _estimate_removed_head0(head, head_sp)
        total += _estimate_removed_head1(head, head_sp)
    return total


def _find_sparsity_by_bisection(
    model: nn.Module,
    target_compression: float,
    max_sparsity: float = 0.95,
    iters: int = 64,
    head_sparsity_scale: float = 0.0,
) -> float:
    if target_compression <= 0:
        return 0.0
    total = _count_total_params(model)
    target_remove = target_compression * total
    lo, hi = 0.0, max_sparsity
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _estimate_total_removed(model, mid, head_sparsity_scale) < target_remove:
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

    성능 최적화 (index_refresh_steps):
        - 매 step topk 재계산 대신, index_refresh_steps 마다 한 번만 계산.
        - 기본값 100: 100 step 간격으로 pruned 채널 순위를 재평가.
        - 초기 수렴 전이라면 50, 완전 수렴 후라면 200~500 으로 조절 가능.
        - 0 으로 설정하면 매 step 재계산 (구버전 동작과 동일).

    head_sparsity_scale:
        Head sparsity = backbone sparsity × scale.
        0.0 이면 head pruning 비활성.
        0.5 (기본값) 이면 backbone 절반 sparsity 로 head pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        target_compression: float,
        max_sparsity: float = 0.95,
        sparsity: float | None = None,
        head_sparsity_scale: float = 0.5,
        index_refresh_steps: int = 100,
    ) -> None:
        self.target_compression = float(target_compression)
        self.max_sparsity = float(max_sparsity)
        self.head_sparsity_scale = float(head_sparsity_scale)
        self.index_refresh_steps = int(index_refresh_steps)
        self._step = 0

        if sparsity is not None:
            self.sparsity = float(sparsity)
        else:
            self.sparsity = _find_sparsity_by_bisection(
                model, self.target_compression, self.max_sparsity,
                head_sparsity_scale=self.head_sparsity_scale,
            )

        # 통계 로그
        n_groups = sum(1 for _ in _iter_prunable_modules(model))
        total = _count_total_params(model)
        est_removed = _estimate_total_removed(model, self.sparsity, self.head_sparsity_scale)
        rate = 100.0 * est_removed / max(total, 1)
        head_info = (
            f"head_sparsity={self.sparsity * self.head_sparsity_scale:.4f} "
            f"(scale={self.head_sparsity_scale})"
            if self.head_sparsity_scale > 0 and _has_prunable_head(model)
            else "head=disabled"
        )
        print(
            f"[EfficientViTPruner] target={self.target_compression*100:.1f}% "
            f"backbone_sparsity={self.sparsity:.4f} "
            f"prunable_groups={n_groups} estimated_compression={rate:.2f}% "
            f"{head_info} index_refresh_steps={self.index_refresh_steps}"
        )

        # init 시 한 번만 모든 prunable 텐서 레퍼런스 수집
        self._groups: list[_PruneGroup] = self._collect_all_groups(model)

        # 첫 마스크 계산
        for g in self._groups:
            g.refresh()

    def _collect_all_groups(self, model: nn.Module) -> list[_PruneGroup]:
        groups: list[_PruneGroup] = []
        for kind, mod in _iter_prunable_modules(model):
            if kind == "mbconv":
                groups.append(_collect_mbconv_group(mod, self.sparsity))
            elif kind == "fmbconv":
                groups.append(_collect_fusedmbconv_group(mod, self.sparsity))
        groups.extend(_collect_stem_groups(model, self.sparsity))
        if self.head_sparsity_scale > 0:
            head_sp = self.sparsity * self.head_sparsity_scale
            groups.extend(_collect_head_groups(model, head_sp))
        return groups

    def apply(self, model: nn.Module) -> None:  # noqa: ARG002  (model 인자는 하위 호환성 유지)
        """캐시된 마스크로 모든 prunable 그룹을 마스킹한다.

        index_refresh_steps 마다 topk 인덱스를 재계산한다.
        model 인자는 하위 호환성을 위해 유지하지만 사용하지 않는다
        (텐서 레퍼런스는 _groups 에 이미 저장되어 있음).
        """
        if self.sparsity <= 0:
            return
        need_refresh = (self.index_refresh_steps <= 0) or (self._step % self.index_refresh_steps == 0)
        if need_refresh:
            for g in self._groups:
                g.refresh()
        for g in self._groups:
            g.apply()
        self._step += 1

    @torch.no_grad()
    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """실제 마스킹된 zero 비율 (검증용). MBConv/FusedMBConv mid + Stem + Head 출력 채널."""
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
        stem = _get_stem_op_seq(model)
        if stem is not None and len(stem.op_list) > 0 and isinstance(stem.op_list[0], ConvLayer):
            w = stem.op_list[0].conv.weight
            n_filt = w.shape[0]
            norms = torch.norm(w.detach().reshape(n_filt, -1), dim=1)
            n_total += n_filt
            n_zero += int((norms == 0).sum().item())
        if self.head_sparsity_scale > 0 and _has_prunable_head(model):
            head = model.head  # type: ignore[attr-defined]
            w0 = head.op_list[0].conv.weight
            n0 = w0.shape[0]
            norms0 = torch.norm(w0.detach().reshape(n0, -1), dim=1)
            n_total += n0
            n_zero += int((norms0 == 0).sum().item())
            w1 = head.op_list[2].linear.weight
            n1 = w1.shape[0]
            norms1 = torch.norm(w1.detach(), dim=1)
            n_total += n1
            n_zero += int((norms1 == 0).sum().item())
        return {
            "prunable_filters": n_total,
            "zero_filters": n_zero,
            "actual_sparsity": n_zero / max(n_total, 1),
            "target_sparsity": self.sparsity,
        }
