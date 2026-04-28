"""Soft Pruning 으로 0 마스킹된 모델을 실제로 작은 Dense 모델로 변환.

핵심 아이디어
  - Soft Pruning 단계에서 prune 된 채널은 Conv weight 의 L2 norm 이 정확히 0.
  - "norm > 0" 인 인덱스만 추려서 새로운 (작은) Conv2d / BatchNorm 으로 교체.
  - 동일한 인덱스 집합을 다음 레이어의 입력 차원에도 일관되게 적용 (coupled pruning).

대상 그룹은 efficientvit_pruning.py 와 1:1 로 짝짓는다:
  - MBConv (inverted_conv → depth_conv → point_conv)
  - FusedMBConv (spatial_conv → point_conv)

교체 방식: ConvLayer.conv / ConvLayer.norm 을 동일 in/out 시그니처의 새 모듈로
in-place 치환하고, 부모 MBConv/FusedMBConv 의 forward 시그니처는 그대로 유지.
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Optional

import torch
import torch.nn as nn

from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.models.efficientvit.cls import EfficientViTCls
from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    FusedMBConv,
    MBConv,
    OpSequential,
    ResBlock,
)

__all__ = ["reduce_efficientvit_cls_model", "main"]


# ---------------------------------------------------------------------------
# Survived index 추출
# ---------------------------------------------------------------------------


@torch.no_grad()
def _survived_idx(weight: torch.Tensor) -> torch.Tensor:
    """첫 차원 기준 L2 norm 이 0 이 아닌 인덱스 (오름차순 보장)."""
    n = weight.shape[0]
    norms = torch.norm(weight.reshape(n, -1), dim=1)
    return torch.nonzero(norms != 0, as_tuple=False).flatten()


# ---------------------------------------------------------------------------
# 새 conv / BN 인스턴스 생성 헬퍼
# ---------------------------------------------------------------------------


def _new_conv_like(
    old: nn.Conv2d,
    in_channels: int,
    out_channels: int,
    groups: int,
) -> nn.Conv2d:
    """old 와 동일한 kernel/stride/padding/... 설정의 새 Conv2d. weight 복사는 호출자."""
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    ).to(weight_device(old))
    new = new.to(old.weight.dtype)
    return new


def weight_device(m: nn.Module) -> torch.device:
    p = next(m.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _new_bn_like(old: nn.BatchNorm2d, num_features: int) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(
        num_features=num_features,
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
    ).to(weight_device(old))
    if old.weight is not None:
        new = new.to(old.weight.dtype)
    return new


# ---------------------------------------------------------------------------
# Block 단위 reduce
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reduce_mbconv(mb: MBConv) -> None:
    """MBConv 의 mid_channels 축을 survived 인덱스만큼 줄여 in-place 교체."""
    inv = mb.inverted_conv.conv  # (mid, in, 1, 1)
    survived = _survived_idx(inv.weight)
    n_new = survived.numel()
    if n_new == inv.weight.shape[0]:
        return  # 줄일 게 없음.
    in_ch = inv.weight.shape[1]
    out_ch = mb.point_conv.conv.weight.shape[0]

    # 1) inverted_conv 교체.
    new_inv = _new_conv_like(inv, in_channels=in_ch, out_channels=n_new, groups=inv.groups)
    new_inv.weight.data.copy_(inv.weight.data[survived])
    if inv.bias is not None:
        new_inv.bias.data.copy_(inv.bias.data[survived])
    mb.inverted_conv.conv = new_inv
    if isinstance(mb.inverted_conv.norm, nn.BatchNorm2d):
        old_bn = mb.inverted_conv.norm
        new_bn = _new_bn_like(old_bn, n_new)
        if old_bn.weight is not None:
            new_bn.weight.data.copy_(old_bn.weight.data[survived])
            new_bn.bias.data.copy_(old_bn.bias.data[survived])
        if old_bn.track_running_stats:
            new_bn.running_mean.data.copy_(old_bn.running_mean.data[survived])
            new_bn.running_var.data.copy_(old_bn.running_var.data[survived])
        mb.inverted_conv.norm = new_bn

    # 2) depth_conv: groups=mid 이므로 in==out==n_new.
    dw = mb.depth_conv.conv  # (mid, 1, k, k)
    new_dw = _new_conv_like(dw, in_channels=n_new, out_channels=n_new, groups=n_new)
    new_dw.weight.data.copy_(dw.weight.data[survived])
    if dw.bias is not None:
        new_dw.bias.data.copy_(dw.bias.data[survived])
    mb.depth_conv.conv = new_dw
    if isinstance(mb.depth_conv.norm, nn.BatchNorm2d):
        old_bn = mb.depth_conv.norm
        new_bn = _new_bn_like(old_bn, n_new)
        if old_bn.weight is not None:
            new_bn.weight.data.copy_(old_bn.weight.data[survived])
            new_bn.bias.data.copy_(old_bn.bias.data[survived])
        if old_bn.track_running_stats:
            new_bn.running_mean.data.copy_(old_bn.running_mean.data[survived])
            new_bn.running_var.data.copy_(old_bn.running_var.data[survived])
        mb.depth_conv.norm = new_bn

    # 3) point_conv: 입력 컬럼 축소 (out_ch 는 그대로).
    pw = mb.point_conv.conv  # (out, mid, 1, 1)
    new_pw = _new_conv_like(pw, in_channels=n_new, out_channels=out_ch, groups=pw.groups)
    new_pw.weight.data.copy_(pw.weight.data[:, survived])
    if pw.bias is not None:
        new_pw.bias.data.copy_(pw.bias.data)
    mb.point_conv.conv = new_pw
    # point_conv 의 norm 은 출력 채널 기준이라 손대지 않음.


@torch.no_grad()
def _reduce_convlayer_out(cl: ConvLayer, survived: torch.Tensor, in_channels: int, groups: int) -> None:
    """ConvLayer.conv / .norm 의 출력 필터를 survived 만 남기도록 in-place 교체."""
    old_conv = cl.conv
    new_conv = _new_conv_like(
        old_conv, in_channels=in_channels, out_channels=survived.numel(), groups=groups
    )
    new_conv.weight.data.copy_(old_conv.weight.data[survived])
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data[survived])
    cl.conv = new_conv
    if isinstance(cl.norm, nn.BatchNorm2d):
        old_bn = cl.norm
        new_bn = _new_bn_like(old_bn, survived.numel())
        if old_bn.weight is not None:
            new_bn.weight.data.copy_(old_bn.weight.data[survived])
            new_bn.bias.data.copy_(old_bn.bias.data[survived])
        if old_bn.track_running_stats:
            new_bn.running_mean.data.copy_(old_bn.running_mean.data[survived])
            new_bn.running_var.data.copy_(old_bn.running_var.data[survived])
        cl.norm = new_bn


@torch.no_grad()
def _reduce_convlayer_inout(
    cl: ConvLayer, survived_in: torch.Tensor, survived_out: torch.Tensor, groups: int = 1
) -> None:
    """ConvLayer 의 입력 컬럼 + 출력 필터 + BN 동시 축소 (양방향)."""
    old_conv = cl.conv
    new_conv = _new_conv_like(
        old_conv, in_channels=survived_in.numel(), out_channels=survived_out.numel(), groups=groups
    )
    w = old_conv.weight.data[survived_out]
    if groups == 1:
        w = w[:, survived_in]
    new_conv.weight.data.copy_(w)
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data[survived_out])
    cl.conv = new_conv
    if isinstance(cl.norm, nn.BatchNorm2d):
        old_bn = cl.norm
        new_bn = _new_bn_like(old_bn, survived_out.numel())
        if old_bn.weight is not None:
            new_bn.weight.data.copy_(old_bn.weight.data[survived_out])
            new_bn.bias.data.copy_(old_bn.bias.data[survived_out])
        if old_bn.track_running_stats:
            new_bn.running_mean.data.copy_(old_bn.running_mean.data[survived_out])
            new_bn.running_var.data.copy_(old_bn.running_var.data[survived_out])
        cl.norm = new_bn


@torch.no_grad()
def _reduce_input_stem(model: nn.Module) -> None:
    """Stem chain (input_stem 또는 stages[0]) 의 모든 레이어를 survived idx 로 축소.

    Soft Pruning 단계에서 동일 idx 가 모든 stem 레이어에 적용되었으므로,
    stem 첫 ConvLayer 의 출력 필터 norm 에서 추출한 survived idx 를
    그대로 chain 전체와 다음 stage 첫 conv 입력 컬럼에 적용한다.
    """
    bb = getattr(model, "backbone", None)
    if bb is None:
        return
    stem: OpSequential | None = None
    stage1_idx: int | None = None
    if hasattr(bb, "input_stem"):
        stem = bb.input_stem
        stage1_idx = 0  # B-series: stages[0] 가 stage1.
    elif hasattr(bb, "stages") and len(bb.stages) > 1:
        stem = bb.stages[0]
        stage1_idx = 1  # L-series: stages[1] 가 stage1.
    if stem is None or not hasattr(stem, "op_list") or len(stem.op_list) == 0:
        return
    first_cl = stem.op_list[0]
    if not isinstance(first_cl, ConvLayer):
        return

    survived = _survived_idx(first_cl.conv.weight)
    n_new = survived.numel()
    if n_new == first_cl.conv.weight.shape[0]:
        return  # 살릴 게 줄지 않았다 → 축소 불필요.

    in_image_ch = first_cl.conv.weight.shape[1]  # = 3
    _reduce_convlayer_out(first_cl, survived, in_channels=in_image_ch, groups=1)

    # Inner blocks: DSConv 또는 ResBlock.
    for op in stem.op_list[1:]:
        main = getattr(op, "main", op)
        if isinstance(main, DSConv):
            # depth_conv: groups=n_new, in=out=n_new.
            _reduce_convlayer_out(main.depth_conv, survived, in_channels=n_new, groups=n_new)
            # point_conv: in=out=n_new (양방향).
            _reduce_convlayer_inout(main.point_conv, survived, survived, groups=1)
        elif isinstance(main, ResBlock):
            _reduce_convlayer_inout(main.conv1, survived, survived, groups=1)
            _reduce_convlayer_inout(main.conv2, survived, survived, groups=1)

    # Stage1 의 첫 down-sampling Conv 입력 컬럼만 축소 (출력은 외부 채널, 그대로).
    if stage1_idx is None or stage1_idx >= len(bb.stages):
        return
    stage1 = bb.stages[stage1_idx]
    if not hasattr(stage1, "op_list") or len(stage1.op_list) == 0:
        return
    first_blk = stage1.op_list[0]
    main = getattr(first_blk, "main", first_blk)
    post: ConvLayer | None = None
    if isinstance(main, MBConv):
        post = main.inverted_conv
    elif isinstance(main, FusedMBConv):
        post = main.spatial_conv
    if post is None:
        return
    old_conv = post.conv
    new_conv = _new_conv_like(
        old_conv,
        in_channels=n_new,
        out_channels=old_conv.out_channels,
        groups=old_conv.groups,
    )
    new_conv.weight.data.copy_(old_conv.weight.data[:, survived])
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    post.conv = new_conv
    # post.norm 은 출력 채널 기준이므로 변경하지 않음.


@torch.no_grad()
def _reduce_fusedmbconv(fmb: FusedMBConv) -> None:
    sp = fmb.spatial_conv.conv  # (mid, in/groups, k, k)
    survived = _survived_idx(sp.weight)
    n_new = survived.numel()
    if n_new == sp.weight.shape[0]:
        return
    out_ch = fmb.point_conv.conv.weight.shape[0]
    in_per_group = sp.weight.shape[1]
    in_ch = in_per_group * sp.groups

    new_sp = _new_conv_like(sp, in_channels=in_ch, out_channels=n_new, groups=sp.groups)
    new_sp.weight.data.copy_(sp.weight.data[survived])
    if sp.bias is not None:
        new_sp.bias.data.copy_(sp.bias.data[survived])
    fmb.spatial_conv.conv = new_sp
    if isinstance(fmb.spatial_conv.norm, nn.BatchNorm2d):
        old_bn = fmb.spatial_conv.norm
        new_bn = _new_bn_like(old_bn, n_new)
        if old_bn.weight is not None:
            new_bn.weight.data.copy_(old_bn.weight.data[survived])
            new_bn.bias.data.copy_(old_bn.bias.data[survived])
        if old_bn.track_running_stats:
            new_bn.running_mean.data.copy_(old_bn.running_mean.data[survived])
            new_bn.running_var.data.copy_(old_bn.running_var.data[survived])
        fmb.spatial_conv.norm = new_bn

    pw = fmb.point_conv.conv
    new_pw = _new_conv_like(pw, in_channels=n_new, out_channels=out_ch, groups=pw.groups)
    new_pw.weight.data.copy_(pw.weight.data[:, survived])
    if pw.bias is not None:
        new_pw.bias.data.copy_(pw.bias.data)
    fmb.point_conv.conv = new_pw


# ---------------------------------------------------------------------------
# 모델 전체 reduce
# ---------------------------------------------------------------------------


@torch.no_grad()
def reduce_efficientvit_cls_model(model: nn.Module) -> nn.Module:
    """모델 전체를 in-place 로 dense reduce 한다. 반환은 동일 객체.

    순서:
      1) MBConv / FusedMBConv mid_channels 축소.
      2) Input stem chain 축소 + stage1 첫 conv 입력 컬럼 축소.
         (stem 출력 채널이 stage1 첫 conv 입력 채널이라 stem 후 처리 가능.)
    """
    for module in model.modules():
        if isinstance(module, MBConv):
            _reduce_mbconv(module)
        elif isinstance(module, FusedMBConv):
            _reduce_fusedmbconv(module)
    _reduce_input_stem(model)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_state_dict_from_ckpt(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "ema"):
            if key in ckpt and isinstance(ckpt[key], dict):
                if key == "ema" and "shadows" in ckpt[key]:
                    return ckpt[key]["shadows"]
                return ckpt[key]
    return ckpt


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Reduce a soft-pruned EfficientViT cls model.")
    parser.add_argument("--model", required=True, help="model name in cls_model_zoo, e.g. efficientvit-b1")
    parser.add_argument("--checkpoint", required=True, help="path to soft-pruned checkpoint")
    parser.add_argument("--output", required=True, help="output path for the reduced model")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--n-classes", type=int, default=1000)
    parser.add_argument("--save-full-model", action="store_true",
                        help="save the entire model object instead of state_dict")
    args = parser.parse_args(argv)

    print(f"=> creating model {args.model} (pretrained=False)")
    model: EfficientViTCls = create_efficientvit_cls_model(args.model, pretrained=False)
    state_dict = _load_state_dict_from_ckpt(args.checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing 5: {unexpected[:5]})")

    n_before = sum(p.numel() for p in model.parameters())
    print(f"=> param count BEFORE reduce: {n_before:,}")

    reduce_efficientvit_cls_model(model)

    n_after = sum(p.numel() for p in model.parameters())
    rate = 100.0 * (n_before - n_after) / max(n_before, 1)
    print(f"=> param count AFTER  reduce: {n_after:,}  (compression {rate:.2f}%)")

    # Forward sanity check on CPU.
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, args.input_size, args.input_size)
        y = model(x)
    assert y.shape[-1] == args.n_classes or y.shape[0] == 1, f"unexpected output shape {tuple(y.shape)}"
    print(f"=> forward OK, output shape: {tuple(y.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.save_full_model:
        torch.save(model, args.output)
    else:
        torch.save(
            {"state_dict": model.state_dict(), "compression_rate": rate, "model_name": args.model},
            args.output,
        )
    print(f"=> saved to {args.output}")


if __name__ == "__main__":
    main()
