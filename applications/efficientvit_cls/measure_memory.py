"""EfficientViT classification 모델의 컴포넌트별 파라미터 메모리 측정.

예제 (`EfficientViT- Example/CLAUDE.md` §11) 의
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
패턴을 우리 B/L 시리즈 구조에 맞게 재구성.

그룹 분류:
    G_STEM    : input_stem (B) / stages[0] (L) chain 의 모든 Conv/BN
    G_MBCONV  : 모든 MBConv (inverted_conv + depth_conv + point_conv + BN)
    G_FUSEDMB : 모든 FusedMBConv (spatial_conv + point_conv + BN)
    G_LITEMLA : 모든 LiteMLA (qkv + aggreg + proj)
    G_HEAD    : ClsHead (Conv + 두 Linear)
    G_OTHER   : 위 어디에도 속하지 않는 잔여 파라미터

사용 예:

  # 단일 모델 분해
  python applications/efficientvit_cls/measure_memory.py \\
      --model efficientvit-b1

  # 사전학습 weight 로드 후 분해
  python applications/efficientvit_cls/measure_memory.py \\
      --model efficientvit-b1 \\
      --checkpoint /path/to/efficientvit_b1_r224.pt

  # 원본 vs reduced (Soft Pruning + Reducing 후 결과) 비교
  python applications/efficientvit_cls/measure_memory.py \\
      --model efficientvit-b1 \\
      --reduced /path/to/reduced_b1_30pct.pt
"""

import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.clscore.pruning.efficientvit_reducing import (
    _load_state_dict_from_ckpt,
    _strip_module_prefix,
    reduce_efficientvit_cls_model,
)
from efficientvit.models.efficientvit.cls import ClsHead
from efficientvit.models.nn import (
    FusedMBConv,
    LiteMLA,
    MBConv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_param_stats(module: nn.Module) -> tuple[int, int]:
    """(numel, bytes) for all parameters of a module (no double-counting via Set)."""
    n_total = 0
    n_bytes = 0
    seen: set[int] = set()
    for p in module.parameters(recurse=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        n_total += p.numel()
        n_bytes += p.numel() * p.element_size()
    # Buffers (BN running_mean/var 등) 도 포함해야 실제 ckpt 크기와 일치.
    for b in module.buffers(recurse=True):
        if id(b) in seen:
            continue
        seen.add(id(b))
        n_total += b.numel()
        n_bytes += b.numel() * b.element_size()
    return n_total, n_bytes


def _stem_module(model: nn.Module) -> Optional[nn.Module]:
    bb = getattr(model, "backbone", None)
    if bb is None:
        return None
    if hasattr(bb, "input_stem"):
        return bb.input_stem
    if hasattr(bb, "stages") and len(bb.stages) > 0:
        return bb.stages[0]
    return None


def _classify_modules(model: nn.Module) -> dict[str, list[nn.Module]]:
    """그룹 → 모듈 리스트. 우선순위에 따라 한 모듈은 가장 깊은 그룹 하나에만 귀속."""
    stem = _stem_module(model)
    head = getattr(model, "head", None)

    # 우선순위 큰 순서로 등록 (stem / head 가 다른 그룹과 중첩되면 안 됨).
    groups: dict[str, list[nn.Module]] = {
        "G_STEM": [stem] if stem is not None else [],
        "G_HEAD": [head] if isinstance(head, ClsHead) else [],
        "G_LITEMLA": [],
        "G_MBCONV": [],
        "G_FUSEDMB": [],
    }

    # 어떤 모듈이 stem/head 의 자식인지 빠르게 확인하기 위한 id-set.
    excluded_ids: set[int] = set()
    for top_mod in [stem, head]:
        if top_mod is None:
            continue
        for m in top_mod.modules():
            excluded_ids.add(id(m))

    for module in model.modules():
        if id(module) in excluded_ids:
            continue
        if isinstance(module, LiteMLA):
            groups["G_LITEMLA"].append(module)
        elif isinstance(module, MBConv):
            groups["G_MBCONV"].append(module)
        elif isinstance(module, FusedMBConv):
            groups["G_FUSEDMB"].append(module)
    return groups


def _total_param_bytes(model: nn.Module) -> tuple[int, int]:
    """전체 model 의 (numel, bytes). buffer 포함."""
    return _module_param_stats(model)


def _format_breakdown(model: nn.Module, label: str) -> dict:
    """그룹별 (count, numel, bytes, % of total) 를 반환 + 콘솔 출력."""
    groups = _classify_modules(model)
    total_n, total_bytes = _total_param_bytes(model)

    # 그룹별 누적.
    rows = []
    accounted_n = 0
    accounted_bytes = 0
    for g_name, mods in groups.items():
        g_n, g_b = 0, 0
        for m in mods:
            n, b = _module_param_stats(m)
            g_n += n
            g_b += b
        rows.append(
            {
                "group": g_name,
                "modules": len(mods),
                "numel": g_n,
                "bytes": g_b,
                "MB": g_b / 1e6,
                "pct_of_total": 100.0 * g_n / max(total_n, 1),
            }
        )
        accounted_n += g_n
        accounted_bytes += g_b
    other_n = total_n - accounted_n
    other_bytes = total_bytes - accounted_bytes
    rows.append(
        {
            "group": "G_OTHER",
            "modules": -1,
            "numel": other_n,
            "bytes": other_bytes,
            "MB": other_bytes / 1e6,
            "pct_of_total": 100.0 * other_n / max(total_n, 1),
        }
    )

    # 출력.
    print(f"\n=== [{label}] component-wise parameter memory ===")
    print(f"{'group':<11}{'#mod':>6}{'numel':>14}{'MB':>12}{'%':>9}")
    print("-" * 52)
    for r in rows:
        mod_str = "-" if r["modules"] < 0 else str(r["modules"])
        print(
            f"{r['group']:<11}{mod_str:>6}{r['numel']:>14,}"
            f"{r['MB']:>12.3f}{r['pct_of_total']:>8.2f}%"
        )
    print("-" * 52)
    print(f"{'TOTAL':<11}{'':>6}{total_n:>14,}{total_bytes/1e6:>12.3f}{100.00:>8.2f}%")

    return {
        "label": label,
        "total_numel": total_n,
        "total_bytes": total_bytes,
        "total_MB": total_bytes / 1e6,
        "groups": rows,
    }


def _format_compare(orig: dict, reduced: dict) -> None:
    """원본 vs reduced 비교 표."""
    # group → row 매핑.
    a = {r["group"]: r for r in orig["groups"]}
    b = {r["group"]: r for r in reduced["groups"]}

    print("\n=== compression breakdown (Original → Reduced) ===")
    print(f"{'group':<11}{'orig MB':>12}{'red MB':>12}{'ΔMB':>12}{'compress':>11}")
    print("-" * 58)
    for name in ["G_STEM", "G_MBCONV", "G_FUSEDMB", "G_LITEMLA", "G_HEAD", "G_OTHER"]:
        ao = a.get(name, {"MB": 0.0, "numel": 0})
        bo = b.get(name, {"MB": 0.0, "numel": 0})
        d_mb = ao["MB"] - bo["MB"]
        rate = (
            100.0 * (ao["numel"] - bo["numel"]) / max(ao["numel"], 1)
            if ao["numel"] > 0
            else 0.0
        )
        print(
            f"{name:<11}{ao['MB']:>12.3f}{bo['MB']:>12.3f}{d_mb:>12.3f}{rate:>10.2f}%"
        )
    print("-" * 58)
    d_mb = orig["total_MB"] - reduced["total_MB"]
    rate = 100.0 * (orig["total_numel"] - reduced["total_numel"]) / max(orig["total_numel"], 1)
    print(
        f"{'TOTAL':<11}{orig['total_MB']:>12.3f}{reduced['total_MB']:>12.3f}"
        f"{d_mb:>12.3f}{rate:>10.2f}%"
    )


# ---------------------------------------------------------------------------
# Optional: per-stage breakdown for backbone
# ---------------------------------------------------------------------------


def _format_per_stage(model: nn.Module, label: str) -> None:
    bb = getattr(model, "backbone", None)
    if bb is None:
        return
    print(f"\n=== [{label}] per-stage parameter memory ===")
    rows = []
    if hasattr(bb, "input_stem"):
        n, b = _module_param_stats(bb.input_stem)
        rows.append(("input_stem", n, b))
    for i, stage in enumerate(getattr(bb, "stages", [])):
        n, b = _module_param_stats(stage)
        rows.append((f"stages[{i}]", n, b))
    if hasattr(model, "head"):
        n, b = _module_param_stats(model.head)
        rows.append(("head", n, b))

    total_n = sum(r[1] for r in rows)
    print(f"{'name':<14}{'numel':>14}{'MB':>12}{'%':>9}")
    print("-" * 50)
    for name, n, b in rows:
        print(f"{name:<14}{n:>14,}{b/1e6:>12.3f}{100.0*n/max(total_n,1):>8.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_into(model: nn.Module, ckpt_path: str) -> None:
    state_dict = _load_state_dict_from_ckpt(ckpt_path)
    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing 5: {unexpected[:5]})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Component-wise parameter memory of EfficientViT cls models."
    )
    parser.add_argument("--model", required=True, help="model name in cls_model_zoo, e.g. efficientvit-b1")
    parser.add_argument("--checkpoint", default=None, help="optional pretrained checkpoint to load.")
    parser.add_argument(
        "--reduced",
        default=None,
        help="optional reduced checkpoint (state_dict from reduce_efficientvit_cls_model). "
        "If given, prints original-vs-reduced comparison.",
    )
    parser.add_argument(
        "--auto-reduce",
        action="store_true",
        help="If --checkpoint is a SOFT-PRUNED ckpt (still original shape, "
        "with zeros), apply reducing on the fly to compute the reduced breakdown.",
    )
    parser.add_argument("--per-stage", action="store_true", help="also show per-stage backbone breakdown.")
    args = parser.parse_args(argv)

    print(f"=> creating model {args.model} (pretrained=False)")
    model = create_efficientvit_cls_model(args.model, pretrained=False)
    if args.checkpoint:
        _load_into(model, args.checkpoint)
        print(f"=> loaded weights from {args.checkpoint}")

    orig_summary = _format_breakdown(model, label="model")
    if args.per_stage:
        _format_per_stage(model, label="model")

    if args.reduced is not None:
        reduced_model = create_efficientvit_cls_model(args.model, pretrained=False)
        # reduced 체크포인트는 shape 가 다르므로, 우선 원본 shape 모델에 soft-pruned ckpt
        # (zeros 포함) 를 거쳐 reducing 하는 방식이 가장 일반적이다. 사용자가 "이미 reduced
        # 된 state_dict" 를 줬다면 strict=False 로 넘기고, 어쨌든 reducing 을 한 번 더 적용해
        # 0 패딩이 있을 경우 정리해 둔다.
        _load_into(reduced_model, args.reduced)
        reduce_efficientvit_cls_model(reduced_model)
        reduced_summary = _format_breakdown(reduced_model, label="reduced")
        if args.per_stage:
            _format_per_stage(reduced_model, label="reduced")
        _format_compare(orig_summary, reduced_summary)
    elif args.auto_reduce and args.checkpoint:
        # Same model object, in-place reduce.
        reduce_efficientvit_cls_model(model)
        reduced_summary = _format_breakdown(model, label="reduced (auto)")
        if args.per_stage:
            _format_per_stage(model, label="reduced (auto)")
        _format_compare(orig_summary, reduced_summary)


if __name__ == "__main__":
    main()
