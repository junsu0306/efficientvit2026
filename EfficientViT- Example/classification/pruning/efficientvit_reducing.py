"""
EfficientViT Reducing (Sparse → Dense 변환)
============================================
Soft Pruning 학습 완료 후, weight가 0인 채널을 물리적으로 제거하여
작고 빠른 Dense 모델을 생성한다.

실행 방법 (서버):
  python -m classification.pruning.efficientvit_reducing \\
    --model EfficientViT_M4 \\
    --checkpoint /workspace/.../checkpoint_best.pth \\
    --output /workspace/.../reduced_m4.pth

압축률 공식: 100 × (B - A) / B
"""

import argparse
import sys
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 저수준 헬퍼
# ---------------------------------------------------------------------------

def _get_survived_conv_out_idx(conv: nn.Conv2d) -> torch.Tensor:
    """Conv2d 출력 필터 기준으로 L2 norm > 0 인 인덱스 반환."""
    n = conv.weight.shape[0]
    norms = torch.norm(conv.weight.view(n, -1), dim=1)
    return torch.where(norms != 0)[0]


def _reduce_conv_bn(old_conv: nn.Conv2d, old_bn: nn.BatchNorm2d,
                    out_idx: torch.Tensor,
                    in_idx: torch.Tensor = None,
                    groups: int = 1) -> tuple:
    """
    Conv2d + BN 쌍을 survived 인덱스 기준으로 축소한 새 레이어를 반환.
    groups > 1 이면 DWConv 용으로 groups=len(out_idx) 로 생성.
    """
    n_out = len(out_idx)
    n_in  = old_conv.in_channels if in_idx is None else len(in_idx)
    ks, stride, pad = old_conv.kernel_size, old_conv.stride, old_conv.padding

    new_conv = nn.Conv2d(n_in, n_out, ks, stride=stride, padding=pad,
                         dilation=old_conv.dilation, groups=groups, bias=False)
    new_bn   = nn.BatchNorm2d(n_out)

    with torch.no_grad():
        w = old_conv.weight[out_idx]
        if in_idx is not None:
            w = w[:, in_idx] if groups == 1 else w   # DWConv: in_per_group=1
        new_conv.weight.copy_(w)
        new_bn.weight.copy_(old_bn.weight[out_idx])
        new_bn.bias.copy_(old_bn.bias[out_idx])
        new_bn.running_mean.copy_(old_bn.running_mean[out_idx])
        new_bn.running_var.copy_(old_bn.running_var[out_idx])
        new_bn.eps      = old_bn.eps
        new_bn.momentum = old_bn.momentum
    return new_conv, new_bn


def _replace_conv_bn(module: nn.Module, new_conv: nn.Conv2d,
                     new_bn: nn.BatchNorm2d) -> None:
    """Conv2d_BN Sequential의 .c, .bn 를 in-place 교체."""
    module.c  = new_conv
    module.bn = new_bn


# ---------------------------------------------------------------------------
# G_FFN reducing
# ---------------------------------------------------------------------------

def _reduce_ffn(ffn) -> None:
    """FFN expand(pw1) survived 채널 기준으로 pw1/pw2 in-place 축소."""
    survived = _get_survived_conv_out_idx(ffn.pw1.c)
    if len(survived) == ffn.pw1.c.weight.shape[0]:
        return

    # pw1: out 채널 축소
    new_c1, new_bn1 = _reduce_conv_bn(ffn.pw1.c, ffn.pw1.bn, survived)
    _replace_conv_bn(ffn.pw1, new_c1, new_bn1)

    # pw2: in 채널 축소, out 채널(ed) 유지
    old_c2 = ffn.pw2.c
    new_c2 = nn.Conv2d(len(survived), old_c2.out_channels,
                       old_c2.kernel_size, stride=old_c2.stride,
                       padding=old_c2.padding, bias=False)
    with torch.no_grad():
        new_c2.weight.copy_(old_c2.weight[:, survived])
    ffn.pw2.c = new_c2


# ---------------------------------------------------------------------------
# G_QK reducing
# ---------------------------------------------------------------------------

def _reduce_cga_qk(cga) -> None:
    """
    CGA qkvs[i], dws[i] in-place 축소.
    Q survived idx → K 동일 (soft pruning 시 동일 인덱스 적용됨).
    cga.key_dim, cga.scale 업데이트.
    """
    key_dim  = cga.key_dim
    new_kd   = None

    for i in range(cga.num_heads):
        qkv  = cga.qkvs[i]
        conv, bn = qkv.c, qkv.bn
        d = cga.d

        # Q survived (0:key_dim)
        q_norms = torch.norm(conv.weight[:key_dim].view(key_dim, -1), dim=1)
        survived_q = torch.where(q_norms != 0)[0]
        if i == 0:
            new_kd = len(survived_q)

        survived_k = survived_q + key_dim
        survived_v = torch.arange(2 * key_dim, 2 * key_dim + d,
                                   device=conv.weight.device)
        out_idx = torch.cat([survived_q, survived_k, survived_v])

        new_conv = nn.Conv2d(conv.in_channels, len(out_idx),
                             conv.kernel_size, stride=conv.stride,
                             padding=conv.padding, bias=False)
        new_bn = nn.BatchNorm2d(len(out_idx))
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight[out_idx])
            new_bn.weight.copy_(bn.weight[out_idx])
            new_bn.bias.copy_(bn.bias[out_idx])
            new_bn.running_mean.copy_(bn.running_mean[out_idx])
            new_bn.running_var.copy_(bn.running_var[out_idx])
            new_bn.eps      = bn.eps
            new_bn.momentum = bn.momentum
        _replace_conv_bn(qkv, new_conv, new_bn)

        # dws[i]: DWConv on Q
        dw_c, dw_bn = _reduce_conv_bn(
            cga.dws[i].c, cga.dws[i].bn,
            survived_q, survived_q,
            groups=len(survived_q),
        )
        _replace_conv_bn(cga.dws[i], dw_c, dw_bn)

    if new_kd is not None and new_kd != key_dim:
        cga.key_dim = new_kd
        cga.scale   = new_kd ** -0.5


# ---------------------------------------------------------------------------
# G_PE1/PE2/PE3 reducing
# ---------------------------------------------------------------------------

def _reduce_patch_embed(model: nn.Module) -> None:
    """
    PatchEmbed PE1→PE2→PE3→PE4(입력만) chain을 in-place 축소.
    PE4 출력(=embed_dim)은 blocks1 연결로 고정, 절대 변경 불가.

    patch_embed 인덱스: [0]=PE1, [2]=PE2, [4]=PE3, [6]=PE4
    """
    pe = model.patch_embed

    s1 = _get_survived_conv_out_idx(pe[0].c)   # PE1 survived out
    s2 = _get_survived_conv_out_idx(pe[2].c)   # PE2 survived out
    s3 = _get_survived_conv_out_idx(pe[4].c)   # PE3 survived out

    # PE1: in=3(고정), out→s1
    if len(s1) < pe[0].c.weight.shape[0]:
        c1, bn1 = _reduce_conv_bn(pe[0].c, pe[0].bn, s1, None)
        _replace_conv_bn(pe[0], c1, bn1)

    # PE2: in→s1, out→s2
    if len(s1) < pe[2].c.weight.shape[1] or len(s2) < pe[2].c.weight.shape[0]:
        c2, bn2 = _reduce_conv_bn(pe[2].c, pe[2].bn, s2, s1)
        _replace_conv_bn(pe[2], c2, bn2)

    # PE3: in→s2, out→s3
    if len(s2) < pe[4].c.weight.shape[1] or len(s3) < pe[4].c.weight.shape[0]:
        c3, bn3 = _reduce_conv_bn(pe[4].c, pe[4].bn, s3, s2)
        _replace_conv_bn(pe[4], c3, bn3)

    # PE4: in→s3, out=embed_dim 고정
    if len(s3) < pe[6].c.weight.shape[1]:
        old_c4 = pe[6].c
        new_c4 = nn.Conv2d(len(s3), old_c4.out_channels,
                           old_c4.kernel_size, stride=old_c4.stride,
                           padding=old_c4.padding, bias=False)
        with torch.no_grad():
            new_c4.weight.copy_(old_c4.weight[:, s3])
        pe[6].c = new_c4   # PE4 BN 출력 채널 불변, 그대로 유지


# ---------------------------------------------------------------------------
# SE 속성명 호환 헬퍼 (timm 버전별 차이 대응)
# 구버전: conv_reduce / conv_expand
# 신버전(0.9+): fc1 / fc2
# ---------------------------------------------------------------------------

def _get_se_convs(se: nn.Module):
    if hasattr(se, 'conv_reduce'):
        return se.conv_reduce, se.conv_expand, 'conv_reduce', 'conv_expand'
    elif hasattr(se, 'fc1'):
        return se.fc1, se.fc2, 'fc1', 'fc2'
    else:
        raise AttributeError(
            f"SE 모듈 구조를 인식할 수 없음: {list(se._modules.keys())}"
        )


# ---------------------------------------------------------------------------
# G_INV reducing (PatchMerging conv1 + conv2 + SE + conv3)
# ---------------------------------------------------------------------------

def _reduce_patch_merging(pm: nn.Module) -> None:
    """
    PatchMerging conv1 survived 채널 기준으로
    conv1 / conv2(DWConv) / SE / conv3(입력만) 을 in-place 축소.

    SE(SqueezeExcite) 처리:
      - conv_reduce 입력: hid_dim → n_survived
      - conv_reduce 출력: old_red → new_red (= max(1, round(n_survived * ratio)))
        survived_reduce_idx: conv_reduce rows의 L2 norm 기준 상위 new_red 선택
      - conv_expand 입력: old_red → new_red (동일 idx)
      - conv_expand 출력: hid_dim → n_survived (동일 survived_hid_idx)
    """
    survived_hid = _get_survived_conv_out_idx(pm.conv1.c)
    n = len(survived_hid)
    old_hid = pm.conv1.c.weight.shape[0]

    if n == old_hid:
        return

    # conv1: out→survived
    c1, bn1 = _reduce_conv_bn(pm.conv1.c, pm.conv1.bn, survived_hid)
    _replace_conv_bn(pm.conv1, c1, bn1)

    # conv2 (DWConv, stride=2): channels → n
    c2, bn2 = _reduce_conv_bn(pm.conv2.c, pm.conv2.bn,
                               survived_hid, survived_hid,
                               groups=n)
    _replace_conv_bn(pm.conv2, c2, bn2)

    # SE (SqueezeExcite from timm) — 버전별 속성명 자동 감지
    se = pm.se
    old_cr, old_ce, cr_attr, ce_attr = _get_se_convs(se)
    old_red = old_cr.weight.shape[0]
    new_red = max(1, round(n * old_red / old_hid))

    # conv_reduce/fc1 survived output rows: 입력 survived_hid 선택 후 norm 상위 new_red
    cr_w = old_cr.weight[:, survived_hid, :, :]   # [old_red, n, 1, 1]
    cr_norms = torch.norm(cr_w.view(old_red, -1), dim=1)
    _, survived_red = torch.topk(cr_norms, new_red, largest=True)

    new_cr = nn.Conv2d(n, new_red, 1, bias=True)
    with torch.no_grad():
        new_cr.weight.copy_(cr_w[survived_red])
        new_cr.bias.copy_(old_cr.bias[survived_red])
    setattr(se, cr_attr, new_cr)

    ce_w = old_ce.weight[survived_hid][:, survived_red]   # [n, new_red, 1, 1]
    new_ce = nn.Conv2d(new_red, n, 1, bias=True)
    with torch.no_grad():
        new_ce.weight.copy_(ce_w)
        new_ce.bias.copy_(old_ce.bias[survived_hid])
    setattr(se, ce_attr, new_ce)

    # conv3: in→survived_hid, out(stage 채널) 유지
    old_c3 = pm.conv3.c
    new_c3 = nn.Conv2d(n, old_c3.out_channels, 1, bias=False)
    with torch.no_grad():
        new_c3.weight.copy_(old_c3.weight[:, survived_hid])
    pm.conv3.c = new_c3


# ---------------------------------------------------------------------------
# 모델 전체 reducing
# ---------------------------------------------------------------------------

def efficientvit_reducing(model: nn.Module) -> None:
    """
    Soft Pruning 완료 모델의 모든 0 채널을 물리적으로 제거 (in-place).
    학습 완료 후 한 번만 실행.

    처리 순서:
      1. G_FFN + G_QK : EfficientViTBlock, SubDWFFN
      2. G_INV         : PatchMerging (blocks2, blocks3)
      3. G_PE          : patch_embed PE1-3 chain
    """
    for block_list in [model.blocks1, model.blocks2, model.blocks3]:
        for block in block_list:
            t = type(block).__name__
            if t == 'EfficientViTBlock':
                _reduce_ffn(block.ffn0.m)
                _reduce_ffn(block.ffn1.m)
                _reduce_cga_qk(block.mixer.m.attn)
            elif t == 'Sequential':
                if len(block) >= 2 and hasattr(block[1], 'm') \
                        and type(block[1].m).__name__ == 'FFN':
                    _reduce_ffn(block[1].m)
            elif t == 'PatchMerging':
                _reduce_patch_merging(block)

    _reduce_patch_embed(model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_reduced_model(checkpoint_path: str,
                       model_name: str = 'EfficientViT_M4',
                       num_classes: int = 1000,
                       device: str = 'cpu') -> nn.Module:
    """
    Soft-pruned 체크포인트를 로드하고 reducing을 적용하여 실행 가능한 Dense 모델 반환.

    사용 예시:
        from pruning.efficientvit_reducing import load_reduced_model
        model = load_reduced_model('/path/to/checkpoint_best.pth')
        model.eval()
        out = model(img_tensor)

    Args:
        checkpoint_path: soft-pruning 학습 완료 체크포인트 (.pth)
                         또는 reducing 완료 후 저장된 체크포인트 (_reduced.pth)
        model_name: timm 등록 모델명 (default: 'EfficientViT_M4')
        num_classes: 분류 클래스 수 (default: 1000)
        device: 'cpu' or 'cuda:N'

    Returns:
        reducing 완료된 Dense nn.Module (eval 모드)
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import model.build  # noqa: F401  ← EfficientViT 계열 timm 등록
    from timm.models import create_model

    dev = torch.device(device)
    model = create_model(model_name, num_classes=num_classes,
                         pretrained=False, fuse=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)

    # state_dict shape으로 soft-pruned인지 reduced인지 자동 감지
    # soft-pruned: 원본 아키텍처 shape → reducing 필요
    # reduced: 이미 줄어든 shape → strict=False로 시도
    try:
        model.load_state_dict(sd, strict=True)
        # 로드 성공 → soft-pruned 체크포인트 → reducing 적용
        efficientvit_reducing(model)
    except RuntimeError:
        # shape 불일치 → 이미 reduced된 체크포인트
        # → 원본 모델에 reducing 적용 후 load (구조 재현 불가)
        # → full_model 키가 있으면 사용, 없으면 에러
        if 'full_model' not in ckpt:
            raise RuntimeError(
                "이미 reducing된 state_dict입니다. "
                "soft-pruned 체크포인트(checkpoint_best.pth)를 입력하거나, "
                "--save-full 옵션으로 저장된 파일(_full.pth)을 사용하세요."
            )
        model = ckpt['full_model']

    model.to(dev).eval()
    return model


def _parse_args():
    p = argparse.ArgumentParser(description='EfficientViT Reducing')
    p.add_argument('--model', default='EfficientViT_M4')
    p.add_argument('--checkpoint', required=True,
                   help='soft-pruning 학습 완료 체크포인트 경로')
    p.add_argument('--output', required=True,
                   help='저장 경로 (.pth). _full.pth도 함께 저장됨')
    p.add_argument('--num-classes', default=1000, type=int)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # classification/

    import model.build  # noqa: F401  ← @register_model 실행 → timm에 EfficientViT 등록
    from timm.models import create_model
    model = create_model(args.model, num_classes=args.num_classes,
                         pretrained=False, fuse=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    B = count_params(model)
    print(f"[Reducing] 원본 파라미터: {B:,}")

    efficientvit_reducing(model)

    A = count_params(model)
    rate = 100.0 * (B - A) / B
    print(f"[Reducing] 축소 후 파라미터: {A:,}")
    print(f"[Reducing] 압축률: {rate:.2f}%")

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224, device=device))
    assert out.shape == (1, args.num_classes), f"Forward 오류: {out.shape}"
    print("[Reducing] Forward pass 검증 OK")

    # state_dict 저장 (추후 load_reduced_model()로 soft-pruned ckpt 경유 로드)
    torch.save({'model': model.state_dict(), 'compression_rate': rate}, args.output)
    print(f"[Reducing] state_dict 저장 완료: {args.output}")

    # full model 저장 → torch.load()로 즉시 사용 가능
    full_path = args.output.replace('.pth', '_full.pth')
    torch.save(model, full_path)
    print(f"[Reducing] full model 저장 완료: {full_path}")
    print(f"[Reducing] 사용법: model = torch.load('{full_path}').eval()")


if __name__ == '__main__':
    main()
