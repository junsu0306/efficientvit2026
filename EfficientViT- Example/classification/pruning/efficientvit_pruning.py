"""
EfficientViT Soft Pruning
=========================
매 optimizer.step() 이후 호출되어 L2-norm 하위 weight를 0으로 마스킹.

압축률 정의:
    compression = (제거된 파라미터 수) / (전체 파라미터 수)
    → target_compression=0.30 이면 전체 파라미터의 30%를 제거

Sparsity 계산 원칙:
    모든 prunable 그룹(G_FFN, G_QK, G_PE1-3, G_INV)의 파라미터 수를 합산한 뒤,
    "전체 파라미터 수 대비 목표 제거량"을 prunable 합계로 나눠 동일 sparsity 적용.
    → 파라미터 수에 비례하여 균등 분배 (FFN이 많으면 절대수도 많이 제거).

최소 생존 채널 (_MIN_SURVIVE):
    소규모 레이어(PE1=16채널 등)에서 int() 절삭으로 pruning이 0이 되는 문제 방지.
    round() 사용 + 반드시 _MIN_SURVIVE개 이상 생존 보장.

Prunable 그룹 (M4 기준):
    G_FFN  : FFN pw1(expand)+pw2(shrink) 전체  ~5,926K  67.3%
    G_QK   : CGA Q+K proj + dws           ~68K    0.8%
    G_PE1  : PatchEmbed Conv1 (3→16)      ~0.5K   0.0%
    G_PE2  : PatchEmbed Conv2 (16→32)     ~5K     0.1%
    G_PE3  : PatchEmbed Conv3 (32→64)     ~19K    0.2%
    G_INV  : PatchMerging conv1+conv3     ~855K   9.7%

Non-prunable (절대 건드리지 않음):
    G_V      : CGA V projection
    W_out    : CGA Output Projection
    PM_DW    : PatchMerging DWConv
    PM_SE    : PatchMerging SqueezeExcite
    G_PE4    : PatchEmbed Conv4 (64→128) — blocks1 입력 채널 고정
    DWConv   : Token interaction DWConv (dw0, dw1)
    Head     : Classifier
    Attn_Bias: attention_biases

실측 레이어 경로 (efficientvit.py 확인):
    FFN.pw1 = Conv2d_BN(ed → h)   → .c (Conv2d), .bn (BatchNorm2d)
    FFN.pw2 = Conv2d_BN(h  → ed)  → .c (Conv2d), .bn (BatchNorm2d)
    CGA.qkvs[i] = Conv2d_BN(ed//H → key_dim*2+d)
    CGA.dws[i]  = DWConv on Q
    EfficientViTBlock: .ffn0.m, .ffn1.m (FFN), .mixer.m.attn (CGA)
    SubDWFFN (Sequential): [1].m (FFN)
    model.patch_embed: [0]=PE1, [2]=PE2, [4]=PE3, [6]=PE4
    PatchMerging: .conv1 (expand), .conv2 (DWConv), .se, .conv3 (reduce)
"""

import torch
import torch.nn as nn

# 레이어당 최소 생존 채널 수.
# 너무 작으면 해당 레이어 정보 손실, 너무 크면 압축률 미달.
_MIN_SURVIVE = 4


# ---------------------------------------------------------------------------
# 저수준 헬퍼
# ---------------------------------------------------------------------------

def _get_conv_pruning_idx(conv: nn.Conv2d, sparsity: float) -> torch.Tensor:
    """
    Conv2d 출력 필터의 L2 norm 기준으로 하위 sparsity 비율의 인덱스 반환.

    소규모 레이어 처리:
      - int() 절삭 대신 round() 사용 → e.g., 16ch × 0.30 = 4.8 → 5 (int 시 4)
      - num_filters - _MIN_SURVIVE 초과 pruning 금지 → 최소 _MIN_SURVIVE개 생존
    """
    with torch.no_grad():
        num_filters = conv.weight.shape[0]
        num_pruning = round(num_filters * sparsity)
        num_pruning = min(num_pruning, num_filters - _MIN_SURVIVE)
        num_pruning = max(0, num_pruning)
        if num_pruning == 0:
            return torch.tensor([], dtype=torch.long, device=conv.weight.device)
        norms = torch.norm(conv.weight.view(num_filters, -1), dim=1)
        _, idx = torch.topk(norms, num_pruning, largest=False)
    return idx


def _zero_out_filters(conv: nn.Conv2d, bn: nn.BatchNorm2d,
                      idx: torch.Tensor) -> None:
    """지정된 출력 필터(conv) + 대응 BN 채널을 0으로 마스킹."""
    if len(idx) == 0:
        return
    with torch.no_grad():
        conv.weight.data[idx] = 0.0
        bn.weight.data[idx]   = 0.0
        bn.bias.data[idx]     = 0.0
        bn.running_mean[idx]  = 0.0
        bn.running_var[idx]   = 1.0   # 분모 0 방지


def _zero_in_channels(conv: nn.Conv2d, idx: torch.Tensor) -> None:
    """지정된 입력 채널(column)을 0으로 마스킹. shrink / next-layer 연동용."""
    if len(idx) == 0:
        return
    with torch.no_grad():
        conv.weight.data[:, idx] = 0.0


# ---------------------------------------------------------------------------
# G_FFN pruning
# ---------------------------------------------------------------------------

def _prune_ffn(ffn, sparsity: float) -> None:
    """
    FFN expand(pw1) 출력 필터 기준으로 pruning.
    pw1(conv+BN) 출력 및 pw2(conv) 입력에 동일 인덱스 적용.
    expand.out_channels == shrink.in_channels 연결 차원 보존.
    """
    idx = _get_conv_pruning_idx(ffn.pw1.c, sparsity)
    _zero_out_filters(ffn.pw1.c, ffn.pw1.bn, idx)
    _zero_in_channels(ffn.pw2.c, idx)


# ---------------------------------------------------------------------------
# G_QK pruning
# ---------------------------------------------------------------------------

def _prune_cga_qk(cga, sparsity: float) -> None:
    """
    CGA 각 head qkvs[i] 에서 Q, K 채널을 동일 인덱스로 pruning.
    qkvs[i] 출력: [Q(0:key_dim) | K(key_dim:2*key_dim) | V(2*key_dim:)]
    Q 기준 → K에 동일 상대 인덱스 적용 (QK^T 차원 일치 필수).
    V, proj 는 절대 미접근.
    """
    key_dim     = cga.key_dim
    num_pruning = min(round(key_dim * sparsity),
                      key_dim - _MIN_SURVIVE)
    num_pruning = max(0, num_pruning)
    if num_pruning == 0:
        return

    for qkv in cga.qkvs:
        conv, bn = qkv.c, qkv.bn
        q_norms = torch.norm(conv.weight[:key_dim].view(key_dim, -1), dim=1)
        _, q_idx = torch.topk(q_norms, num_pruning, largest=False)
        _zero_out_filters(conv, bn, q_idx)
        k_idx = q_idx + key_dim
        with torch.no_grad():
            conv.weight.data[k_idx] = 0.0
            bn.weight.data[k_idx]   = 0.0
            bn.bias.data[k_idx]     = 0.0
            bn.running_mean[k_idx]  = 0.0
            bn.running_var[k_idx]   = 1.0


# ---------------------------------------------------------------------------
# G_PE1/PE2/PE3 pruning
# ---------------------------------------------------------------------------

def _prune_patch_embed(model: nn.Module, sparsity: float) -> None:
    """
    PatchEmbed Conv1(PE1), Conv2(PE2), Conv3(PE3) 출력 필터를 pruning.
    각 레이어 독립적으로 sparsity 적용 후, 다음 레이어 입력 채널 연동 zero.

    chain: PE1 → PE2 → PE3 → PE4(입력만 연동, 출력=embed_dim 고정 금지)

    patch_embed 인덱스:
      [0]=PE1(Conv2d_BN), [1]=ReLU, [2]=PE2, [3]=ReLU,
      [4]=PE3, [5]=ReLU, [6]=PE4
    """
    pe = model.patch_embed

    # PE1 출력 → PE2 입력
    idx1 = _get_conv_pruning_idx(pe[0].c, sparsity)
    _zero_out_filters(pe[0].c, pe[0].bn, idx1)
    _zero_in_channels(pe[2].c, idx1)

    # PE2 출력 → PE3 입력
    idx2 = _get_conv_pruning_idx(pe[2].c, sparsity)
    _zero_out_filters(pe[2].c, pe[2].bn, idx2)
    _zero_in_channels(pe[4].c, idx2)

    # PE3 출력 → PE4 입력 (PE4 출력은 embed_dim=128 고정, 건드리지 않음)
    idx3 = _get_conv_pruning_idx(pe[4].c, sparsity)
    _zero_out_filters(pe[4].c, pe[4].bn, idx3)
    _zero_in_channels(pe[6].c, idx3)


# ---------------------------------------------------------------------------
# G_INV pruning (PatchMerging conv1/conv3)
# ---------------------------------------------------------------------------

def _prune_patch_merging(pm: nn.Module, sparsity: float) -> None:
    """
    PatchMerging.conv1(expand, 1×1) 출력 필터 기준으로 pruning.
    conv3(reduce, 1×1) 입력 채널에 동일 인덱스 적용.

    conv2(DWConv stride=2)와 SE는 forward에서 zeros가 자연스럽게 전파됨:
      - conv2: depthwise → zero input channel → zero output channel
      - SE: global avg → zero channel → squeeze/excite 계산에서 0 기여

    출력 채널(out_dim)은 다음 stage 입력과 연결 → conv3 출력 pruning 금지.
    """
    idx = _get_conv_pruning_idx(pm.conv1.c, sparsity)
    _zero_out_filters(pm.conv1.c, pm.conv1.bn, idx)
    _zero_in_channels(pm.conv3.c, idx)


# ---------------------------------------------------------------------------
# 모델 전체 순회
# ---------------------------------------------------------------------------

def efficientvit_pruning(model: nn.Module,
                         sparsity_ffn: float,
                         sparsity_qk: float) -> None:
    """
    EfficientViT 전체 모델에 soft pruning 적용. 매 optimizer.step() 이후 호출.

    동일 sparsity 값을 모든 prunable 그룹에 적용:
      G_FFN  : EfficientViTBlock ffn0/ffn1, SubDWFFN FFN
      G_QK   : EfficientViTBlock CGA QK
      G_PE   : patch_embed PE1, PE2, PE3
      G_INV  : blocks2, blocks3 내 PatchMerging conv1
    """
    # G_FFN + G_QK
    for block_list in [model.blocks1, model.blocks2, model.blocks3]:
        for block in block_list:
            t = type(block).__name__
            if t == 'EfficientViTBlock':
                _prune_ffn(block.ffn0.m, sparsity_ffn)
                _prune_ffn(block.ffn1.m, sparsity_ffn)
                _prune_cga_qk(block.mixer.m.attn, sparsity_qk)
            elif t == 'Sequential':
                if len(block) >= 2 and hasattr(block[1], 'm') \
                        and type(block[1].m).__name__ == 'FFN':
                    _prune_ffn(block[1].m, sparsity_ffn)
            elif t == 'PatchMerging':
                # G_INV
                _prune_patch_merging(block, sparsity_ffn)

    # G_PE1, PE2, PE3
    _prune_patch_embed(model, sparsity_ffn)


# ---------------------------------------------------------------------------
# Prunable 파라미터 계산
# ---------------------------------------------------------------------------

def count_prunable_params(model: nn.Module) -> dict:
    """
    Pruning으로 제거 가능한 파라미터 수를 그룹별로 계산.

    계산 방식:
      각 레이어의 '기본 prunable' = 해당 레이어 자신의 출력 필터 params + BN
      expand+shrink 쌍(FFN, G_INV)은 shrink 입력 side도 함께 계산.

    M4 실측 기준 (표):
      G_FFN  5,925,888  /  G_QK   68,480  /  G_INV  856,320
      G_PE1      464    /  G_PE2   4,672  /  G_PE3   18,560

    Returns dict:
      total, prunable_ffn, prunable_qk, prunable_pe, prunable_inv,
      prunable_total, ffn_fraction
    """
    total = sum(p.numel() for p in model.parameters())
    p_ffn = 0
    p_qk  = 0
    p_pe  = 0
    p_inv = 0

    # G_FFN + G_QK
    for block_list in [model.blocks1, model.blocks2, model.blocks3]:
        for block in block_list:
            t = type(block).__name__
            if t == 'EfficientViTBlock':
                for ffn in [block.ffn0.m, block.ffn1.m]:
                    h  = ffn.pw1.c.weight.shape[0]
                    ed = ffn.pw1.c.weight.shape[1]
                    p_ffn += h * ed + h * 2 + ed * h   # pw1.c + pw1.bn + pw2.c
                cga     = block.mixer.m.attn
                kd      = cga.key_dim
                for i, qkv in enumerate(cga.qkvs):
                    dpth = qkv.c.weight.shape[1]       # dim_per_head
                    ks   = cga.dws[i].c.kernel_size[0]
                    p_qk += 2 * kd * dpth + 2 * kd * 2        # Q+K conv + BN
                    p_qk += kd * ks * ks + kd * 2             # dws conv + BN
            elif t == 'Sequential':
                if len(block) >= 2 and hasattr(block[1], 'm') \
                        and type(block[1].m).__name__ == 'FFN':
                    ffn = block[1].m
                    h  = ffn.pw1.c.weight.shape[0]
                    ed = ffn.pw1.c.weight.shape[1]
                    p_ffn += h * ed + h * 2 + ed * h
            elif t == 'PatchMerging':
                # G_INV: conv1(expand) + conv3 input side
                hid    = block.conv1.c.weight.shape[0]
                in_d   = block.conv1.c.weight.shape[1]
                out_d  = block.conv3.c.weight.shape[0]
                p_inv += hid * in_d + hid * 2       # conv1.c + conv1.bn
                p_inv += hid * out_d                 # conv3.c input side

    # G_PE1, PE2, PE3 (primary: own output filters + BN)
    pe = model.patch_embed
    for pe_idx in [0, 2, 4]:   # PE1, PE2, PE3 (PE4 출력 고정)
        c = pe[pe_idx].c
        p_pe += c.weight.numel() + c.weight.shape[0] * 2

    p_total = p_ffn + p_qk + p_pe + p_inv
    return {
        'total':          total,
        'prunable_ffn':   p_ffn,
        'prunable_qk':    p_qk,
        'prunable_pe':    p_pe,
        'prunable_inv':   p_inv,
        'prunable_total': p_total,
        'ffn_fraction':   p_ffn / total,
    }


# ---------------------------------------------------------------------------
# 이진탐색 기반 정확한 sparsity 계산
# ---------------------------------------------------------------------------
# count_prunable_params는 직접 pruning 대상(G_FFN/QK/PE/INV)만 계산.
# 하지만 reducing 시 추가로 제거되는 파라미터가 존재:
#   - PM_SE (SqueezeExcite): hid과 SE 내부 rd 양방향 축소 → 이차 효과 (2s-s²)
#   - PM_DWConv: hid 축소에 연동
#   - PE4 입력: PE3 출력 축소에 cascade
#   - PE2/PE3 weight: 입출력 양방향 축소 (cascade 이차 효과)
# 이 모든 효과를 _estimate_total_removed()로 정확히 계산하고
# _find_sparsity()의 이진탐색에서 target과 일치시킨다.
# ---------------------------------------------------------------------------

def _se_rd(se: nn.Module):
    """SE reduce conv의 (out_channels, has_bias) 반환. timm 버전 호환."""
    if hasattr(se, 'conv_reduce'):
        c = se.conv_reduce
    elif hasattr(se, 'fc1'):
        c = se.fc1
    else:
        raise AttributeError(f"SE 구조 인식 불가: {list(se._modules.keys())}")
    return c.weight.shape[0], (c.bias is not None)


def _est_ffn_removed(ffn, s: float) -> int:
    """FFN pw1 출력 rows pruning → pw1.c + pw1.bn + pw2.c 입력 제거량 (선형)."""
    h      = ffn.pw1.c.weight.shape[0]   # expanded hidden dim
    ed_in  = ffn.pw1.c.weight.shape[1]   # embedding dim (pw1 input)
    ed_out = ffn.pw2.c.weight.shape[0]   # embedding dim (pw2 output)
    p = min(round(h * s), h - _MIN_SURVIVE)
    p = max(0, p)
    # pw1.c rows: p×ed_in + pw1.bn weight+bias: p×2 + pw2.c cols: ed_out×p
    return p * (ed_in + 2 + ed_out)


def _est_cga_removed(cga, s: float) -> int:
    """CGA Q+K rows + dws rows 제거량 (선형)."""
    kd = cga.key_dim
    p = min(round(kd * s), kd - _MIN_SURVIVE)
    p = max(0, p)
    removed = 0
    for i, qkv in enumerate(cga.qkvs):
        dpth = qkv.c.weight.shape[1]          # dim_per_head (input channels)
        ks   = cga.dws[i].c.kernel_size[0]
        removed += p * (dpth + 2) * 2         # Q rows + K rows (weight+BN)
        removed += p * (ks * ks + 2)          # dws conv + BN
    return removed


def _est_pm_removed(pm, s: float) -> int:
    """PatchMerging: conv1 + conv2(DW) + SE + conv3 전체 제거량.
    SE는 hid·rd 양방향 축소로 이차 효과 포함.
    """
    hid   = pm.conv1.c.weight.shape[0]
    in_d  = pm.conv1.c.weight.shape[1]
    out_d = pm.conv3.c.weight.shape[0]
    p = min(round(hid * s), hid - _MIN_SURVIVE)
    p = max(0, p)
    n = hid - p                               # survived hid channels

    removed = 0
    # conv1.c + conv1.bn
    removed += p * (in_d + 2)
    # conv2 DWConv [hid,1,ks,ks] + BN
    ks = pm.conv2.c.kernel_size[0]
    removed += p * (ks * ks + 2)
    # conv3.c 입력 side
    removed += p * out_d

    # SE: conv_reduce [old_rd, hid] → [new_rd, n],  conv_expand [hid, old_rd] → [n, new_rd]
    old_rd, has_bias = _se_rd(pm.se)
    new_rd = max(1, round(n * old_rd / hid))
    removed += old_rd * hid - new_rd * n      # conv_reduce weight
    removed += hid * old_rd - n * new_rd      # conv_expand weight
    if has_bias:
        removed += (old_rd - new_rd) + (hid - n)   # 각 conv bias

    return removed


def _est_pe_removed(model, s: float) -> int:
    """PatchEmbed PE1~3 + PE4 cascade 제거량.
    PE2.c / PE3.c 는 입출력 양방향 축소 (cascade 이차 효과).
    PE4.c 는 입력만 축소 (출력=embed_dim 고정).
    """
    pe = model.patch_embed
    removed = 0

    # PE1: [n1, n_in, k, k]  (n_in = 3, 고정)
    n1   = pe[0].c.weight.shape[0]
    n_in = pe[0].c.weight.shape[1]     # = 3
    k1   = pe[0].c.kernel_size[0]
    p1   = min(round(n1 * s), n1 - _MIN_SURVIVE); p1 = max(0, p1)
    n1_s = n1 - p1
    removed += p1 * (n_in * k1 * k1 + 2)    # PE1.c rows + PE1.bn

    # PE2: [n2, n1, k, k] — 입출력 모두 축소 (cascade)
    n2 = pe[2].c.weight.shape[0]
    k2 = pe[2].c.kernel_size[0]
    p2 = min(round(n2 * s), n2 - _MIN_SURVIVE); p2 = max(0, p2)
    n2_s = n2 - p2
    removed += (n2 * n1 - n2_s * n1_s) * k2 * k2    # weight cascade
    removed += p2 * 2                                  # PE2.bn (출력만)

    # PE3: [n3, n2, k, k] — 입출력 모두 축소 (cascade)
    n3 = pe[4].c.weight.shape[0]
    k3 = pe[4].c.kernel_size[0]
    p3 = min(round(n3 * s), n3 - _MIN_SURVIVE); p3 = max(0, p3)
    n3_s = n3 - p3
    removed += (n3 * n2 - n3_s * n2_s) * k3 * k3    # weight cascade
    removed += p3 * 2                                  # PE3.bn

    # PE4: [n4, n3, k, k] — 입력만 축소 (출력=embed_dim 절대 불변)
    n4 = pe[6].c.weight.shape[0]
    k4 = pe[6].c.kernel_size[0]
    removed += n4 * p3 * k4 * k4

    return removed


def _estimate_total_removed(model: nn.Module, s: float) -> int:
    """sparsity s 적용 후 reducing으로 제거될 총 파라미터 수 추정.

    모든 축소 컴포넌트를 포함:
      FFN, CGA QK/dws       : 선형 (s에 비례)
      PatchMerging SE        : 이차 (hid × rd 양방향 → 2s-s² 인수)
      PatchMerging DWConv    : 선형
      PatchEmbed PE2/PE3     : 이차 cascade
      PatchEmbed PE4 입력    : 선형 cascade
    """
    removed = 0
    for block_list in [model.blocks1, model.blocks2, model.blocks3]:
        for block in block_list:
            t = type(block).__name__
            if t == 'EfficientViTBlock':
                removed += _est_ffn_removed(block.ffn0.m, s)
                removed += _est_ffn_removed(block.ffn1.m, s)
                removed += _est_cga_removed(block.mixer.m.attn, s)
            elif t == 'Sequential':
                if len(block) >= 2 and hasattr(block[1], 'm') \
                        and type(block[1].m).__name__ == 'FFN':
                    removed += _est_ffn_removed(block[1].m, s)
            elif t == 'PatchMerging':
                removed += _est_pm_removed(block, s)
    removed += _est_pe_removed(model, s)
    return removed


def _find_sparsity(model: nn.Module, target_compression: float,
                   max_s: float = 0.95) -> float:
    """이진탐색으로 target_compression을 달성하는 sparsity 반환.

    SE 이차 효과 + PE cascade를 정확히 반영하므로
    반환된 sparsity로 pruning+reducing하면 target과 실제 압축률이 일치.
    """
    total          = sum(p.numel() for p in model.parameters())
    target_remove  = target_compression * total
    lo, hi         = 0.0, max_s
    for _ in range(64):            # 64회 → 약 1e-19 정밀도
        mid = (lo + hi) * 0.5
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


# ---------------------------------------------------------------------------
# Pruner 클래스
# ---------------------------------------------------------------------------

class EfficientViTPruner:
    """
    EfficientViT용 Soft Pruner.
    engine.py 의 loss_scaler() 직후 pruner.apply(model) 호출.

    Pruning 그룹:
        G_FFN  : FFN expand+shrink  (dominant, ~67%)
        G_QK   : CGA Q+K + dws     (~1%)
        G_PE   : PatchEmbed 1-3     (~0.3%)
        G_INV  : PatchMerging 1x1   (~10%)

    Sparsity 계산 (이진탐색 기반, 정확):
        _find_sparsity()가 SE 이차 효과 + PE cascade를 포함하여
        target_compression과 실제 압축률이 일치하는 sparsity를 탐색.

        구 공식 (단순 선형): sparsity = target × total / prunable_total
          → SE/DW/PE4 cascade 누락으로 실제 압축률이 target보다 5~9% 과도 압축됨.

    M4 이진탐색 결과:
        target 30% → sparsity ≈ 0.295  (실제 ≈ 30%)
        target 50% → sparsity ≈ 0.505  (실제 ≈ 50%)
        target 70% → sparsity ≈ 0.750  (실제 ≈ 70%)
        target 75% → sparsity ≈ 0.820  (실제 ≈ 75%)
    """

    PRESETS = [0.30, 0.50, 0.70, 0.75]

    def __init__(self, model: nn.Module, target_compression: float = 0.30):
        total = sum(p.numel() for p in model.parameters())

        # ── 이진탐색: SE 이차 효과 + PE cascade 포함, target과 실제 압축률 일치 ──
        s = _find_sparsity(model, target_compression)
        self.sparsity_ffn = min(s, 0.95)
        self.sparsity_qk  = min(s, 0.90)

        # 추정 검증
        est_removed     = _estimate_total_removed(model, s)
        est_compression = est_removed / total

        # 참고용: 직접 pruning 대상 파라미터 수
        info = count_prunable_params(model)

        print(
            f"[EfficientViTPruner]\n"
            f"  total={total:,}\n"
            f"  직접prunable(G_FFN+QK+PE+INV)={info['prunable_total']:,}"
            f"  cascade포함추정={est_removed:,}\n"
            f"  target={target_compression:.1%}  bisection_sparsity={s:.4f}\n"
            f"  estimated_compression={est_compression:.2%}"
            + ("  [sparsity_qk 0.90 클리핑]" if s > 0.90 else "")
        )

    def apply(self, model: nn.Module) -> None:
        """optimizer.step() 직후 호출. DDP → model.module 전달."""
        efficientvit_pruning(model, self.sparsity_ffn, self.sparsity_qk)

    def log_sparsity(self, model: nn.Module) -> dict:
        """현재 모델 FFN의 실제 zero 비율 (검증/로깅용)."""
        total, zeros = 0, 0
        for block_list in [model.blocks1, model.blocks2, model.blocks3]:
            for block in block_list:
                if type(block).__name__ == 'EfficientViTBlock':
                    for res in [block.ffn0, block.ffn1]:
                        w = res.m.pw1.c.weight
                        total += w.numel()
                        zeros += int((w == 0).sum())
        return {
            'total_ffn_params': total,
            'zero_ffn_params':  zeros,
            'actual_sparsity':  zeros / total if total > 0 else 0.0,
        }
