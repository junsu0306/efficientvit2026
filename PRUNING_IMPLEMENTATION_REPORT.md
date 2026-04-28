# EfficientViT (분류) — Soft Pruning 적용 보고서

> 본 보고서는 [PRUNING_METHODOLOGY.md](PRUNING_METHODOLOGY.md) 의 방법론을
> EfficientViT classification 모델(`efficientvit-{b0,b1,b2,b3,l1,l2,l3}`) 에
> 적용한 구체적인 구현 결과를 정리한다.

---

## 0. 한눈에 보기

| 항목 | 내용 |
|------|------|
| 적용 대상 | `applications/efficientvit_cls/` (분류 학습/추론 파이프라인) |
| 기준 모델 | **EfficientViT-B1** (config: `efficientvit_b1.yaml`) |
| 방법론 | Soft Pruning (학습 중 0 마스킹) → Reducing (학습 후 dense 변환) |
| Prunable 그룹 | `MBConv.mid_channels`, `FusedMBConv.mid_channels`, **`input_stem` chain** (stage0 채널) |
| 학습 코드 변화량 | 추가/수정된 라인 수 ~25 줄 (기존 학습 흐름 그대로 유지) |
| 신규 파일 | `efficientvit/clscore/pruning/` 패키지 + `applications/efficientvit_cls/{reduce_efficientvit_cls_model.py, measure_memory.py}` |
| 수정 파일 | `efficientvit/apps/trainer/base.py`, `efficientvit/clscore/trainer/cls_trainer.py`, `applications/efficientvit_cls/train_efficientvit_cls_model.py` |
| Opt-in 플래그 | `--target_compression <0~1>` (0 이면 pruning 비활성, 기존 학습과 동일) |
| 참고 레퍼런스 | `EfficientViT- Example/` (M-series Soft Pruning 구현; B/L 시리즈 적용 시 매핑 §4.4 참고) |

---

## 1. 목차

1. [한눈에 보기](#0-한눈에-보기)
2. [디렉터리 / 파일 변화 요약](#2-디렉터리--파일-변화-요약)
3. [EfficientViT 아키텍처 분석](#3-efficientvit-아키텍처-분석)
4. [Prunable / Non-Prunable 결정 근거](#4-prunable--non-prunable-결정-근거)
   - [4.4 예제(M-series) ↔ 우리(B/L) 매핑](#44-예제-m-series--우리-bl-series-매핑)
   - [4.5 input_stem chain 단일 인덱스 Pruning](#45-input_stem-chain--단일-인덱스-pruning)
5. [Soft Pruning 구현 (학습 중)](#5-soft-pruning-구현-학습-중) — input_stem 마스킹 포함
6. [Reducing 구현 (학습 후)](#6-reducing-구현-학습-후) — input_stem 변환 포함
7. [학습 루프와의 통합 — "최소 변경" 설계](#7-학습-루프와의-통합--최소-변경-설계)
8. [Sparsity 이진탐색 — Secondary Effect 처리](#8-sparsity-이진탐색--secondary-effect-처리) — stem cascade 포함
9. [실행 명령어 가이드](#9-실행-명령어-가이드) — B1, ImageNet, measure_memory 포함
10. [의사결정 / 트레이드오프 정리](#10-의사결정--트레이드오프-정리)
11. [메서드 별 줄 단위 변경 diff](#11-메서드-별-줄-단위-변경-diff)
12. [향후 확장 가능 그룹](#12-향후-확장-가능-그룹)
13. [업데이트 이력 (Changelog)](#14-업데이트-이력-changelog)

---

## 2. 디렉터리 / 파일 변화 요약

```
efficientvit2026/
├── efficientvit/
│   ├── apps/
│   │   └── trainer/
│   │       └── base.py                                  ← MODIFIED (after_step 에 hook 4줄 추가)
│   └── clscore/
│       ├── trainer/
│       │   └── cls_trainer.py                           ← MODIFIED (생성자 인자 1개 추가)
│       └── pruning/                                     ← NEW (패키지)
│           ├── __init__.py                              ← NEW
│           ├── efficientvit_pruning.py                  ← NEW (Soft Pruning 핵심)
│           └── efficientvit_reducing.py                 ← NEW (Reducing 핵심 + CLI)
└── applications/
    └── efficientvit_cls/
        ├── train_efficientvit_cls_model.py              ← MODIFIED (--target_compression 플래그 + 분기)
        ├── reduce_efficientvit_cls_model.py             ← NEW (Reducing CLI 진입점)
        └── measure_memory.py                            ← NEW (컴포넌트별 파라미터 메모리 분해)
```

수정량 통계 (rough):
- `base.py`: **+5 줄** (pruner hook 호출만)
- `cls_trainer.py`: **+3 줄** (생성자 인자 + 저장)
- `train_efficientvit_cls_model.py`: **+19 줄** (argparse + pruner 분기)
- 신규 라이브러리 코드: 약 **750 줄** (pruning + reducing + measure_memory)

→ **학습 코드 자체에 침투하는 라인은 총 8 줄.** 나머지는 모두 별도 모듈/CLI.

---

## 3. EfficientViT 아키텍처 분석

### 3.1 백본 두 계열

| 시리즈 | 클래스 | 기본 블록 구성 |
|--------|--------|---------------|
| B (b0~b3) | `EfficientViTBackbone` | input_stem(Conv+DSConv) → stage1~2 (`MBConv`×d) → stage3~4 (`MBConv` downsample + `EfficientViTBlock`×d) |
| L (l1~l3) | `EfficientViTLargeBackbone` | stage0(Conv+`ResBlock`) → stage1~2 (`FusedMBConv`×d) → stage3 (`MBConv`×d) → stage4 (`MBConv` downsample + `EfficientViTBlock`×d) |

`EfficientViTBlock` 내부:
- `context_module = ResidualBlock(LiteMLA, IdentityLayer())`
- `local_module   = ResidualBlock(MBConv | GLUMBConv, IdentityLayer())`

### 3.2 핵심 빌딩 블록의 채널 흐름

| 블록 | 채널 흐름 | "내부 hidden" |
|------|----------|--------------|
| `MBConv` | `in → mid (1×1 expand) → mid (k×k DW, groups=mid) → out (1×1 shrink)` | **mid_channels** |
| `FusedMBConv` | `in → mid (k×k spatial) → out (1×1 shrink)` | **mid_channels** |
| `DSConv` (expand=1) | `in → in (k×k DW) → out (1×1)` | 외부 채널만 존재 |
| `ResBlock` (expand=1) | `in → mid=in (k×k) → out=in (k×k)` | 외부 채널과 동일 |
| `LiteMLA` | `in → 3·heads·dim (qkv) + multi-scale aggreg → out (proj)` | 헤드 차원, multi-scale 결합 |

### 3.3 ConvLayer / BN 구조

EfficientViT 의 모든 conv 는 `ConvLayer` 래퍼를 거치며, 내부에 `.conv`(`nn.Conv2d`),
`.norm`(BN/LN/None), `.act` 를 가진다. Pruning 의 1차 대상은 `.conv.weight` 와
대응되는 `.norm.{weight,bias,running_mean,running_var}` 이다.

---

## 4. Prunable / Non-Prunable 결정 근거

### 4.1 Prunable (이번 구현에서 채택)

| 그룹명 | 위치 | 이유 |
|-------|------|------|
| **`MBConv.mid_channels`** | stage1~4 의 모든 `MBConv` (downsample 포함) | inverted_conv 출력=DW=point_conv 입력 의 hidden dim. 외부 채널과 무관 → "FFN-like" 표준 prunable. |
| **`FusedMBConv.mid_channels`** | L 시리즈 stage1~2 | spatial_conv 출력 = point_conv 입력. 동일 원리. |
| **`input_stem` chain** ([§4.5](#45-input_stem-chain-단일-인덱스-pruning)) | stage0 (`width_list[0]`) | stem 의 모든 레이어가 단일 채널 차원을 공유 → single index 로 chain 전체 + 다음 stage 첫 conv 입력 컬럼을 동기화 마스킹. 예제 (`G_PE` chain) 의 동급 처리. |

### 4.2 Non-Prunable (의도적 제외)

| 그룹 | 제외 사유 |
|------|---------|
| Stage 사이 채널 경계 (`width_list[1..4]`) | stage 간 채널 수는 backbone 의 외부 hyperparameter. 끊으면 channel mismatch. |
| `LiteMLA` (qkv / aggreg / proj) | 1) qkv 단일 텐서에 Q/K/V 가 인터리브 → 셋이 동일 인덱스로 묶여야 함, 2) multi-scale aggreg 와 proj 입력이 모두 `total_dim·(1+len(scales))` 로 결합되어 secondary effect 가 큼 → **정확도 보존 우선으로 이번 구현에서는 제외.** |
| `ClsHead` (`Conv → Linear → Linear`) | 헤드 hidden dim 도 prunable 하지만, 1) 헤드의 파라미터 비중이 "한 곳에 몰린" 형태라 동일 sparsity 적용 시 정확도 영향이 큼, 2) 백본만 잘라도 충분한 압축이 이미 가능 → 1차 적용에서 제외. ([향후 확장](#12-향후-확장-가능-그룹) 참조) |

### 4.3 Coupled (연동) 관계 — 자동으로 "동일 인덱스"

```
MBConv.inverted_conv.out_ch  (= mid)
   ↓ 같은 인덱스
MBConv.depth_conv (groups=mid 이므로 채널 == 그룹)
   ↓ 같은 인덱스
MBConv.point_conv.in_ch  (= mid)
```

```
FusedMBConv.spatial_conv.out_ch (= mid)
   ↓ 같은 인덱스
FusedMBConv.point_conv.in_ch (= mid)
```

```
input_stem.op_list[0].conv.out_ch  (= C0 = width_list[0])
   ↓ 같은 인덱스 (stem 전체 chain 공유)
DSConv.depth_conv (groups=C0)
   ↓ 같은 인덱스
DSConv.point_conv.in_ch == out_ch  (= C0, 양방향)
   ↓ 같은 인덱스
... (다른 DSConv 반복)
   ↓ 같은 인덱스
stages[0].op_list[0].main.inverted_conv.in_ch  (= C0, 다음 stage 첫 conv 입력 컬럼)
```

따라서 `inverted_conv` / `spatial_conv` / `input_stem.op_list[0].conv` 의
출력 필터 L2 norm 으로 인덱스를 산정한 뒤, 그 **하나의 인덱스 집합** 을 모든
coupled 위치에 적용한다.

### 4.4 예제 (M-series) ↔ 우리 (B/L-series) 매핑

`EfficientViT- Example/` 의 M-series 구현은 동일 방법론을 다른 아키텍처에
적용한다. 구조는 다르지만 **그룹 의미** 는 1:1 로 매핑된다:

| 예제 (M-series) | 우리 프로젝트 (B/L-series) | 처리 위치 |
|-----------------|----------------------------|-----------|
| `G_FFN` (FFN.pw1 → FFN.pw2 expand-shrink) | `G_MBCONV` (MBConv.inverted_conv → depth → point) | `_prune_mbconv` |
| `G_FFN` (SubDWFFN) | (해당 없음 — B/L 시리즈 stem 은 DSConv) | — |
| `G_INV` (PatchMerging conv1 → conv3 + SE) | downsample MBConv `mid_channels` | 일반 MBConv 처리에 흡수 (B/L 에는 별도 SE 없음) |
| `G_QK` (CGA Q+K) | LiteMLA Q/K | **이번 구현에서 제외** (multi-scale aggreg coupling) |
| `G_PE1~3` (PatchEmbed chain) | **`input_stem` chain** (Conv + DSConv*) | `_prune_input_stem` ([§4.5](#45-input_stem-chain-단일-인덱스-pruning)) |
| `G_PE4` 입력 (블록 입력 채널 고정) | `stages[0].op_list[0].main.inverted_conv` 입력 컬럼 | 동기화 마스킹 |
| Head (제외) | `ClsHead` (제외) | — |

> 예제의 `G_PE` 는 PE1, PE2, PE3 가 **각자 다른 채널 수** (16, 32, 64) 라
> 인덱스가 chain 단계마다 독립적이지만, 우리 `input_stem` 은 모든 inner 블록이
> **C0 단일 채널** 을 공유하므로 single index 로 chain 전체를 통일한다.

### 4.5 `input_stem` chain — 단일 인덱스 Pruning

#### 구조 (B-series 기준)

```
backbone.input_stem = OpSequential([
    ConvLayer(3 → C0, stride=2),                              # op_list[0]
    ResidualBlock(DSConv(C0→C0), IdentityLayer()),            # op_list[1]
    ResidualBlock(DSConv(C0→C0), IdentityLayer()),            # op_list[2..n]  (depth_list[0]>1 인 경우)
])

# DSConv 내부:
#     .depth_conv = ConvLayer(C0, C0, k, groups=C0)           # depthwise (in==out==C0)
#     .point_conv = ConvLayer(C0, C0, 1)                      # pointwise (in==out==C0)
```

L-series 의 stem 은 `backbone.stages[0]` 이며 `DSConv` 자리가 `ResBlock`
(`expand_ratio=1`) 으로 바뀐 동등 구조.

#### 마스킹 흐름

1. `idx = topk_smallest_l2(input_stem.op_list[0].conv.weight, n_prune)`
2. 같은 `idx` 를 다음 위치 모두에 적용:
   - `op_list[0]` 출력 필터 (Conv + BN)
   - 각 inner DSConv:
     - `depth_conv` 출력 필터 (groups=C0 이라 그대로 1:1)
     - `point_conv` 출력 필터 + 입력 컬럼 (in=out=C0 양방향)
   - 각 inner ResBlock (L-series): `conv1`, `conv2` 모두 양방향
3. **다음 stage** 첫 down-sampling 블록의 첫 ConvLayer 입력 컬럼:
   - B-series: `stages[0].op_list[0].main.inverted_conv` 입력 컬럼
   - L-series: `stages[1].op_list[0].main.spatial_conv` 입력 컬럼

#### 왜 forward 가 자연스럽게 0 을 전파하는가

- 첫 Conv 출력 채널 i 가 0 → DSConv 입력 채널 i 가 0
- DW(`depth_conv`) 채널 i 의 weight 가 0 → DW 출력 채널 i 가 0
- point_conv 출력 필터 i 의 weight 가 0 → main 출력 채널 i 가 0
- shortcut(IdentityLayer) 의 입력 채널 i 도 이미 0 (앞단에서 0)
- residual sum = main + shortcut = 0 on channel i → 다음 inner block 으로
  0 이 그대로 전파.

→ chain 끝의 `stages[0]` 첫 MBConv 입력 측에서도 채널 i 는 0 이므로
입력 컬럼 마스킹과 정합성 OK.

---

## 5. Soft Pruning 구현 (학습 중)

파일: [`efficientvit/clscore/pruning/efficientvit_pruning.py`](efficientvit/clscore/pruning/efficientvit_pruning.py)

### 5.1 클래스 인터페이스

```python
class EfficientViTPruner:
    def __init__(self, model, target_compression, max_sparsity=0.95, sparsity=None):
        # __init__ 내부에서 target_compression 만으로 per-group sparsity 를
        # 이진탐색으로 결정 (sparsity 파라미터로 override 가능).
        ...

    def apply(self, model: nn.Module) -> None:
        """매 optimizer.step() 직후 호출. 모든 prunable 그룹을 0 마스킹."""
        ...

    @torch.no_grad()
    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """검증용. 실제 마스킹된 zero 필터 비율을 반환."""
        ...
```

### 5.2 마스킹 로직 (MBConv)

```python
def _prune_mbconv(mb: MBConv, sparsity: float) -> None:
    weight = mb.inverted_conv.conv.weight              # (mid, in, 1, 1)
    mid = weight.shape[0]
    n_prune = _calc_n_prune(mid, sparsity)             # MIN_SURVIVE 보장
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)       # L2 norm 하위 k

    with torch.no_grad():
        # inverted_conv 출력 필터.
        mb.inverted_conv.conv.weight.data[idx] = 0.0
        if mb.inverted_conv.conv.bias is not None:
            mb.inverted_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(mb.inverted_conv.norm, idx)

        # depth_conv: groups=mid 이므로 동일 idx.
        mb.depth_conv.conv.weight.data[idx] = 0.0
        if mb.depth_conv.conv.bias is not None:
            mb.depth_conv.conv.bias.data[idx] = 0.0
        _zero_bn_(mb.depth_conv.norm, idx)

        # point_conv 입력 컬럼.
        mb.point_conv.conv.weight.data[:, idx] = 0.0
```

`FusedMBConv` 도 동일한 패턴으로 `spatial_conv` (출력 필터) ↔ `point_conv` (입력 컬럼) 을 1:1 짝짓는다.

### 5.2.1 `input_stem` 마스킹 (`_prune_input_stem`)

```python
def _prune_input_stem(model, sparsity):
    stem = _get_stem_op_seq(model)             # input_stem (B) 또는 stages[0] (L)
    first_cl = stem.op_list[0]                  # ConvLayer(3 → C0)
    n_total = first_cl.conv.weight.shape[0]
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(first_cl.conv.weight, n_prune)

    # 1) stem[0]: 출력 필터 + BN
    _zero_conv_out_filters_(first_cl, idx)

    # 2) stem[1..]: 각 inner DSConv 또는 ResBlock 양방향 마스킹
    for inner in _iter_stem_inner_blocks(stem):
        if isinstance(inner, DSConv):
            _zero_conv_out_filters_(inner.depth_conv, idx)        # groups=C0
            _zero_conv_out_filters_(inner.point_conv, idx)        # 출력
            _zero_conv_in_cols_(inner.point_conv, idx)            # 입력
        else:  # ResBlock (L-series stage0)
            _zero_conv_out_filters_(inner.conv1, idx)
            _zero_conv_in_cols_(inner.conv1, idx)
            _zero_conv_out_filters_(inner.conv2, idx)
            _zero_conv_in_cols_(inner.conv2, idx)

    # 3) stage1 첫 down-sampling Conv 입력 컬럼 동기화
    post = _get_post_stem_first_conv(model)
    if post is not None:
        _zero_conv_in_cols_(post, idx)
```

L-series 도 같은 함수로 처리되며, `_get_stem_op_seq` 가 `input_stem` 또는
`stages[0]` 을 자동으로 반환한다.

### 5.3 BN 처리 — 방법론 도큐먼트의 권고 그대로

```python
def _zero_bn_(bn, idx):
    if bn is None:
        return
    with torch.no_grad():
        if bn.weight is not None:  bn.weight.data[idx] = 0.0
        if bn.bias   is not None:  bn.bias.data[idx]   = 0.0
        if isinstance(bn, _BatchNorm):
            bn.running_mean.data[idx] = 0.0
            bn.running_var.data[idx]  = 1.0   # ← 0 이 아닌 1.0 (분모 0 방지)
```

### 5.4 안전장치

- **`MIN_SURVIVE = 4`**: 어떤 그룹이든 최소 4 채널은 살린다. (작은 hidden 에서 정보가 완전히 끊기는 사태 방지)
- **`max_sparsity=0.95`**: 이진탐색 상한.
- **`round()` 사용**: `int()` 의 버림으로 인한 소규모 그룹 편향 회피.
- **`sparsity == 0.0` 인 경우 `apply` 즉시 return**: pruner 자체가 attached 되어 있어도 안전한 no-op.

---

## 6. Reducing 구현 (학습 후)

파일: [`efficientvit/clscore/pruning/efficientvit_reducing.py`](efficientvit/clscore/pruning/efficientvit_reducing.py)

### 6.1 핵심 함수

```python
@torch.no_grad()
def reduce_efficientvit_cls_model(model: nn.Module) -> nn.Module:
    """모델 전체를 in-place 로 dense reduce. 반환은 동일 객체.

    순서:
      1) MBConv / FusedMBConv mid_channels 축소.
      2) Input stem chain 축소 + stage1 첫 conv 입력 컬럼 축소.
    """
    for module in model.modules():
        if isinstance(module, MBConv):
            _reduce_mbconv(module)
        elif isinstance(module, FusedMBConv):
            _reduce_fusedmbconv(module)
    _reduce_input_stem(model)
    return model
```

### 6.2 단일 블록 변환 흐름 (MBConv 예시)

1. `survived = where(L2norm(inverted_conv.weight) != 0)` 로 살아남은 인덱스 추출.
2. `inverted_conv.conv` 를 동일 설정의 새 `nn.Conv2d(in, n_new, ...)` 로 교체. weight/bias 는 `survived` 인덱싱.
3. `inverted_conv.norm` 을 `nn.BatchNorm2d(n_new)` 로 교체. weight/bias/running_mean/running_var 모두 `survived` 인덱싱.
4. `depth_conv.conv` 를 새 `nn.Conv2d(n_new, n_new, k, groups=n_new, ...)` 로 교체. **DWConv 의 channel == groups 인 점이 핵심.**
5. `point_conv.conv` 를 새 `nn.Conv2d(n_new, out_ch, 1, ...)` 로 교체. `weight[:, survived]` 로 입력 컬럼만 축소. `out_ch`, `point_conv.norm` 은 그대로 둔다 (외부 연결).

### 6.2.1 Input Stem 변환 흐름 (`_reduce_input_stem`)

1. `survived = where(L2norm(stem[0].conv.weight) != 0)` 로 stem 출력 채널 인덱스 추출.
2. `stem[0]` ConvLayer 의 `.conv` / `.norm` 을 새 `Conv2d(3 → len(survived))` /
   `BatchNorm2d(len(survived))` 로 교체.
3. 각 inner block (DSConv 또는 ResBlock):
   - DSConv: `depth_conv` 를 `Conv2d(n_new, n_new, k, groups=n_new)` (DW), `point_conv` 를
     `Conv2d(n_new, n_new, 1)` (입력+출력 양방향) 으로 교체.
   - ResBlock (L-series): `conv1`, `conv2` 모두 `Conv2d(n_new, n_new, k)` (양방향) 로 교체.
4. **다음 stage 첫 down-sampling Conv** (B 의 `inverted_conv` / L 의 `spatial_conv`)
   의 입력 컬럼만 `weight[:, survived]` 로 축소. 출력 / norm 은 그대로 (외부 채널).

> 같은 helper (`_reduce_convlayer_out`, `_reduce_convlayer_inout`) 가
> 후속 그룹 (head 등) 확장 시에도 그대로 재사용되도록 분리되어 있다.

### 6.3 ConvLayer 의 `.conv` / `.norm` 만 in-place 교체하는 이유

- `ConvLayer.forward` 는 `dropout → conv → norm → act` 순서로 호출되므로, 두 속성만 갈아끼우면 forward 시그니처가 보존된다.
- 부모 블록(`MBConv`, `FusedMBConv`) 의 `forward` 는 손대지 않아도 채널 변화가 자연스럽게 흘러간다.

### 6.4 CLI

```
python -m efficientvit.clscore.pruning.efficientvit_reducing \
    --model      efficientvit-b1 \
    --checkpoint /path/to/run/checkpoint/model_best.pt \
    --output     /path/to/reduced_b1.pt
```

또는 application 진입점:

```
python applications/efficientvit_cls/reduce_efficientvit_cls_model.py \
    --model efficientvit-b1 \
    --checkpoint ... \
    --output ...
```

CLI 동작:
1. zoo 로 원본 아키텍처 모델 생성 (pretrained=False).
2. `state_dict` (또는 `ema.shadows`) 를 로드. 외부 DDP `module.` prefix 자동 strip.
3. 변환 전후 파라미터 수 출력 + 압축률 계산.
4. `torch.zeros(1, 3, 224, 224)` forward 검증.
5. `state_dict` (기본) 또는 `--save-full-model` 시 모델 객체 자체를 저장.

---

## 7. 학습 루프와의 통합 — "최소 변경" 설계

### 7.1 Hook 위치

`efficientvit/apps/trainer/base.py:after_step()` 에만 4 줄 hook 을 추가했다.

```python
def after_step(self) -> None:
    self.scaler.unscale_(self.optimizer)
    if self.run_config.grad_clip is not None:
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.run_config.grad_clip)
    self.scaler.step(self.optimizer)
    self.scaler.update()

    # ★ Soft Pruning hook ★
    pruner = getattr(self, "pruner", None)
    if pruner is not None:
        pruner.apply(self.network)

    self.lr_scheduler.step()
    self.run_config.step()
    if self.ema is not None:
        self.ema.step(self.network, self.run_config.global_step)
```

### 7.2 왜 이 위치인가 — 방법론 도큐먼트 그대로

```
optimizer.step  →  scaler.update  →  ★ pruner.apply ★  →  ema.step  →  next forward
                                       ↑                ↑
                                       gradient 반영 후     EMA 가 pruning 후 weight 추적
```

- **`scaler.update()` 이후**: AMP unscale + optimizer.step 가 모두 끝나 weight 가 최신 상태일 때 마스킹.
- **`ema.step()` 이전**: EMA shadow 가 prune 된 weight 분포를 추적하도록 (권장).
- **`lr_scheduler.step()` 이전이든 이후든 무관**: lr 은 weight 와 독립적으로 동작.

### 7.3 `getattr(self, "pruner", None)` — 옵셔널 디자인

base trainer 자체는 `pruner` 라는 속성을 모른다. **`ClsTrainer` 만 생성자에서 `self.pruner = pruner` 로 주입**하므로, 다른 trainer (Seg/SAM/AE 등) 가 base 를 상속해도 영향이 0 이다.

### 7.4 `ClsTrainer` 변경

```python
class ClsTrainer(Trainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider,
        auto_restart_thresh: Optional[float] = None,
        pruner: Optional[Any] = None,                 # ← 추가
    ) -> None:
        super().__init__(path=path, model=model, data_provider=data_provider)
        self.auto_restart_thresh = auto_restart_thresh
        self.pruner = pruner                          # ← 추가
        self.test_criterion = nn.CrossEntropyLoss()
```

### 7.5 학습 진입점 변경 (`train_efficientvit_cls_model.py`)

추가된 argparse:
```python
parser.add_argument("--target_compression", type=float, default=0.0,
                    help="...0.0 disables pruning, 0.3 = remove ~30%.")
parser.add_argument("--pruning_max_sparsity", type=float, default=0.95)
```

추가된 분기:
```python
if args.target_compression > 0:
    pruner = EfficientViTPruner(
        model,
        target_compression=args.target_compression,
        max_sparsity=args.pruning_max_sparsity,
    )
else:
    pruner = None

trainer = ClsTrainer(..., pruner=pruner)
```

→ **`--target_compression 0` (기본값) 인 경우 모든 동작이 기존과 100% 동일.**

---

## 8. Sparsity 이진탐색 — Secondary Effect 처리

### 8.1 문제 (방법론 §5)

단순 선형 계산:
```python
# WRONG
sparsity = target_compression * total_params / prunable_params
```
은 다음 secondary effect 를 무시한다:
- DWConv (groups=mid) 가 mid 와 함께 자동으로 줄어듦.
- shrink layer 의 입력 컬럼이 expand 출력과 함께 줄어듦.

→ 결과적으로 **선형 계산은 압축률을 과소 추정 → 실제로는 더 많이 잘림.**

### 8.2 본 구현의 정확한 추정식

`_estimate_removed_mbconv` (sparsity=s 일 때 한 MBConv 에서 사라지는 파라미터 수):

```
n_prune = round(mid * s) − (만일 mid - n_prune < MIN_SURVIVE 이면 MIN_SURVIVE 보장)

removed
   += n_prune × in_ch                                  # inverted_conv weight
   += n_prune                       (bias 있을 때)      # inverted_conv bias
   += n_prune × 2                   (BN 있을 때)        # inverted_conv BN

   += n_prune × k × k                                   # depth_conv weight (groups=mid)
   += n_prune                       (bias 있을 때)      # depth_conv bias
   += n_prune × 2                   (BN 있을 때)        # depth_conv BN

   += out_ch × n_prune                                  # point_conv input cols
```

`_estimate_removed_fusedmbconv` 은 spatial_conv weight 가 `(mid, in/groups, k, k)` 인 점만 다르고 동일 패턴.

### 8.3 이진탐색

```python
def _find_sparsity_by_bisection(model, target_compression, max_sparsity=0.95, iters=64):
    target_remove = target_compression * total_params
    lo, hi = 0.0, max_sparsity
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
```

64 회 → 약 `1e-19` 정밀도. 충분히 일찍 수렴.

### 8.4 `input_stem` 의 이차/연쇄 효과

stem chain 은 기본 `n_prune * (in_ch_image · k²)` (첫 Conv) 외에 다음을 포함한다:

| 항목 | 추정식 |
|------|-------|
| stem[0] BN | `n_prune × 2` (BN 있을 때) |
| 각 DSConv `depth_conv` (groups=C0) | `n_prune × k²` weight + BN `n_prune × 2` |
| 각 DSConv `point_conv` (in=out=C0, 양방향) | `C0² − (C0 − n_prune)²` weight + BN `n_prune × 2` |
| 각 ResBlock `conv1` / `conv2` (양방향) | 위와 같은 `(C0² − (C0 − n_prune)²) × k²` 패턴 |
| 다음 stage 첫 conv 입력 컬럼 (cascade) | `n_prune × out_ch_first × k²` |

`(C0 − n_prune)²` 항은 **이차 효과** 로, 단순 선형 추정 (`n_prune × C0`) 대비
실제 제거량을 정확히 반영한다 (예제의 PE2/PE3 cascade 와 같은 형식).

cascade 마지막 항 (`n_prune × out_ch_first × k²`) 은 다음 stage 첫 conv 의 mid
pruning 과 **약간 중복** 되는데 (cross-term `n_prune_C0 × n_prune_mid`),
이는 추정량이 살짝 큰 쪽 (=실제 압축률이 target 보다 약간 높게 잡히는 안전 방향)
으로 작용하므로 그대로 둔다.

---

## 9. 실행 명령어 가이드

> **기준 모델**: `efficientvit-b1` (config: [`efficientvit_b1.yaml`](applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml))
> **데이터**: ImageNet (서버 절대경로 예: `/workspace/etri_iitp/JS/EfficientViT/data/imagenet`)
> **사전학습 weight**: `efficientvit-b1-r224` zoo 항목 (`assets/checkpoints/efficientvit_cls/efficientvit_b1_r224.pt`)

YAML 의 데이터 경로는 CLI override 로 바꿀 수 있다 (우리 train script 의
`parse_unknown_args` 가 dict 경로 문법을 그대로 받는다):

```bash
# YAML 의 data_provider.data_dir 를 CLI 에서 바꾸기 (반드시 공백으로 분리, = 사용 금지)
... --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

> **GPU 설정 주의사항**
> - `CUDA_VISIBLE_DEVICES` 에 나열한 GPU 수와 `--nproc_per_node` 값을 **반드시 일치**시켜야 한다.
> - 예: GPU 4번 1장 → `CUDA_VISIBLE_DEVICES=4 ... --nproc_per_node=1`
> - 예: GPU 4,5,6,7 4장 → `CUDA_VISIBLE_DEVICES=4,5,6,7 ... --nproc_per_node=4`

### 9.1 일반 학습 (pruning 비활성, 변동 없음)

GPU 1장 (GPU 4번) 기준:

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/etri_iitp/JS/EfficientViT/output/b1_baseline \
    --amp bf16 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_baseline \
    --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

GPU 여러 장 (예: 4,5,6,7 — 4장) 기준:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/etri_iitp/JS/EfficientViT/output/b1_baseline \
    --amp bf16 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_baseline_4gpu \
    --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

### 9.2 Soft Pruning 학습 (B1, target=30%)

GPU 1장 (GPU 4번) 기준:

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/etri_iitp/JS/EfficientViT/output/b1_prune30 \
    --amp bf16 \
    --target_compression 0.30 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_prune30 \
    --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

GPU 여러 장 (예: 4,5,6,7 — 4장) 기준:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/etri_iitp/JS/EfficientViT/output/b1_prune30 \
    --amp bf16 \
    --target_compression 0.30 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_prune30_4gpu \
    --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

다른 target 예 (50% / 75%):

```bash
# 50%
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/etri_iitp/JS/EfficientViT/output/b1_prune50 \
    --amp bf16 \
    --target_compression 0.50 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_prune50 \
    --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet

# 75%
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/etri_iitp/JS/EfficientViT/output/b1_prune75 \
    --amp bf16 \
    --target_compression 0.75 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_prune75 \
    --data_provider.data_dir /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

권장:
- 사전학습 weight 에서 출발 → fine-tune 형태가 가장 안정적. 본 구현은 zoo 의
  `pretrained=False` 로 모델을 만든 뒤 학습을 진행하므로, 사전학습 가중치를
  쓰려면 `setup.init_model` 의 `init_from` 에 weight 경로를 yaml 로 지정하거나,
  `--rand_init` 후 별도 finetune 단계를 둘 수 있다.
- 처음 1~2 epoch 후 `pruner.log_sparsity(network)` 로 실제 zero 비율을 한 번 확인.

### 9.2-W. Wandb 사전 준비 및 로깅 항목 상세

**사전 준비 (서버에서 1회):**
```bash
pip install wandb
wandb login   # 프롬프트에 API 키 입력
# 또는 환경변수로: export WANDB_API_KEY=<your_key>
```

`--wandb` 플래그를 빼면 wandb 없이 그대로 실행된다 (기본값).

**로깅되는 항목:**

| wandb 키 | 내용 |
|----------|------|
| `train/loss` | epoch 평균 학습 손실 |
| `train/top1` | epoch 평균 학습 Top-1 정확도 |
| `train/lr` | 현재 learning rate |
| `val/top1` | 검증 Top-1 정확도 |
| `val/top5` | 검증 Top-5 정확도 (ImageNet 기준 포함) |
| `val/loss` | 검증 손실 |
| `val/top1_best` | 현재까지 최고 검증 Top-1 |
| `pruning/<group>_zero_ratio` | 각 그룹의 실제 0 필터 비율 (pruner 활성 시에만) |

gradient 추적: `wandb.watch(model, log='gradients', log_freq=500)` 활성화되어 있어 500 step 마다 gradient histogram 이 기록된다.

> 전체 실행 명령어는 §9.1 / §9.2 참고. `--wandb_run_name` 은 실험마다 다르게 지정해 같은 project 아래 비교가 용이하다.

### 9.3 학습 후 Reducing (B1, 30% 예시)

```bash
python applications/efficientvit_cls/reduce_efficientvit_cls_model.py \
    --model efficientvit-b1 \
    --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/b1_prune30/checkpoint/model_best.pt \
    --output    /workspace/etri_iitp/JS/EfficientViT/output/b1_prune30/reduced_b1_30pct.pt
```

옵션:
- `--save-full-model`: state_dict 대신 모델 객체 자체 저장 (`torch.save(model)`).
- `--input-size 224`, `--n-classes 1000`: forward 검증용 (B1 기본).

### 9.4 메모리 분해 측정 (`measure_memory.py`)

```bash
# 1) 원본 B1 모델만 — 컴포넌트별 분해 + per-stage 분해
python applications/efficientvit_cls/measure_memory.py \
    --model efficientvit-b1 \
    --per-stage

# 2) 사전학습 weight 로드 후
python applications/efficientvit_cls/measure_memory.py \
    --model efficientvit-b1 \
    --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/b1_baseline/checkpoint/model_best.pt

# 3) 원본 vs Reduced 비교 (이미 reducing 된 결과 파일이 있을 때)
python applications/efficientvit_cls/measure_memory.py \
    --model efficientvit-b1 \
    --reduced /workspace/etri_iitp/JS/EfficientViT/output/b1_prune30/reduced_b1_30pct.pt \
    --per-stage

# 4) Soft-pruned ckpt 에서 즉석 reducing 후 분해 (별도 reduced 파일 없이)
python applications/efficientvit_cls/measure_memory.py \
    --model efficientvit-b1 \
    --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/b1_prune30/checkpoint/model_best.pt \
    --auto-reduce \
    --per-stage
```

출력 예시 (스키마):

```
=== [model] component-wise parameter memory ===
group       #mod         numel          MB        %
----------------------------------------------------
G_STEM         2         X,XXX       0.XXX    XX.XX%
G_HEAD         1         X,XXX       0.XXX    XX.XX%
G_LITEMLA      n         X,XXX       0.XXX    XX.XX%
G_MBCONV       n         X,XXX       0.XXX    XX.XX%
G_FUSEDMB      n         X,XXX       0.XXX    XX.XX%
G_OTHER        -         X,XXX       0.XXX    XX.XX%
----------------------------------------------------
TOTAL                  X,XXX,XXX     X.XXX   100.00%

=== compression breakdown (Original → Reduced) ===
group        orig MB     red MB        ΔMB   compress
----------------------------------------------------------
G_STEM       0.XXX       0.XXX        0.XXX     XX.XX%
G_MBCONV     X.XXX       X.XXX        X.XXX     XX.XX%
...
TOTAL        X.XXX       X.XXX        X.XXX     XX.XX%
```

> 측정 단위는 **bytes** 기반 (`numel × element_size`) 이며, 학습 weight + BN
> running stats (buffers) 까지 모두 포함한다. fp32 기준 1 param ≈ 4 bytes.
> bf16/fp16 모델일 경우 자동으로 element_size 가 2 로 잡혀 일관된 결과.

---

## 10. 의사결정 / 트레이드오프 정리

| 결정 | 채택 | 대안 | 이유 |
|------|------|------|------|
| Pruning hook 위치 | base trainer `after_step()` 4 줄 추가 | ClsTrainer 에서 `after_step` 오버라이드 | EMA 갱신이 base 안에 있어 오버라이드만으로 "EMA 이전" 위치를 잡기 까다로움. 4 줄 추가가 가장 깔끔. |
| Pruner 주입 방식 | `ClsTrainer.__init__` 에 `pruner=None` 인자 | trainer 의 attribute 를 외부에서 monkey-patch | API 명시성 ↑. 다른 trainer 영향 0. |
| Prunable 그룹 범위 | MBConv / FusedMBConv mid + **input_stem chain** | + LiteMLA Q/K, ClsHead | "최소 변경 + 정확도 보존" 우선. 1차 적용에서 input_stem 까지는 포함 (예제의 `G_PE` 와 동등). LiteMLA 와 ClsHead 는 후속 확장. (§12) |
| input_stem chain idx 산정 기준 | `op_list[0].conv` 출력 필터 L2 norm | 모든 inner block 출력 필터 평균 / 결합 norm | inner block 들이 동일 채널 공간을 공유하므로 단일 layer 만으로 ranking 해도 충분히 일관됨. 예제 PE chain 도 각 PE 의 자체 출력 norm 을 사용. |
| 인덱스 산정 기준 | `inverted_conv` (또는 `spatial_conv`) 의 출력 필터 L2 norm | `point_conv` 입력 컬럼 / 둘의 평균 | 표준 관행 + DWConv groups 가 입력단(=expand 출력) 에 종속이라 그쪽이 자연. |
| `MIN_SURVIVE` | 4 | 0 / 1 / `max(4, 0.05·n)` | 방법론 도큐먼트 권장값. 너무 작으면 정보 손실, 너무 크면 압축률 저하. |
| `max_sparsity` | 0.95 | 1.0 / 0.90 | 1.0 은 그룹 통째 사라짐 위험. 0.90 은 큰 hidden 에서 보수적. 0.95 가 균형. |
| BN running_var | 1.0 | 0.0 | 0.0 이면 `(x − 0)/sqrt(0 + eps)` 로 수치 불안정. 1.0 이 안전. |
| Reducing 정확 임계 | `norm != 0` (정확 0) | `norm < 1e-6` | Soft Pruning 이 정확히 0 으로 마스킹하므로 `!= 0` 가 더 정확. EMA shadow 가 아닌 raw `state_dict` 를 reducing 대상으로 사용. |
| Reducing 출력 형식 | 기본 state_dict | 기본 full model | state_dict 가 작고 호환성 ↑. `--save-full-model` 로 옵션. |

---

## 11. 메서드 별 줄 단위 변경 diff

### 11.1 `efficientvit/apps/trainer/base.py`

```diff
 def after_step(self) -> None:
     self.scaler.unscale_(self.optimizer)
     # gradient clip
     if self.run_config.grad_clip is not None:
         torch.nn.utils.clip_grad_value_(self.model.parameters(), self.run_config.grad_clip)
     # update
     self.scaler.step(self.optimizer)
     self.scaler.update()

+    # Soft Pruning: optimizer.step() 직후, EMA 업데이트 이전에 마스킹.
+    # pruner 가 attach 되어 있지 않으면 no-op.
+    pruner = getattr(self, "pruner", None)
+    if pruner is not None:
+        pruner.apply(self.network)
+
     self.lr_scheduler.step()
     self.run_config.step()
     # update ema
     if self.ema is not None:
         self.ema.step(self.network, self.run_config.global_step)
```

### 11.2 `efficientvit/clscore/trainer/cls_trainer.py`

```diff
 class ClsTrainer(Trainer):
     def __init__(
         self,
         path: str,
         model: nn.Module,
         data_provider,
         auto_restart_thresh: Optional[float] = None,
+        pruner: Optional[Any] = None,
     ) -> None:
         super().__init__(
             path=path,
             model=model,
             data_provider=data_provider,
         )
         self.auto_restart_thresh = auto_restart_thresh
+        # Soft Pruning 컨트롤러 (None 이면 base.after_step 의 hook 이 no-op).
+        self.pruner = pruner
         self.test_criterion = nn.CrossEntropyLoss()
```

### 11.3 `applications/efficientvit_cls/train_efficientvit_cls_model.py`

```diff
 from efficientvit.apps import setup
 from efficientvit.apps.utils import dump_config, parse_unknown_args
 from efficientvit.cls_model_zoo import create_efficientvit_cls_model
 from efficientvit.clscore.data_provider import ImageNetDataProvider
+from efficientvit.clscore.pruning import EfficientViTPruner
 from efficientvit.clscore.trainer import ClsRunConfig, ClsTrainer
 from efficientvit.models.nn.drop import apply_drop_func
 ...
 parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
 parser.add_argument("--save_freq", type=int, default=1)

+# Soft Pruning options (optional, opt-in).
+parser.add_argument("--target_compression", type=float, default=0.0,
+                    help="target parameter compression rate (0.0 disables pruning, e.g. 0.3 = remove ~30%).")
+parser.add_argument("--pruning_max_sparsity", type=float, default=0.95,
+                    help="upper bound for per-group sparsity used in bisection.")
 ...
     # setup model
     model = create_efficientvit_cls_model(...)
     apply_drop_func(model.backbone.stages, config["backbone_drop"])

+    # setup pruner (opt-in via --target_compression > 0).
+    if args.target_compression > 0:
+        pruner = EfficientViTPruner(
+            model,
+            target_compression=args.target_compression,
+            max_sparsity=args.pruning_max_sparsity,
+        )
+    else:
+        pruner = None
+
     # setup trainer
     trainer = ClsTrainer(
         path=args.path,
         model=model,
         data_provider=data_provider,
         auto_restart_thresh=args.auto_restart_thresh,
+        pruner=pruner,
     )
```

### 11.4 신규 파일 (요약)

| 파일 | 역할 | 외부 노출 |
|------|------|----------|
| `efficientvit/clscore/pruning/__init__.py` | 패키지 초기화 + re-export | `EfficientViTPruner`, `reduce_efficientvit_cls_model` |
| `efficientvit/clscore/pruning/efficientvit_pruning.py` | Soft Pruning 본체 (이진탐색, 마스킹, sparsity 로깅, input_stem chain) | `EfficientViTPruner` |
| `efficientvit/clscore/pruning/efficientvit_reducing.py` | Dense 변환 (MBConv / FusedMBConv / input_stem chain) + CLI | `reduce_efficientvit_cls_model`, `main` |
| `applications/efficientvit_cls/reduce_efficientvit_cls_model.py` | Reducing 사용자 진입점 | `python ... reduce_...py` |
| `applications/efficientvit_cls/measure_memory.py` | 컴포넌트별 파라미터 메모리 분해 + 원본/Reduced 비교 | `python ... measure_memory.py` |

---

## 12. 향후 확장 가능 그룹

다음은 본 구현에서 **의도적으로 제외** 했지만 추가 가능한 그룹들이다.
(input_stem chain 은 본 업데이트에서 이미 포함됨 — §4.5)

### 12.1 `ClsHead` (Conv → Linear → Linear)

```
head.op_list[0]  ConvLayer(in_ch → width_list[0])      # output prunable
head.op_list[2]  LinearLayer(width_list[0] → width_list[1])   # input/output 모두 prunable (output은 head[3]과 coupled)
head.op_list[3]  LinearLayer(width_list[1] → n_classes)        # output 고정 (n_classes)
```

추가 그룹:
- **G_HEAD0**: `head.op_list[0].conv.weight[idx]` + BN[idx] + `head.op_list[2].linear.weight[:, idx]`
- **G_HEAD1**: `head.op_list[2].linear.weight[idx]` + LN(`head.op_list[2].norm`)[idx] + `head.op_list[3].linear.weight[:, idx]`

`B1` 기준 head 가 전체 파라미터의 ~30% 를 차지하므로, 추가 압축이 필요하면 우선순위 후보.

### 12.2 `LiteMLA` Q / K (multi-scale 인지하면서)

- `qkv` 텐서가 (Q, K, V) 인터리브되어 있어 Q와 K 만 별도 인덱스로 자르려면 텐서 슬라이싱이 까다롭다.
- 또한 `aggreg` 의 두 conv (DW + 1×1) 가 `3 × total_dim` 채널 전체를 그대로 유지해야 하므로, Q/K 인덱스를 잘라도 aggreg 출력에 그대로 반영시켜야 한다.
- proj 입력은 `total_dim · (1 + len(scales))` 로 두 배 이상 확장되므로, secondary effect 추정식이 한 단계 더 복잡해진다.
- 현재 구현은 LiteMLA 를 건드리지 않으므로 안전하다 — 추가 시 Q/K head_dim 축으로의 동일 인덱스 강제, aggreg conv 의 동일 인덱스 슬라이스 처리가 필요.

### 12.3 `GLUMBConv`

`EfficientViTBlock` 의 `local_module="GLUMBConv"` 일 때 사용되는 변형. inverted_conv 가
`mid_channels * 2` 를 출력해 chunk 로 데이터/게이트 분리 → element-wise 곱 → point_conv.
mid 를 줄이려면 inverted_conv 출력의 `[0:mid]` (data) 와 `[mid:2*mid]` (gate) 가
**동일 인덱스로 짝지어 잘려야** 한다. 현재 구현은 GLUMBConv 를 별도 처리하지 않으므로
default config 에서는 영향 없으나, `local_module=GLUMBConv` 사용 시 분기 추가 필요.

---

## 13. 체크리스트 (방법론 §9 대조)

| 단계 | 항목 | 상태 |
|------|------|------|
| Step 1 | 모든 레이어 / 채널 의존성 분석 | ✅ §3, §4 |
| Step 1 | Prunable 그룹 / 비-Prunable 분류 | ✅ §4 |
| Step 1 | Coupled 관계 명세 | ✅ §4.3 |
| Step 2 | `estimate_total_removed` 구현 | ✅ §8.2 |
| Step 2 | Secondary effect (DWConv groups, point_conv 입력) 반영 | ✅ |
| Step 3 | `get_pruning_idx` (L2 norm 하위 topk) | ✅ `_topk_smallest_l2_idx` |
| Step 3 | 그룹별 `_prune_*` 구현 | ✅ `_prune_mbconv`, `_prune_fusedmbconv` |
| Step 3 | Coupled 인덱스 일관성 | ✅ 단일 idx 를 inverted/depth/point 에 모두 적용 |
| Step 3 | BN 마스킹 (running_var=1.0 포함) | ✅ `_zero_bn_` |
| Step 4 | `EfficientViTPruner.__init__` 이진탐색 + 로그 | ✅ |
| Step 4 | `apply(model)` | ✅ |
| Step 4 | `log_sparsity(model)` | ✅ |
| Step 5 | 학습 루프에 `pruner=None` 옵션 | ✅ §7.4 |
| Step 5 | optimizer.step 직후 hook | ✅ §7.1 |
| Step 5 | DDP 호환 (`self.network` 가 unwrap 처리) | ✅ — `Trainer.network` 는 `model.module if is_parallel ... else model`. |
| Step 6 | `get_survived_idx` | ✅ `_survived_idx` |
| Step 6 | 그룹별 `_reduce_*` in-place 구현 | ✅ |
| Step 6 | DWConv groups 동기화 | ✅ `groups=n_new` |
| Step 6 | Forward 검증 | ✅ CLI 안에 `torch.zeros(1, 3, H, W)` 호출 |
| Step 6 | 압축률 검증 | ✅ CLI 출력 |
| Step 7 | sparsity 로그 (학습 중 검증용) | ✅ `pruner.log_sparsity` |
| Step 7 | Reducing 후 forward OK | ✅ |
| Step 7 | Reducing 후 압축률 ≥ target | △ — 본 구현은 추정값 기반 sparsity 를 한 번에 결정. 학습 종료 후 실제 zero 비율이 추정과 약간 어긋날 수 있다 (소규모 그룹의 round 차이). |

---

*작성 일자: 2026-04-27. 최종 수정: 2026-04-29. 기준 코드: EfficientViT classification 파이프라인 (master, commit 53e8795 시점).*

---

## 14. 업데이트 이력 (Changelog)

### 2026-04-29 (rev 4) — Wandb 연동

**추가**
- `efficientvit/clscore/trainer/cls_trainer.py`:
  - `ClsTrainer.__init__`에 `wandb_project`, `wandb_run_name` 인자 추가.
  - `train()` 시작 시 `wandb.init(project=..., name=..., config=...)` + `wandb.watch(model, log='gradients', log_freq=500)`.
  - 에폭 루프 종료 시 `wandb.log(train/loss, train/top1, train/lr, val/top1, val/top5, val/loss, val/top1_best, pruning/*)`.
  - 학습 완료 시 `wandb.finish()`. 에러 시 graceful fallback (로그만 출력하고 학습 계속).
- `applications/efficientvit_cls/train_efficientvit_cls_model.py`:
  - `--wandb` (flag), `--wandb_project` (str), `--wandb_run_name` (str) argparse 인자 추가.
  - `ClsTrainer` 생성 시 `wandb_project` / `wandb_run_name` 전달.
- 보고서 §9.2-W: wandb 연동 실행 명령어 및 로깅 항목 표 추가.

---

### 2026-04-29 (rev 3) — GPU 설정 및 명령어 형식 수정

**수정**
- §9 전 구간 명령어에 `CUDA_VISIBLE_DEVICES=<GPU_ID>` 추가.
- `--nproc_per_node` 를 `CUDA_VISIBLE_DEVICES` 에 기재한 GPU 수와 일치시킴
  (예: GPU 1장 → `--nproc_per_node=1`, 4장 → `--nproc_per_node=4`).
- `--data_provider.data_dir=...` (`=` 연결) → `--data_provider.data_dir ...` (공백 분리) 수정.
  `parse_unknown_args` 가 `key val` 두 토큰 형식만 처리하므로 `=` 사용 시 `IndexError` 발생.
- §9.1, §9.2 에 단일 GPU (GPU 4번) 기준과 다중 GPU (4,5,6,7) 기준 명령어를 각각 명시.
- §9.2 의 50% / 75% 예시를 완전한 명령어로 확장.
- §9 서두에 GPU 설정 주의사항 노트 추가.

---

### 2026-04-27 (rev 2) — 예제 대조 + input_stem chain + 메모리 측정 도구

`EfficientViT- Example/` (M-series 레퍼런스 구현) 을 검토하여 다음을 보강.

**추가**
- `_prune_input_stem`, `_estimate_removed_input_stem` 함수 (efficientvit_pruning.py).
  - B-series `input_stem` 및 L-series `stages[0]` 둘 다 자동 처리.
  - 첫 ConvLayer 의 출력 필터 L2 norm 으로 single index 산정 → chain 전체 + 다음
    stage 첫 conv 입력 컬럼에 동기 마스킹.
- `_reduce_input_stem`, `_reduce_convlayer_out`, `_reduce_convlayer_inout` (efficientvit_reducing.py).
  - DSConv groups=n_new 처리, point_conv 양방향 축소, 다음 stage 첫 conv 입력 컬럼 축소.
- `applications/efficientvit_cls/measure_memory.py`:
  - 그룹별 (G_STEM / G_HEAD / G_LITEMLA / G_MBCONV / G_FUSEDMB / G_OTHER) 파라미터 메모리 분해.
  - per-stage breakdown (`--per-stage`).
  - 원본 vs Reduced 비교 (`--reduced`, `--auto-reduce`).

**수정**
- `EfficientViTPruner.apply()`: stem 마스킹 호출 추가.
- `EfficientViTPruner.log_sparsity()`: stem 출력 채널 zero 비율 함께 집계.
- `_estimate_total_removed`: stem 제거량 합산.
- 보고서: 매핑 표 (§4.4), input_stem chain 상세 (§4.5), stem 마스킹/축소 흐름
  (§5.2.1, §6.2.1), stem 이차 효과 (§8.4), B1 + ImageNet 명령어 (§9), 메모리 측정
  사용법 (§9.4) 섹션 추가.

**예제 코드 대조 결과 (변경 없이 일치 확인)**
- BN running_var = 1.0 (분모 0 방지) ✓
- MIN_SURVIVE = 4 ✓
- `round()` 사용 (소규모 레이어 절삭 회피) ✓
- 이진탐색 64 회 ≈ 1e-19 정밀도 ✓
- pruner hook 위치 = `optimizer.step()` 직후, EMA `update()` 이전 ✓
- DDP 환경에서 `self.network` (= `model.module if parallel`) 전달 ✓
- 인덱스 산정 = expand 출력 필터 L2 norm 의 하위 topk ✓
- coupled 인덱스 동기화 (expand-shrink, DW groups, chain) ✓
