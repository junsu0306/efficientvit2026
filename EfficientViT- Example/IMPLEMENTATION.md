# EfficientViT M4 Soft Pruning — 구현 문서

> **프로젝트**: RS-2024-00339187 | 고려대학교 | 3차년도 ViT 확장 연구
> **목표**: EfficientViT M4에 Soft Pruning 적용 → 압축률 30% / 50% / 75% 달성
> **압축률 공식**: `100 × (B - A) / B`  (B: 원본 파라미터 수, A: 압축 후 파라미터 수)

---

## 목차

1. [M4 파라미터 실측 분석](#1-m4-파라미터-실측-분석)
2. [모델 구조 분석](#2-모델-구조-분석)
3. [Pruning 방법론](#3-pruning-방법론)
4. [압축률 ↔ Sparsity 변환 공식](#4-압축률--sparsity-변환-공식)
5. [소규모 레이어 처리 (_MIN_SURVIVE)](#5-소규모-레이어-처리-_min_survive)
6. [파일별 구현 설명](#6-파일별-구현-설명)
7. [수정된 기존 파일](#7-수정된-기존-파일)
8. [학습 실행 명령어](#8-학습-실행-명령어)
9. [Reducing (학습 완료 후)](#9-reducing-학습-완료-후)
10. [달성 가능 압축률 분석](#10-달성-가능-압축률-분석)
11. [핵심 제약사항](#11-핵심-제약사항)
12. [개발 환경 참고사항](#12-개발-환경-참고사항)
13. [전체 압축 파이프라인 — 상세 구현 원리](#13-전체-압축-파이프라인--상세-구현-원리)
14. [파일 구조](#14-파일-구조)
15. [Backbone과 Head 개념](#15-backbone과-head-개념)
16. [10-Class 서브셋 훈련 — 실행 방법](#16-10-class-서브셋-훈련--실행-방법)

---

## 1. M4 파라미터 실측 분석

GPU 서버에서 실측한 EfficientViT-M4 파라미터 분포 (총 **8,804,228** params = 35.22 MB):

| 그룹 | 대상 | 파라미터 수 | 비율 | MB | Pruning |
|------|------|------------|------|----|---------|
| **G_FFN** | FFN expand+shrink (전체 블록) | 5,925,888 | 67.3% | 23.7 | ✅ |
| **G_QK** | CGA Q+K proj + Q DWConv (전체 head) | 68,480 | 0.8% | 0.27 | ✅ |
| **G_INV** | Subsample 1×1 expand+reduce (PatchMerging conv1+conv3) | 856,320 | 9.7% | 3.43 | ✅ |
| **G_PE1** | PatchEmbed Conv1 (3 → C/8 = 16ch) | 464 | 0.0% | 0.0 | ✅ |
| **G_PE2** | PatchEmbed Conv2 (C/8 → C/4 = 32ch) | 4,672 | 0.1% | 0.02 | ✅ |
| **G_PE3** | PatchEmbed Conv3 (C/4 → C/2 = 64ch) | 18,560 | 0.2% | 0.07 | ✅ |
| **G_V** | CGA V projection (전체 head) | 151,040 | 1.7% | 0.6 | ❌ |
| **W_out** | CGA Output Projection (W'') | 593,408 | 6.7% | 2.37 | ❌ |
| **Attn_Bias** | CGA 학습 가능 위치 편향 | 780 | 0.0% | 0.0 | ❌ |
| **PM_DW** | PatchMerging DWConv (stride=2) | 16,896 | 0.2% | 0.07 | ❌ |
| **PM_SE** | PatchMerging SqueezeExcite | 657,280 | 7.5% | 2.63 | ❌ (reducing 시 연동) |
| **G_PE4** | PatchEmbed Conv4 (C/2 → C = 128ch) | 73,984 | 0.8% | 0.3 | ❌ |
| **DWConv** | Token Interaction DWConv (dw0+dw1) | 50,688 | 0.6% | 0.2 | ❌ |
| **Head** | Classifier (BN + Linear) | 385,768 | 4.4% | 1.54 | ❌ |
| **합계** | | **8,804,228** | 100.0% | 35.22 | |

### 그룹별 역할 설명

| 그룹 | 역할 |
|------|------|
| **G_FFN** | 각 EfficientViTBlock 내 FFN(pw1=expand, pw2=shrink) + SubDWFFN. 모델 파라미터의 67%를 차지하는 핵심 타겟 |
| **G_QK** | Cascaded Group Attention(CGA)의 Q, K 통합 projection(qkvs)과 Q에 적용되는 DWConv(dws). QK^T 연산에 사용 |
| **G_INV** | PatchMerging 내 1×1 inverted residual: conv1(expand, dim→4dim)과 conv3(reduce, 4dim→out_dim). Stage 간 해상도 다운샘플링 담당 |
| **G_PE1~3** | 입력 이미지를 token으로 변환하는 stem conv 1~3. 채널 수가 작아(16, 32, 64) 절대 파라미터 수는 적지만, 파라미터 비율 기준으로 함께 pruning |
| **G_V** | CGA V(Value) projection. V 채널 수는 output projection(W_out) 입력과 연결되어 있어 독립 pruning 시 차원 불일치 발생 → 제외 |
| **W_out** | CGA output projection. 모든 head의 V 출력을 합쳐 stage 채널로 변환 → 전체 채널 얼라인먼트 핵심, 절대 금지 |
| **PM_DW** | PatchMerging 내 DWConv(stride=2). 채널 수 감소 없이 공간 해상도만 2×2 다운샘플링 |
| **PM_SE** | PatchMerging 내 SqueezeExcite. Soft pruning 시 zeros 전파, reducing 시 G_INV reducing에 연동하여 함께 축소 |
| **G_PE4** | Stem 마지막 Conv(64→128). 출력이 blocks1 입력과 직결되어 out_channel=128 고정 |
| **DWConv** | EVBlock 내 token interaction DWConv(dw0, dw1). 제거 시 정확도 -1.4% 하락(실험 확인) |
| **Head** | BN + Linear 분류기. 출력 클래스 수(1000) 고정 |
| **Attn_Bias** | CGA 학습 가능 위치 편향(attention_biases). 크기 미미, 변경 불필요 |

### Prunable 파라미터 합계

```
G_FFN + G_QK + G_INV + G_PE1~3 = 5,925,888 + 68,480 + 856,320 + 23,696
                                 = 6,874,384  (전체의 78.1%)

Non-prunable                    = 1,929,844  (전체의 21.9%)
```

---

## 2. 모델 구조 분석

### 2.1 CLAUDE.md 예상 vs 실제 구조

`classification/model/efficientvit.py` 직접 확인 결과, CLAUDE.md 예상과 **중요한 차이** 발견:

| 항목 | CLAUDE.md 예상 | 실제 구조 |
|------|---------------|-----------|
| FFN expand | `Linear(C → C*r)` | `Conv2d_BN(ed, h, ks=1)` ← **1×1 Conv** |
| FFN shrink | `Linear(C*r → C)` | `Conv2d_BN(h, ed, ks=1)` ← **1×1 Conv** |
| Q projection | `head.q: Linear` | `qkvs[i].c: Conv2d` (Q+K+V 통합 후 split) |
| EVBlock FFN | `block.ffn` | `block.ffn0`, `block.ffn1` **(2개 존재)** |

**FFN은 Linear가 아닌 1×1 Conv2d** 로 구현됨.

### 2.2 M4 전체 블록 구조

```
model.patch_embed  ← Conv2d_BN × 4 (PE1~PE4, PE1-3만 pruning)
model.blocks1      ← [EVBlock(ed=128)]
model.blocks2      ← [SubDWFFN(ed=128), PatchMerging(128→256), SubDWFFN(ed=256),
                       EVBlock(ed=256), EVBlock(ed=256)]
model.blocks3      ← [SubDWFFN(ed=256), PatchMerging(256→384), SubDWFFN(ed=384),
                       EVBlock(ed=384), EVBlock(ed=384), EVBlock(ed=384)]
model.head         ← BN_Linear(384 → 1000)
```

### 2.3 EfficientViTBlock 내부 레이어 경로

```python
EfficientViTBlock
  ├── dw0  = Residual(Conv2d_BN)          # DWConv, 금지
  ├── ffn0 = Residual(FFN)   → .m = FFN   # G_FFN
  │    FFN.pw1 = Conv2d_BN(ed→h)  → .c (Conv2d), .bn (BN)
  │    FFN.pw2 = Conv2d_BN(h→ed)  → .c (Conv2d), .bn (BN)
  ├── mixer = Residual(LWA)  → .m.attn = CGA
  │    CGA.qkvs[i] = Conv2d_BN(ed//H → kd*2+d)  # G_QK
  │    CGA.dws[i]  = DWConv on Q                 # G_QK (reducing 시 연동)
  │    CGA.proj    = ReLU + Conv2d_BN             # W_out, 금지
  ├── dw1  = Residual(Conv2d_BN)          # DWConv, 금지
  └── ffn1 = Residual(FFN)   → .m = FFN   # G_FFN
```

### 2.4 PatchMerging 내부

```python
PatchMerging
  ├── conv1 = Conv2d_BN(dim, hid=4*dim, 1)  # G_INV expand
  ├── act   = ReLU
  ├── conv2 = Conv2d_BN(hid, hid, 3, stride=2, groups=hid)  # PM_DW (reducing 연동)
  ├── se    = SqueezeExcite(hid, 0.25)       # PM_SE (reducing 연동)
  └── conv3 = Conv2d_BN(hid, out_dim, 1)    # G_INV reduce
```

### 2.5 SubDWFFN 구조 (Sequential)

```python
torch.nn.Sequential
  [0]: Residual(Conv2d_BN)  # DWConv, 금지
  [1]: Residual(FFN)        # .m = FFN  → G_FFN
```

### 2.6 CGA Q/K/V 분리 방식

```python
feat = qkvs[i](feat)                              # Conv2d_BN 단일 projection
q, k, v = feat.split([key_dim, key_dim, d], dim=1)
#  Q: channels [0:key_dim]
#  K: channels [key_dim:2*key_dim]
#  V: channels [2*key_dim:]
q = dws[i](q)   # DWConv
```

M4 Q/K/V 크기:

| Stage | ed | H | ed/H | key_dim | d |
|-------|----|---|------|---------|---|
| 1 | 128 | 4 | 32 | 16 | 32 |
| 2 | 256 | 4 | 64 | 16 | 64 |
| 3 | 384 | 4 | 96 | 16 | 96 |

---

## 3. Pruning 방법론

### 3.1 Soft Pruning (학습 중)

```
매 optimizer.step() 이후:
  1. L2 norm 기준으로 하위 sparsity% 채널의 weight를 0으로 설정
  2. gradient로 살아난 값도 다음 step에서 다시 0으로 리셋
  3. 살아남은 채널만 최적화됨
  4. 학습 완료 후 reducing으로 물리적 제거
```

### 3.2 G_FFN Pruning

```
pw1(expand, ed→h):
  ① L2 norm = norm(weight[i, :, 0, 0]) for i in range(h)
  ② 하위 sparsity% → pruning_idx
  ③ pw1.c[pruning_idx] = 0, pw1.bn[pruning_idx] = 0

pw2(shrink, h→ed):
  ④ pw2.c[:, pruning_idx] = 0  ← 동일 인덱스 (연결 차원 보존)
```

### 3.3 G_QK Pruning

```
qkvs[i] 출력 레이아웃: [Q(0:kd) | K(kd:2kd) | V(2kd:)]
  ① Q portion(0:kd)의 L2 norm → q_idx
  ② Q zero: conv[q_idx] = 0
  ③ K zero: conv[q_idx + kd] = 0  ← 동일 상대 인덱스 (QK^T 차원 필수)
  ④ V, proj: 절대 미접근
```

### 3.4 G_PE1~3 Pruning (신규 추가)

```
PE chain: PE1 → PE2 → PE3 → PE4(입력만)
  ① PE1 출력 필터 pruning → PE2 입력 채널 zero (동일 idx)
  ② PE2 출력 필터 pruning → PE3 입력 채널 zero (독립 sparsity)
  ③ PE3 출력 필터 pruning → PE4 입력 채널 zero
  ④ PE4 출력(=embed_dim=128): 절대 금지 (blocks1 입력과 직결)
```

각 PE 레이어 독립적으로 sparsity 적용 (per-layer).

### 3.5 G_INV Pruning (신규 추가)

```
PatchMerging conv1 (expand, dim → 4dim):
  ① conv1 출력 필터 pruning
  ② conv3 입력 채널 zero (동일 idx)
  ③ conv2(DWConv), SE: zeros가 forward에서 자연 전파
     - conv2: DW → zero input → zero output
     - SE: global avg → zero 채널 → squeeze 기여 없음
  ④ conv3 출력(=out_dim): 절대 금지 (다음 stage 입력과 직결)
```

### 3.6 Pruning 금지 대상

| 레이어 | 이유 |
|--------|------|
| DWConv (dw0, dw1) | 제거 시 -1.4% 정확도 하락 (실험 확인) |
| CGA proj (W_out) | 전체 head 출력 채널 얼라인먼트 핵심 |
| CGA V projection | W_out 입력과 차원 연결 |
| G_PE4 출력 | blocks1 입력 채널(128) 고정 |
| PatchMerging conv3 출력 | 다음 stage 입력 채널 고정 |
| Head | 분류 출력 고정 |

---

## 4. 압축률 ↔ Sparsity 변환 공식

### 4.1 설계 원칙

> **파라미터 수에 비례하여 모든 prunable 그룹에 동일 sparsity 적용**

동일 sparsity를 적용하면 각 그룹의 절대 제거량이 자동으로 파라미터 수에 비례.
(FFN이 QK보다 85배 크므로, 같은 sparsity로도 85배 많은 파라미터 제거)

### 4.2 공식

```
sparsity = target_compression × total_params / prunable_total

  where prunable_total = G_FFN + G_QK + G_PE1~3 + G_INV
                       = 5,925,888 + 68,480 + 23,696 + 856,320
                       ≈ 6,874,384  (전체의 78.1%)
```

### 4.3 M4 압축률별 값 (G_INV + G_PE 추가 후)

| target | raw_sparsity | applied | 예상 실제 압축률 |
|--------|-------------|---------|----------------|
| 30%    | ≈ 0.38      | 0.38    | ≈ 30%          |
| 50%    | ≈ 0.64      | 0.64    | ≈ 50%          |
| 75%    | ≈ 0.96      | **0.95** (클리핑) | ≈ 74% |

> G_INV (9.7%) 추가로 prunable 비율이 68% → 78%로 증가.
> 75% 목표가 이전 대비 훨씬 가까워짐 (이전 ≈65% → 현재 ≈74%).

---

## 5. 소규모 레이어 처리 (_MIN_SURVIVE)

### 5.1 문제: int() 절삭으로 pruning 0 발생

```python
# PE1: 16채널, sparsity=0.10
int(16 * 0.10) = int(1.6) = 1   # OK
int(16 * 0.05) = int(0.8) = 0   # 영원히 pruning 안 됨!
```

### 5.2 해결책: round() + 최소 생존 보장

```python
_MIN_SURVIVE = 4   # 레이어당 최소 생존 채널 수

num_pruning = round(num_filters * sparsity)           # 반올림
num_pruning = min(num_pruning, num_filters - _MIN_SURVIVE)  # 최소 4채널 생존
num_pruning = max(0, num_pruning)
```

| 채널 수 | sparsity | int() | round() | 비고 |
|--------|---------|-------|---------|------|
| 16 (PE1) | 0.30 | 4 | 5 | round가 더 정확 |
| 16 (PE1) | 0.80 | 12 | **12** → min(12, 16-4)=12 | 4채널 생존 |
| 16 (PE1) | 0.95 | 15 | **15** → min(15, 12)=12 | 4채널 강제 생존 |

---

## 6. 파일별 구현 설명

### 6.1 `classification/pruning/efficientvit_pruning.py`

#### 함수 목록

| 함수 | 역할 |
|------|------|
| `_get_conv_pruning_idx(conv, sparsity)` | L2 norm 하위 인덱스 반환. round() + _MIN_SURVIVE 적용 |
| `_zero_out_filters(conv, bn, idx)` | 출력 필터 + BN zero |
| `_zero_in_channels(conv, idx)` | 입력 채널 zero |
| `_prune_ffn(ffn, sparsity)` | G_FFN: pw1→pw2 연동 |
| `_prune_cga_qk(cga, sparsity)` | G_QK: Q→K 동일 인덱스 |
| `_prune_patch_embed(model, sparsity)` | **신규** G_PE1-3: chain 연동 |
| `_prune_patch_merging(pm, sparsity)` | **신규** G_INV: conv1→conv3 연동 |
| `efficientvit_pruning(model, ...)` | 전체 순회 (G_INV, G_PE 포함) |
| `count_prunable_params(model)` | G_FFN/QK/PE/INV 별도 집계 |
| `EfficientViTPruner` | sparsity 역산 + apply() |

#### G_PE 구현 핵심

```python
def _prune_patch_embed(model, sparsity):
    pe = model.patch_embed  # Sequential: [0]=PE1, [2]=PE2, [4]=PE3, [6]=PE4
    for pe_out, pe_next_in in [(pe[0], pe[2]), (pe[2], pe[4]), (pe[4], pe[6])]:
        idx = _get_conv_pruning_idx(pe_out.c, sparsity)
        _zero_out_filters(pe_out.c, pe_out.bn, idx)   # 현재 레이어 출력
        _zero_in_channels(pe_next_in.c, idx)           # 다음 레이어 입력 연동
```

#### G_INV 구현 핵심

```python
def _prune_patch_merging(pm, sparsity):
    idx = _get_conv_pruning_idx(pm.conv1.c, sparsity)
    _zero_out_filters(pm.conv1.c, pm.conv1.bn, idx)  # expand 출력
    _zero_in_channels(pm.conv3.c, idx)                # reduce 입력 연동
    # conv2(DWConv), SE: zero가 forward에서 자연 전파
```

### 6.2 `classification/pruning/efficientvit_reducing.py`

#### G_PE reducing

```python
def _reduce_patch_embed(model):
    s1 = survived(PE1)  # PE1 surviving out channels
    s2 = survived(PE2)  # PE2 surviving out channels
    s3 = survived(PE3)  # PE3 surviving out channels

    PE1: Conv2d(3→n1, 3×3)        ← n1=len(s1)
    PE2: Conv2d(n1→n2, 3×3)       ← in=s1, out=s2
    PE3: Conv2d(n2→n3, 3×3)       ← in=s2, out=s3
    PE4: Conv2d(n3→128, 3×3)      ← in=s3, out=128 고정
```

#### G_INV reducing (PatchMerging 완전 구현)

```python
def _reduce_patch_merging(pm):
    survived_hid = survived(conv1)   # n = len(survived_hid)

    conv1: Conv2d(dim → n)
    conv2: DWConv(n, n, 3×3, groups=n)        ← DWConv 채널도 함께 축소
    SE:
      old_red = conv_reduce.out_channels
      new_red = max(1, round(n × old_red/old_hid))
      survived_red = topk norms of conv_reduce rows (after survived input)
      conv_reduce: Conv2d(n → new_red, 1×1)
      conv_expand: Conv2d(new_red → n, 1×1)
    conv3: Conv2d(n → out_dim)               ← out_dim 고정
```

SE reducing 시 내부 bottleneck 비율(0.25)을 유지하면서 채널 수를 비례 축소.

---

## 7. 수정된 기존 파일

### 7.1 `classification/engine.py` (+4줄)

```python
def train_one_epoch(..., pruner=None):   # 인자 추가

# loss_scaler() 직후
if pruner is not None:
    pruner.apply(model)
```

### 7.2 `classification/main.py` (+14줄)

```python
# args 추가
parser.add_argument('--pruning', action='store_true')
parser.add_argument('--target-compression', type=float, default=0.30)

# 학습 루프 직전
if args.pruning:
    pruner = EfficientViTPruner(model=model_without_ddp,
                                target_compression=args.target_compression)

# train_one_epoch 호출
train_stats = train_one_epoch(..., pruner=pruner)
```

---

## 8. 학습 실행 명령어

> **중요**: `main.py`의 import들(`from engine import`, `from model import`, `from data.samplers import` 등)은
> `classification/` 디렉토리를 기준으로 설계되어 있다.
> **반드시 `cd classification` 후 실행**해야 한다. `python -m classification.main`으로 실행하면
> `ModuleNotFoundError: No module named 'engine'` 오류가 발생한다.

```bash
> **주의**: pretrained 체크포인트에서 pruning 학습을 시작할 때는 반드시 `--finetune`을 사용한다.
> `--resume`은 optimizer/lr_scheduler/epoch까지 복원하기 때문에, 300 epoch 완료된 pretrained 체크포인트를
> `--resume`으로 로드하면 cosine annealing 끝 지점(lr ≈ 1e-6)에서 시작해 학습이 사실상 멈춘다.
>
> | 인자 | 복원 대상 | 사용 시점 |
> |------|---------|---------|
> | `--finetune` | 모델 가중치만 | **pretrained → 새 학습 시작** ← pruning에 사용 |
> | `--resume` | 모델 + optimizer + lr_scheduler + epoch | 중단된 학습 재개 |

```bash
# ✅ 올바른 실행 방법
cd /workspace/etri_iitp/JS/EfficientViT/classification

# 30% 압축 (권장 첫 실험)
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --finetune /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.50 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/pruning_50pct \
  --device cuda:0

# 50% 압축: --target-compression 0.50
# 75% 압축: --target-compression 0.75

# 중단된 pruning 학습 재개 (--resume 사용)
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --resume /workspace/etri_iitp/JS/EfficientViT/output/pruning_30pct/checkpoint_100.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.30 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/pruning_30pct \
  --device cuda:0

# Multi-GPU (classification/ 디렉토리 내에서)
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --finetune /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.30 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/pruning_30pct \
  --dist-eval
```

---

## 9. Reducing (학습 완료 후)

```bash
# classification/ 디렉토리 내에서 실행
cd /workspace/etri_iitp/JS/EfficientViT/classification

python -m pruning.efficientvit_reducing \
  --model EfficientViT_M4 \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/pruning_30pct/checkpoint_best.pth \
  --output /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_30pct.pth
```

reducing 순서: G_FFN/QK → G_INV (PatchMerging) → G_PE (PatchEmbed chain)

출력 예시:
```
[Reducing] 원본 파라미터: 8,804,228
[Reducing] 축소 후 파라미터: 6,163,000
[Reducing] 압축률: 30.0%
[Reducing] Forward pass 검증 OK
[Reducing] state_dict 저장 완료: .../reduced_m4_30pct.pth
[Reducing] full model 저장 완료: .../reduced_m4_30pct_full.pth
[Reducing] 사용법: model = torch.load('.../reduced_m4_30pct_full.pth').eval()
```

### Reducing 후 모델 실행 방법

> **중요**: Reducing 후 모델은 Conv 레이어 크기가 원본과 달라진다.
> `create_model('EfficientViT_M4')` + `load_state_dict(reduced.pth)` 는 **shape 불일치로 실패**.
> 아래 세 가지 방법 중 하나를 사용한다.

#### 방법 1 — full model 직접 로드 (가장 단순)

Reducing 시 `_full.pth`가 자동 생성된다. 이 파일로 즉시 실행 가능.

```python
import torch

model = torch.load('/workspace/.../reduced_m4_30pct_full.pth')
model.eval()

img = torch.zeros(1, 3, 224, 224)  # 실제 전처리된 이미지 텐서
with torch.no_grad():
    logits = model(img)          # (1, 1000)
pred = logits.argmax(dim=1)
```

#### 방법 2 — soft-pruned 체크포인트에서 직접 로드+reducing (유틸 함수)

`_full.pth` 없이 soft-pruned 학습 체크포인트(`.pth`)만 있어도 실행 가능.

```python
# classification/ 디렉토리 내에서
from pruning.efficientvit_reducing import load_reduced_model

model = load_reduced_model(
    checkpoint_path='/workspace/.../output/pruning_30pct/checkpoint_best.pth',
    model_name='EfficientViT_M4',
    num_classes=1000,
    device='cuda:0',
)
# 내부에서 자동으로 efficientvit_reducing() 적용 후 반환

with torch.no_grad():
    logits = model(img_tensor)
```

#### 방법 3 — main.py --eval 모드 (sparse 모델 평가)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --resume /workspace/etri_iitp/JS/EfficientViT/output/pruning_30pct/checkpoint_best.pth \
  --eval \
  --device cuda:0
```

> **주의**: `--eval` 모드는 **soft-pruned(sparse) 모델** 그대로 평가한다.
> Conv2d 레이어 크기는 원본과 같지만 weight에 0이 많은 상태이므로,
> 실제 Dense 모델보다 파라미터 수는 동일하지만 정확도가 약간 다를 수 있다.
> Reducing 후 Dense 모델의 정확도를 보려면 방법 4를 사용한다.

#### 방법 4 — eval_reduced.py (Reduced Dense 모델 정확도 평가) ← 권장

Reducing 완료 후 **실제 압축된 Dense 모델**의 정확도를 측정한다.

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

# ── 전체 1K 평가 ─────────────────────────────────────────────────────────────

# full.pth가 있을 때 (reducing 완료 후 자동 생성됨)
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_30pct_full.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --device cuda:0

# full.pth 없이 soft-pruned 체크포인트에서 바로 reducing + 평가
python eval_reduced.py \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/pruning_30pct/checkpoint_best.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --device cuda:0
```

출력 예시:
```
[Eval] full 모델 로드: .../reduced_m4_30pct_full.pth
[Eval] 파라미터 수: 6,163,000
[Eval] 검증 샘플: 50000, 클래스: 1000

[결과] Acc@1: 74.31%  Acc@5: 91.87%  Loss: 1.1042
```

### 파일 정리

| 파일 | 용도 | 평가 방법 |
|------|------|---------|
| `checkpoint_best.pth` | soft-pruned 학습 체크포인트 (sparse) | 방법 3 (`main.py --eval`) 또는 방법 4 (`eval_reduced.py --checkpoint`) |
| `reduced_m4_30pct.pth` | reducing 후 state_dict (dense, 소형) | **직접 평가 불가** (shape 불일치) → `load_reduced_model()`로 코드 내 사용 |
| `reduced_m4_30pct_full.pth` | reducing 후 전체 모델 객체 | **방법 4** (`eval_reduced.py --model-path`) ← 가장 단순 |

> **정리**: 최종 압축 성능 측정은 항상 방법 4(`eval_reduced.py`)를 사용한다.
> 방법 3(`main.py --eval`)은 학습 중간 확인용으로만 쓴다.

---

## 10. 달성 가능 압축률 분석

### Prunable 비율 (G_INV + G_PE 추가 후)

```
Prunable 그룹 합계  = 6,874,384 / 8,804,228 = 78.1%
Non-prunable 합계  = 1,929,844 / 8,804,228 = 21.9%

이론적 최대 압축률 ≈ 78.1%  (sparsity=1.0 가정 시)
실용적 최대 압축률 ≈ 74%    (sparsity=0.95 캡 적용 시)
```

### G_INV 추가 전후 비교

| 상태 | Prunable 비율 | 75% target sparsity | 예상 실제 압축률 |
|------|-------------|---------------------|----------------|
| G_FFN + G_QK만 | 68% | 1.10 → 클리핑 | ≈ 65% |
| **+ G_INV + G_PE** | **78%** | **0.96 → 0.95** | **≈ 74%** |

---

## 11. 핵심 제약사항

```
1. QK pruning: Q와 K 반드시 동일 상대 인덱스
   이유: Q^T @ K 에서 차원 일치 필요

2. FFN pruning: pw1.out == pw2.in 유지
   이유: expand → shrink 연결 차원

3. G_PE: PE4 출력 채널 고정
   이유: blocks1 입력 채널(embed_dim=128)과 직결

4. G_INV: conv3 출력 채널 고정
   이유: 다음 stage 첫 번째 블록 입력과 직결

5. DWConv (dw0, dw1) pruning 금지
   이유: -1.4% 정확도 하락 확인

6. CGA proj (W_out) pruning 금지
   이유: 전체 head V 출력 얼라인먼트 핵심

7. reducing 시 G_INV: SE도 반드시 함께 축소
   이유: conv2(DWConv)와 SE의 in_channels가 conv1 out_channels와 일치 필요

8. reducing 시 G_PE: PE1→PE4 chain 순서 유지
   이유: 각 PE의 survived_idx가 다음 PE의 in_idx로 사용됨
```

---

## 12. 개발 환경 참고사항

```
로컬 (macOS): Claude 코드 작성 / 파일 편집
GPU 서버:    실제 학습 및 reducing 실행

IDE 경고 (무시 가능):
  "Import 'timm.models' could not be resolved" — Pylance 정적 분석 경고.
  timm이 로컬에 미설치되어 발생. 서버에는 설치되어 있으므로 런타임 문제 없음.
  해당 import는 efficientvit_reducing.py의 main() CLI 함수 내부에만 위치.
```

---

## 13. 전체 압축 파이프라인 — 상세 구현 원리

### 13.1 파이프라인 개요

```
[1단계: 학습 중 Soft Pruning]
  학습 시작 → EfficientViTPruner 초기화 (sparsity 계산)
           → 매 step: forward → loss → optimizer.step()
                                     ↓
                              pruner.apply(model)
                              L2 norm 하위 필터 weight → 0.0
           → 다음 step: gradient가 0을 다시 살릴 수 있으나,
                        step 직후 또 0으로 강제 → 반복
           → 살아남은 채널만 실질적으로 최적화됨
           → 학습 완료 후: checkpoint 저장 (weight에 zeros 포함된 sparse 모델)

[2단계: Reducing (학습 후 1회)]
  sparse checkpoint 로드 → efficientvit_reducing(model)
  → norm==0인 필터 감지 → 물리적으로 작은 Conv2d/BN 신규 생성 → 가중치 복사
  → Dense 소형 모델 → 저장
```

---

### 13.2 target_compression → sparsity 자동 계산

`--target-compression` 값 하나만 바꾸면 모든 sparsity가 자동으로 결정된다.

#### 계산 흐름 (EfficientViTPruner.__init__)

```python
# 1. 모델 전체 파라미터 수 계산
info = count_prunable_params(model)
total   = info['total']           # 예: 8,804,228 (M4 전체)
p_total = info['prunable_total']  # 예: 6,874,384 (prunable 78.1%)

# 2. target_compression으로부터 sparsity 역산
#    "prunable 그룹에서 sparsity를 균등 적용했을 때 전체 압축률이 target이 되려면?"
raw_sparsity = (target_compression * total) / p_total
#    예) target=0.30 → raw = (0.30 × 8,804,228) / 6,874,384 ≈ 0.384

# 3. 안전 클리핑 (레이어 붕괴 방지)
self.sparsity_ffn = min(raw_sparsity, 0.95)   # G_FFN, G_PE, G_INV에 적용
self.sparsity_qk  = min(raw_sparsity, 0.90)   # G_QK에 적용 (별도 상한)
```

#### 세 가지 target 값별 계산 결과

| `--target-compression` | raw_sparsity | sparsity_ffn | sparsity_qk | 예상 실제 압축 |
|------------------------|-------------|-------------|------------|--------------|
| `0.30` | ≈ 0.384 | 0.384 | 0.384 | ≈ 30% |
| `0.50` | ≈ 0.640 | 0.640 | 0.640 | ≈ 50% |
| `0.75` | ≈ 0.960 | **0.950** (클리핑) | **0.900** (클리핑) | ≈ 74% |

> `0.75` target에서 raw_sparsity ≈ 0.96으로 상한 초과 → 클리핑 → 실제 ≈ 74%.
> Non-prunable 22%가 항상 남기 때문에 이론적 최대치가 78%이고 실용적 상한은 약 74%.

#### count_prunable_params 상세

```python
def count_prunable_params(model) -> dict:
    # G_FFN: FFN마다 pw1.c(h×ed) + pw1.bn(h×2) + pw2.c(ed×h)
    #   h=hidden dim, ed=embed_dim
    p_ffn += h * ed + h * 2 + ed * h

    # G_QK: head마다 Q+K conv(2*kd*dpth) + Q+K BN(2*kd*2) + dw conv(kd*ks²) + dw BN(kd*2)
    p_qk += 2 * kd * dpth + 2 * kd * 2 + kd * ks * ks + kd * 2

    # G_INV: PatchMerging마다 conv1.c(hid*in_d) + conv1.bn(hid*2) + conv3 input(hid*out_d)
    p_inv += hid * in_d + hid * 2 + hid * out_d

    # G_PE: PE1,PE2,PE3마다 conv.weight.numel() + out_ch*2(BN)
    p_pe += c.weight.numel() + c.weight.shape[0] * 2
```

---

### 13.3 요소별 Pruning 구현 원리

#### G_FFN — FFN expand/shrink (코드: `_prune_ffn`)

FFN은 `pw1(expand: ed→h)` → `pw2(shrink: h→ed)` 구조.
pw1의 출력 채널 = pw2의 입력 채널이므로 **반드시 동일 인덱스**를 양쪽에 적용해야 차원 불일치가 없다.

```python
def _prune_ffn(ffn, sparsity):
    # 1. pw1의 출력 필터(행) L2 norm 계산 → 하위 sparsity% 선택
    idx = _get_conv_pruning_idx(ffn.pw1.c, sparsity)
    #    norm = ||weight[i, :, 0, 0]||₂  for i in range(h)
    #    topk(norms, num_pruning, largest=False) → 최솟값 인덱스

    # 2. pw1 출력 필터 zero (conv weight + BN weight/bias/mean/var)
    _zero_out_filters(ffn.pw1.c, ffn.pw1.bn, idx)
    #    conv.weight.data[idx] = 0.0
    #    bn.weight[idx] = bn.bias[idx] = bn.running_mean[idx] = 0.0
    #    bn.running_var[idx] = 1.0  ← 분모 0 방지

    # 3. pw2 입력 채널(열) zero → 동일 인덱스로 연결 차원 유지
    _zero_in_channels(ffn.pw2.c, idx)
    #    conv.weight.data[:, idx] = 0.0
```

EfficientViTBlock은 `ffn0`, `ffn1` **두 개**의 FFN을 가지므로 둘 다 처리한다.
SubDWFFN(Sequential)은 `block[1].m`이 FFN.

#### G_QK — CGA Q+K projection (코드: `_prune_cga_qk`)

CGA는 `qkvs[i]`라는 단일 Conv2d_BN이 Q+K+V를 한꺼번에 출력한다.
출력 채널 레이아웃: `[Q(0:kd) | K(kd:2kd) | V(2kd:)]`

QK^T 연산에서 Q와 K는 채널 수가 같아야 하므로 **동일 상대 인덱스**를 Q와 K에 적용해야 한다.
V, proj는 절대 건드리지 않는다.

```python
def _prune_cga_qk(cga, sparsity):
    key_dim = cga.key_dim  # M4: 16 (모든 stage 동일)
    num_pruning = min(round(key_dim * sparsity), key_dim - _MIN_SURVIVE)

    for qkv in cga.qkvs:  # head 수만큼 반복
        conv, bn = qkv.c, qkv.bn

        # 1. Q portion(0:key_dim)만 norm 계산
        q_norms = torch.norm(conv.weight[:key_dim].view(key_dim, -1), dim=1)
        _, q_idx = torch.topk(q_norms, num_pruning, largest=False)

        # 2. Q zero (conv filter + BN)
        _zero_out_filters(conv, bn, q_idx)

        # 3. K zero: K의 절대 인덱스 = q_idx + key_dim (동일 상대 위치)
        k_idx = q_idx + key_dim
        conv.weight.data[k_idx] = 0.0
        bn.weight.data[k_idx] = bn.bias.data[k_idx] = bn.running_mean[k_idx] = 0.0
        bn.running_var[k_idx] = 1.0
        # V(2kd:) 채널 → 절대 불가 (W_out 입력과 직결)
```

#### G_PE — PatchEmbed Conv1~3 chain (코드: `_prune_patch_embed`)

`patch_embed`는 Sequential로 `[0]=PE1, [1]=ReLU, [2]=PE2, [3]=ReLU, [4]=PE3, [5]=ReLU, [6]=PE4`.
PE1의 출력이 PE2의 입력이 되는 chain 구조이므로, **현재 레이어 출력 pruning → 다음 레이어 입력 채널 연동 zero**가 필요하다.

```python
def _prune_patch_embed(model, sparsity):
    pe = model.patch_embed

    # PE1 출력(16ch) → PE2 입력 연동
    idx1 = _get_conv_pruning_idx(pe[0].c, sparsity)   # PE1 필터 선택
    _zero_out_filters(pe[0].c, pe[0].bn, idx1)         # PE1 출력 zero
    _zero_in_channels(pe[2].c, idx1)                   # PE2 입력 연동 zero

    # PE2 출력(32ch) → PE3 입력 연동 (독립적 sparsity, 별도 idx)
    idx2 = _get_conv_pruning_idx(pe[2].c, sparsity)
    _zero_out_filters(pe[2].c, pe[2].bn, idx2)
    _zero_in_channels(pe[4].c, idx2)

    # PE3 출력(64ch) → PE4 입력 연동
    idx3 = _get_conv_pruning_idx(pe[4].c, sparsity)
    _zero_out_filters(pe[4].c, pe[4].bn, idx3)
    _zero_in_channels(pe[6].c, idx3)
    # PE4 출력(128ch=embed_dim) → blocks1 입력과 직결 → 절대 건드리지 않음
```

각 PE 레이어는 독립적으로 sparsity를 적용 (per-layer). idx1, idx2, idx3는 각각 다른 필터를 선택한다.

#### G_INV — PatchMerging expand/reduce (코드: `_prune_patch_merging`)

PatchMerging은 `conv1(expand, dim→4dim)` → `conv2(DWConv)` → `SE` → `conv3(reduce, 4dim→out_dim)` 구조.
conv1 출력과 conv3 입력이 같은 차원이므로 동일 인덱스 연동이 필요하다.
conv2(DWConv)와 SE는 soft pruning 중에는 zero가 forward에서 자연 전파되므로 별도 처리 불필요.

```python
def _prune_patch_merging(pm, sparsity):
    # conv1 출력 필터(4*dim개) 중 sparsity% 선택
    idx = _get_conv_pruning_idx(pm.conv1.c, sparsity)
    _zero_out_filters(pm.conv1.c, pm.conv1.bn, idx)  # expand 출력 zero

    # conv3 입력 채널 연동 zero (출력=out_dim은 고정, 절대 불가)
    _zero_in_channels(pm.conv3.c, idx)

    # conv2(DWConv): zero input channel → zero output channel (자동)
    # SE: zero channel → global avg pool 시 0 기여 (자동)
```

---

### 13.4 Soft Pruning의 작동 원리 (학습 루프)

```python
# engine.py - train_one_epoch
for samples, targets in data_loader:
    outputs = model(samples)
    loss = criterion(samples, outputs, targets)

    optimizer.zero_grad()
    loss_scaler(loss, optimizer, ...)  # loss.backward() + optimizer.step() 포함

    # ← 이 시점: optimizer.step()이 weight를 업데이트한 직후
    if pruner is not None:
        pruner.apply(model)   # pruner.apply → efficientvit_pruning 호출
        # → L2 norm 하위 필터들의 weight 강제 0 설정
        # → BN도 함께 zero (bias=0, weight=0, mean=0, var=1)

    torch.cuda.synchronize()
    model_ema.update(model)   # EMA도 masked 상태에서 업데이트
```

**왜 gradient로 0이 살아나지 않는가?**

```
step N:   weight[i] = 0  (pruner가 강제 zero)
step N+1: backward → grad[i] ≠ 0 (살아있는 다른 채널과의 연결로 gradient 발생)
          optimizer.step() → weight[i] = 0 + lr × grad[i] ≠ 0  (잠깐 살아남)
          pruner.apply() → weight[i] = 0  (즉시 다시 강제 zero)
```

결과: 선택된 필터는 매 step에서 weight가 0으로 리셋되어 사실상 비활성화.
살아남은 필터들은 그 채널들만으로 최적화를 진행 → 정보 압축 학습.

---

### 13.5 요소별 Reducing 구현 원리

Soft pruning 학습 완료 후 `efficientvit_reducing.py`로 zero 채널을 물리적으로 제거한다.
핵심 원칙: **norm==0인 출력 필터 감지 → 작은 새 레이어 생성 → survived weight 복사**.

#### 공통 헬퍼: `_get_survived_conv_out_idx` + `_reduce_conv_bn`

```python
def _get_survived_conv_out_idx(conv):
    n = conv.weight.shape[0]
    norms = torch.norm(conv.weight.view(n, -1), dim=1)
    return torch.where(norms != 0)[0]   # norm > 0인 필터 인덱스

def _reduce_conv_bn(old_conv, old_bn, out_idx, in_idx=None, groups=1):
    n_out = len(out_idx)
    n_in  = old_conv.in_channels if in_idx is None else len(in_idx)
    new_conv = nn.Conv2d(n_in, n_out, ks, stride, padding, groups=groups, bias=False)
    new_bn   = nn.BatchNorm2d(n_out)
    with torch.no_grad():
        w = old_conv.weight[out_idx]          # 출력 필터 선택
        if in_idx is not None and groups == 1:
            w = w[:, in_idx]                  # 입력 채널 선택 (DWConv 제외)
        new_conv.weight.copy_(w)
        new_bn.weight.copy_(old_bn.weight[out_idx])
        new_bn.bias.copy_(old_bn.bias[out_idx])
        new_bn.running_mean.copy_(old_bn.running_mean[out_idx])
        new_bn.running_var.copy_(old_bn.running_var[out_idx])
    return new_conv, new_bn
```

#### G_FFN reducing (`_reduce_ffn`)

```python
def _reduce_ffn(ffn):
    survived = _get_survived_conv_out_idx(ffn.pw1.c)  # norm>0인 pw1 출력 필터
    if len(survived) == ffn.pw1.c.weight.shape[0]:
        return   # 변화 없음

    # pw1: out_channels를 survived 수로 줄인 새 Conv2d + BN 생성
    new_c1, new_bn1 = _reduce_conv_bn(ffn.pw1.c, ffn.pw1.bn, survived)
    ffn.pw1.c, ffn.pw1.bn = new_c1, new_bn1

    # pw2: in_channels만 축소 (out_channels=ed는 고정)
    new_c2 = nn.Conv2d(len(survived), old_c2.out_channels, ...)
    new_c2.weight.copy_(old_c2.weight[:, survived])   # 열 선택
    ffn.pw2.c = new_c2
    # pw2.bn: 출력 채널(ed) 불변 → 그대로 유지
```

#### G_QK reducing (`_reduce_cga_qk`)

```python
def _reduce_cga_qk(cga):
    for i in range(cga.num_heads):
        qkv = cga.qkvs[i]
        # Q portion 기준 survived 인덱스 계산
        q_norms = torch.norm(conv.weight[:key_dim].view(key_dim,-1), dim=1)
        survived_q = torch.where(q_norms != 0)[0]
        survived_k = survived_q + key_dim              # K: 동일 상대 인덱스
        survived_v = torch.arange(2*key_dim, 2*key_dim+d)  # V: 전부 유지
        out_idx = torch.cat([survived_q, survived_k, survived_v])

        # qkvs[i]: out_channels = len(survived_q)*2 + d (V는 전부)
        new_conv = Conv2d(in_ch, len(out_idx), ...)
        new_conv.weight.copy_(conv.weight[out_idx])

        # dws[i]: DWConv → in_ch=out_ch=survived_q, groups=survived_q
        dw_c, dw_bn = _reduce_conv_bn(..., survived_q, survived_q, groups=len(survived_q))

    # key_dim, scale 업데이트 (M4: 16 → 줄어든 수로)
    cga.key_dim = new_kd
    cga.scale   = new_kd ** -0.5
```

#### G_PE reducing (`_reduce_patch_embed`)

chain 방식: 각 PE의 survived out 인덱스가 다음 PE의 in 인덱스로 사용된다.

```python
def _reduce_patch_embed(model):
    pe = model.patch_embed
    s1 = _get_survived_conv_out_idx(pe[0].c)  # PE1 survived out (≤16)
    s2 = _get_survived_conv_out_idx(pe[2].c)  # PE2 survived out (≤32)
    s3 = _get_survived_conv_out_idx(pe[4].c)  # PE3 survived out (≤64)

    # PE1: in=3(고정), out=n1
    _reduce_conv_bn(pe[0].c, pe[0].bn, s1, in_idx=None)

    # PE2: in=n1(s1 선택), out=n2(s2 선택)
    _reduce_conv_bn(pe[2].c, pe[2].bn, s2, in_idx=s1)

    # PE3: in=n2(s2 선택), out=n3(s3 선택)
    _reduce_conv_bn(pe[4].c, pe[4].bn, s3, in_idx=s2)

    # PE4: in=n3(s3 선택), out=128(embed_dim) 고정
    new_c4 = Conv2d(len(s3), 128, ...)
    new_c4.weight.copy_(old_c4.weight[:, s3])  # 출력 128행은 전부 유지
    pe[6].c = new_c4  # PE4 BN 출력 채널 불변
```

#### G_INV reducing + SE (`_reduce_patch_merging`)

PatchMerging reducing은 가장 복잡하다. conv2(DWConv)와 SE도 함께 축소해야 한다.

```python
def _reduce_patch_merging(pm):
    survived_hid = _get_survived_conv_out_idx(pm.conv1.c)  # conv1 survived out
    n = len(survived_hid)
    old_hid = pm.conv1.c.weight.shape[0]

    # conv1: out→n
    _reduce_conv_bn(pm.conv1.c, pm.conv1.bn, survived_hid)

    # conv2: DWConv → channels=n, groups=n
    _reduce_conv_bn(pm.conv2.c, pm.conv2.bn,
                    survived_hid, survived_hid, groups=n)

    # SE: conv_reduce(hid→red) + conv_expand(red→hid) 모두 축소
    se = pm.se
    old_red = se.conv_reduce.weight.shape[0]    # 기존 squeeze 채널 수
    new_red = max(1, round(n * old_red / old_hid))  # 비율 유지

    # conv_reduce: 입력 survived_hid 선택 후, 출력 row norm 상위 new_red 선택
    cr_w = old_cr.weight[:, survived_hid]       # [old_red, n, 1, 1]
    cr_norms = torch.norm(cr_w.view(old_red, -1), dim=1)
    _, survived_red = torch.topk(cr_norms, new_red, largest=True)
    new_cr = Conv2d(n, new_red, 1, bias=True)
    new_cr.weight.copy_(cr_w[survived_red])
    new_cr.bias.copy_(old_cr.bias[survived_red])

    # conv_expand: 입력 survived_red 선택, 출력 survived_hid 선택
    ce_w = old_ce.weight[survived_hid][:, survived_red]  # [n, new_red, 1, 1]
    new_ce = Conv2d(new_red, n, 1, bias=True)
    new_ce.weight.copy_(ce_w)
    new_ce.bias.copy_(old_ce.bias[survived_hid])

    # conv3: in=n(survived_hid 선택), out=out_dim 고정
    new_c3 = Conv2d(n, out_dim, 1, bias=False)
    new_c3.weight.copy_(old_c3.weight[:, survived_hid])
```

SE reducing에서 핵심은 `new_red = max(1, round(n × old_red/old_hid))`.
hid_dim이 줄어든 비율만큼 SE 내부 bottleneck도 비례 축소해서 0.25 비율을 유지한다.

#### reducing 실행 순서 (`efficientvit_reducing`)

```python
def efficientvit_reducing(model):
    # 1. G_FFN + G_QK (blocks1/2/3 전체)
    for block_list in [model.blocks1, model.blocks2, model.blocks3]:
        for block:
            EfficientViTBlock → _reduce_ffn(ffn0.m), _reduce_ffn(ffn1.m), _reduce_cga_qk(attn)
            Sequential → _reduce_ffn(block[1].m)
            PatchMerging → _reduce_patch_merging(block)

    # 2. G_PE (patch_embed chain)
    _reduce_patch_embed(model)
```

순서가 중요한 이유: G_INV reducing이 PatchMerging.conv1 out_channels를 변경하지만,
그 출력이 blocks의 다음 레이어 입력에는 영향 없으므로 G_PE와 독립.
G_PE chain은 PE4 출력(embed_dim)을 고정한 채 PE1~PE3만 축소.

---

### 13.6 target_compression 하나로 모든 게 결정되는가?

**Yes — 단 한 가지 인자만 바꾸면 된다.**

```bash
# 30% 압축
python -m classification.main ... --pruning --target-compression 0.30

# 50% 압축
python -m classification.main ... --pruning --target-compression 0.50

# 75% 압축 (이론 한계 ≈74%)
python -m classification.main ... --pruning --target-compression 0.75
```

내부 자동 계산 흐름:

```
target_compression (사용자 입력)
    ↓
count_prunable_params(model)   ← 실제 모델 파라미터 수에서 직접 계산
    ↓
raw_sparsity = target × total / prunable_total
    ↓
sparsity_ffn = min(raw_sparsity, 0.95)   ← G_FFN, G_PE, G_INV
sparsity_qk  = min(raw_sparsity, 0.90)   ← G_QK
    ↓
매 step: pruner.apply(model) → 각 레이어 독립적으로 per-layer pruning
    ↓
reducing 시: norm==0 필터 자동 감지 → 물리적 제거
```

**주의사항**: 75% target에서 raw_sparsity가 상한을 초과해 클리핑되면, `EfficientViTPruner` 초기화 시 출력 메시지에 `[클리핑: target 미달]`이 표시된다. 이 경우 G_PE와 G_INV에 추가 pruning 그룹을 확장하거나 _MIN_SURVIVE를 줄이는 방식으로 대응할 수 있다.

---

## 15. Backbone과 Head 개념

### 15.1 두 부분의 역할

딥러닝 분류 모델은 기능적으로 두 부분으로 나뉜다.

```
입력 이미지 (3×224×224)
        ↓
┌─────────────────────────────┐
│          Backbone           │  ← "특징 추출기"
│  (patch_embed + blocks1~3)  │    이미지에서 의미 있는 패턴을 뽑아냄
└─────────────────────────────┘
        ↓  특징 벡터 (384차원)
┌─────────────────────────────┐
│            Head             │  ← "분류기"
│       (BN + Linear)         │    특징 벡터를 클래스 점수로 변환
└─────────────────────────────┘
        ↓  클래스 점수 (N개)
```

| | Backbone | Head |
|---|---|---|
| **역할** | 이미지 → 추상적 특징 벡터 추출 | 특징 벡터 → 클래스 점수 변환 |
| **학습하는 것** | "고양이 귀 모양", "바퀴 패턴" 같은 시각적 특징 | 어떤 특징 조합이 어떤 클래스인지 |
| **클래스 의존성** | **없음** — 이미지 특징은 클래스 수와 무관 | **있음** — 출력 뉴런 수 = 클래스 수 |
| **재사용 가능성** | 다른 태스크에도 그대로 쓸 수 있음 | 클래스 수가 바뀌면 교체 필요 |

### 15.2 EfficientViT M4의 실제 구조

```
model.patch_embed              ← Backbone 시작
  Conv(3→16) → Conv(16→32) → Conv(32→64) → Conv(64→128)
  역할: 이미지를 토큰(패치) 시퀀스로 변환

model.blocks1                  ← Backbone (Stage 1)
  EfficientViTBlock × 1 (채널 128)
  역할: 저수준 특징 (엣지, 텍스처)

model.blocks2                  ← Backbone (Stage 2)
  SubDWFFN + PatchMerging + SubDWFFN + EVBlock × 2 (채널 256)
  역할: 중간 수준 특징 (부분 모양)

model.blocks3                  ← Backbone (Stage 3)
  SubDWFFN + PatchMerging + SubDWFFN + EVBlock × 3 (채널 384)
  역할: 고수준 특징 (전체 구조, 맥락)

                ↓ Global Average Pooling → 384차원 벡터

model.head                     ← Head
  BatchNorm(384) + Linear(384 → N)
  역할: 384차원 특징을 N개 클래스 점수로 변환
```

Soft Pruning의 타겟(G_FFN, G_QK 등)은 모두 **Backbone** 안에 있다. Head는 pruning 대상이 아니다.

### 15.3 클래스 수가 바뀌면 무슨 일이 일어나는가

```
                 Backbone                     Head
                    ↓                           ↓
1000-class:  [특징 추출, 8.4M params]  →  Linear(384 → 1000)  [384K params]
10-class:    [특징 추출, 8.4M params]  →  Linear(384 → 10)    [3.8K params]
```

Backbone은 "이미지를 이해하는 능력"을 학습한다. 이 능력은 클래스 수와 무관하다.
고양이/개를 구분하는 특징이나 10,000가지를 구분하는 특징이나 추출 방식은 동일하다.

반면 Head의 Linear 레이어는 **출력 뉴런 수 = 클래스 수**이므로 클래스 수가 바뀌면 shape가 달라져 재사용 불가.

### 15.4 Pretrained 파일을 쓰는 게 의미가 있는가?

**의미 있다. 오히려 반드시 써야 한다.**

이유: Backbone에 pretrained 가중치를 쓰면 처음부터 학습할 필요가 없기 때문이다.

```
시나리오 A: pretrained backbone 사용 (--finetune)
  epoch 1부터 384차원 특징이 이미 "의미 있는 표현"
  head만 10개 클래스를 구분하도록 빠르게 수렴
  → 20~50 epoch으로 충분히 수렴 가능

시나리오 B: 처음부터 학습 (pretrained 없음)
  epoch 1에는 backbone도 아무것도 모르는 상태 (랜덤 특징)
  backbone + head 전부 처음부터 학습
  → 300 epoch 이상 필요, 최종 성능도 보통 낮음
```

pretrained backbone은 ImageNet 1000개 클래스를 학습하면서 **이미지의 범용적 시각 특징**을 이미 학습한 상태다. 10개 클래스 학습을 할 때도 그 특징 추출 능력이 그대로 유효하게 적용된다.

#### 10-class 서브셋에서 pretrained 로드 흐름

```python
# 1. 모델을 10개 클래스로 생성
model = create_model('EfficientViT_M4', num_classes=10)
#   → head.l: Linear(384, 10)   [shape: (10, 384)]

# 2. pretrained 체크포인트 로드 시도
checkpoint = load('efficientvit_m4.pth')
#   checkpoint['head.l.weight'].shape = (1000, 384)  ← shape 불일치!

# 3. main.py의 자동 처리
for k in ['head.l.weight', 'head.l.bias']:
    if checkpoint[k].shape != model.state_dict()[k].shape:
        del checkpoint[k]   # ← head만 제거, backbone은 그대로 유지

# 4. 결과
model.load_state_dict(checkpoint, strict=False)
#   backbone (8.4M params): pretrained 가중치 ✅
#   head (3.8K params):     Xavier 랜덤 초기화 ✅ (학습으로 수렴)
```

#### 정리

| | pretrained 사용 | 처음부터 학습 |
|---|---|---|
| Backbone 초기 상태 | 이미 최적화된 특징 추출기 | 완전 랜덤 |
| Head 초기 상태 | 랜덤 (동일) | 랜덤 (동일) |
| 수렴 속도 | 빠름 (수십 epoch) | 느림 (수백 epoch) |
| 최종 성능 | 높음 | 상대적으로 낮음 |
| 10-class에서 추천 | **강력 권장** | 비추천 |

Pruning 실험 맥락에서는 특히 중요하다. Pruning은 "이미 잘 학습된 모델의 덜 중요한 가중치를 제거"하는 것이므로, pretrained 없이 시작하면 pruning할 의미 있는 가중치 자체가 없다.

---

## 16. 10-Class 서브셋 훈련 — 실행 방법

전체 ImageNet 1K 대신 10개 클래스만 뽑아서 훈련·평가할 수 있다.
데이터셋 복사 없이 **원본 ImageNet 디렉토리를 그대로 사용**하며, 코드 레벨에서 필터링한다.

### 16.1 핵심 원리: 분류 헤드 교체

```
원본 M4 head:  Linear(384 → 1000)  ← pretrained 가중치 존재
10-class head: Linear(384 → 10)    ← 새로 초기화 필요
```

`--finetune`으로 pretrained 체크포인트를 로드할 때, `main.py`에 이미 다음 코드가 있다:

```python
for k in ['head.l.weight', 'head.l.bias', ...]:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]   # ← shape 불일치 시 자동 제거
```

`num_classes=10`으로 모델을 생성하면 `head.l`의 shape이 `(10, 384)`가 되어 pretrained `(1000, 384)`와 불일치 → 자동으로 제거 후 Xavier 초기화로 랜덤 재설정된다.
**backbone(patch_embed, blocks1~3)은 pretrained 가중치를 그대로 사용**하고, head만 처음부터 학습한다.

#### 요약

| 레이어 | pretrained 로드 여부 | 비고 |
|--------|-------------------|------|
| patch_embed | ✅ | stem conv 가중치 재사용 |
| blocks1~3 | ✅ | FFN, CGA 가중치 재사용 |
| head.bn | ✅ | BN 통계 재사용 |
| **head.l** | ❌ (랜덤 초기화) | shape 불일치(1000→10)로 자동 제거 |

### 16.2 데이터셋 구조 (서버 기준)

서버의 ImageNet은 숫자 폴더명으로 구성되어 있다:

```
/workspace/etri_iitp/JS/EfficientViT/data/imagenet/
├── train/
│   ├── 00000/   ← 클래스 0
│   ├── 00001/   ← 클래스 1
│   │   ...
│   └── 00999/   ← 클래스 999
└── val/
    ├── 00000/
    │   ...
    └── 00999/
```

`--data-set IMNET10` 옵션은 이 구조에서 지정한 10개 폴더만 로드하고 레이블을 0~9로 재매핑한다. 파일 복사/심볼릭 링크 생성 불필요.

### 16.3 SubsetImageFolder 작동 원리

```python
# datasets.py 내 SubsetImageFolder 흐름
ImageFolder(root)                         # 전체 1000개 클래스 로드
    ↓ filter
selected = ['00000', '00001', ..., '00009']

old_to_new = {0: 0, 1: 1, ..., 9: 9}     # 폴더 순서가 레이블 순서

self.samples = [                           # 10개 클래스 샘플만 유지
    (path, new_label)
    for path, old_label in all_samples
    if old_label in old_to_new
]
# 결과: len(selected)개 클래스, 레이블 0..9
```

### 16.4 학습 명령어

> **중요**: pretrained 체크포인트를 로드할 때 반드시 `--finetune` 사용 (헤드 자동 교체).
> `--resume`은 head shape 불일치로 실패하므로 10-class 훈련에는 사용 불가.

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

# ── Baseline (pruning 없음) — 비교 기준 성능 측정용 ──────────────────────────
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --finetune /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/subset10_baseline \
  --device cuda:7

# 특정 10개 클래스 지정 (--subset-classes)
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --subset-classes "00000,00001,00002,00003,00004,00005,00006,00007,00008,00009" \
  --finetune /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/subset10_baseline \
  --device cuda:0

# Soft Pruning + 10-class 서브셋 (30% 압축 예시)
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --finetune /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.50 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/subset10_pruning_50 \
  --device cuda:7
```

### 16.5 Reducing (10-class 서브셋 학습 완료 후)

Reducing 시 `--num-classes 10` 인자를 반드시 전달한다:

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

python -m pruning.efficientvit_reducing \
  --model EfficientViT_M4 \
  --num-classes 10 \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/subset10_pruning_70/checkpoint_299.pth \
  --output /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_subset10_70.pth
```

생성 파일:
- `reduced_m4_subset10_30pct.pth` — state_dict
- `reduced_m4_subset10_30pct_full.pth` — 즉시 사용 가능한 full 모델

### 16.6 평가

#### Sparse 모델 평가 (soft-pruned, reducing 전)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --resume /workspace/etri_iitp/JS/EfficientViT/output/subset10_pruning_30pct/checkpoint_best.pth \
  --eval \
  --device cuda:0
```

> `main.py --eval`은 sparse(soft-pruned) 모델을 평가한다. Conv 크기는 원본과 동일하지만 weight에 0이 많은 상태.

#### Reduced Dense 모델 평가 (reducing 후) ← 최종 성능 측정 시 사용

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

# full.pth로 평가 (reducing 완료 후)
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_subset10_70_full.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --device cuda:0

# soft-pruned 체크포인트에서 바로 reducing + 평가 (full.pth 없어도 됨)
python eval_reduced.py \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/subset10_pruning_30pct/checkpoint_best.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --num-classes 10 \
  --device cuda:0

# Baseline (pruning 없음) 서브셋 평가
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/subset10_baseline/checkpoint_299.pth  \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET10 \
  --device cuda:0
```

### 16.7 전체 1K와 10-class 비교

| 항목 | 전체 1K | 10-class 서브셋 |
|------|---------|----------------|
| 데이터 크기 | ~120GB | ~1.2GB (추정) |
| 에폭당 시간 | ~15분 (8GPU) | ~1분 (1GPU) |
| 분류 헤드 | Linear(384→1000) | Linear(384→10) |
| pretrained head | ✅ 재사용 | ❌ 랜덤 초기화 |
| backbone | ✅ 재사용 | ✅ 재사용 |
| 용도 | 최종 성능 검증 | 빠른 pruning 코드 검증 |

---

## 14. 파일 구조

```
EfficientViT/
├── CLAUDE.md                           ← 원래 구현 가이드
├── IMPLEMENTATION.md                   ← 본 문서
├── classification/
│   ├── model/
│   │   └── efficientvit.py             ← 원본 (미수정)
│   ├── pruning/                        ← 신규
│   │   ├── __init__.py
│   │   ├── efficientvit_pruning.py     ← Soft Pruning (G_FFN/QK/PE/INV)
│   │   └── efficientvit_reducing.py   ← Dense 변환 (PE chain + PM SE 포함)
│   ├── data/
│   │   └── datasets.py                 ← SubsetImageFolder + IMNET10 추가
│   ├── engine.py                       ← +4줄 (pruner 삽입)
│   ├── main.py                         ← +14줄 (CLI + 초기화) + IMNET10/subset-classes 인자
│   └── eval_reduced.py                 ← Reduced Dense 모델 정확도 평가 스크립트
└── YOLO pruning/                       ← 참고용 원본 (미수정)
```

---

## 15. Full ImageNet 1K Pruning 명령어 (WandB 포함)

> **사전 준비 (서버에서 한 번만)**
> ```bash
> pip install wandb
> wandb login  # API 키 입력 (환경변수 WANDB_API_KEY 로도 가능)
> ```

### 15.1 Baseline 평가 (pretrained 체크포인트 사용)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --device cuda:0
```

### 15.2 Soft Pruning 학습 (30% / 50% / 70%)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

# ── 30% 압축 ──────────────────────────────────────────────────────────────
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --resume /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-5 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.30 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/Fullimnet_pruning_30pct \
  --wandb \
  --wandb-project Fullimnet-pruning_1k \
  --wandb-run-name "m4_Fullimnet_pruning_30pct_1k" 
  --num_workers 2 \
  --device cuda:0
  

# ── 50% 압축 ──────────────────────────────────────────────────────────────
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --resume /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-5 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.50 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/Fullimnet_pruning_50pct \
  --wandb \
  --wandb-project Fullimnet-pruning_1k \
  --wandb-run-name "m4_Fullimnet_pruning_50pct_1k" \
  --num_workers 2 \
  --device cuda:0

# ── 70% 압축 ──────────────────────────────────────────────────────────────
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --resume /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 --epochs 300 \
  --opt adamw --lr 1e-5 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning --target-compression 0.70 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/Fullimnet_pruning_70pct \
  --wandb \
  --wandb-project Efficientvit-pruning_1k \
  --wandb-run-name "m4_pruning_70pct_1k" \
   --num_workers 2 \
  --device cuda:7
```

### 15.3 학습 완료 후 Reducing (Dense 변환)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

# 30%
python -m pruning.efficientvit_reducing \
  --model EfficientViT_M4 \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/fullimnet_pruning_30pct/checkpoint_best.pth \
  --output /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_fullimnet_30pct_full.pth

# 50%
python -m pruning.efficientvit_reducing \
  --model EfficientViT_M4 \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/fullimnet_pruning_50pct/checkpoint_best.pth \
  --output /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_fullimnet_50pct_full.pth

# 70%
python -m pruning.efficientvit_reducing \
  --model EfficientViT_M4 \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/fullimnet_pruning_70pct/checkpoint_best.pth \
  --output /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_fullimnet_70pct_full.pth
```

### 15.4 Dense 모델 평가 (최종 정확도 측정)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/classification

# Baseline 평가
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/fullimnet_baseline/checkpoint_best.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --device cuda:0

# 30% 압축 dense 모델 평가
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_fullimnet_30pct_full.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --device cuda:0

# 50% 압축 dense 모델 평가
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_fullimnet_50pct_full.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --device cuda:0

# 70% 압축 dense 모델 평가
python eval_reduced.py \
  --model-path /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_fullimnet_70pct_full.pth \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --data-set IMNET \
  --device cuda:0
```

### 15.5 파라미터 수 (Full ImageNet 기준)

| 설정 | 파라미터 수 | 압축률 |
|------|------------|--------|
| M4 원본 | 8,423,078 | - |
| 30% 압축 후 | ~5,896,155 | 30.0% |
| 50% 압축 후 | ~4,211,539 | 50.0% |
| 70% 압축 후 | ~2,526,923 | 70.0% |

> 실제 파라미터 수는 reducing 후 출력되는 값으로 확인할 것
