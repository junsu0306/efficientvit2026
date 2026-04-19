# Soft Pruning 방법론 — 다른 아키텍처 적용 가이드

> 본 문서는 EfficientViT M4에 적용한 Soft Pruning 구현을 바탕으로,  
> **임의의 모델에 동일한 방법론을 적용할 때 따라야 할 원칙과 주의사항**을 기술한다.

---

## 목차

1. [핵심 철학](#1-핵심-철학)
2. [전체 파이프라인](#2-전체-파이프라인)
3. [Phase 1 — Soft Pruning (학습 중)](#3-phase-1--soft-pruning-학습-중)
4. [Phase 2 — Reducing (학습 완료 후)](#4-phase-2--reducing-학습-완료-후)
5. [Sparsity 계산 — 이진탐색 방법](#5-sparsity-계산--이진탐색-방법)
6. [Prunable / Non-Prunable 레이어 분류 원칙](#6-prunable--non-prunable-레이어-분류-원칙)
7. [연동 Pruning — coupled 레이어 처리](#7-연동-pruning--coupled-레이어-처리)
8. [주의사항 및 흔한 실수](#8-주의사항-및-흔한-실수)
9. [새 아키텍처 적용 체크리스트](#9-새-아키텍처-적용-체크리스트)
10. [EfficientViT 구현 요약 (레퍼런스)](#10-efficientvit-구현-요약-레퍼런스)

---

## 1. 핵심 철학

Soft Pruning은 weight를 **완전히 삭제하지 않고 0으로 마스킹**한 채로 학습을 계속하는 방식이다.

```
[일반 학습]   forward → loss → backward → optimizer.step()

[Soft Pruning]  forward → loss → backward → optimizer.step()
                                                       ↓
                                              L2 norm 하위 X% weight를 0으로 리셋
                                                       ↓
                                              다음 iteration (gradient로 조금 살아나도)
                                                       ↓
                                              다시 0으로 리셋 → 반복
```

**왜 이 방식이 효과적인가:**

| 특성 | 설명 |
|------|------|
| **Soft** | 0으로 리셋되었어도 gradient를 받으면 살아날 수 있음 → 진짜 중요한 weight만 살아남음 |
| **Dynamic** | 어떤 채널이 중요한지를 매 step마다 재평가 → 학습 초기에 잘못 제거된 채널이 복구 가능 |
| **Dense 유지** | 학습 중에는 구조가 바뀌지 않음 → 기존 학습 코드 재사용, DDP 호환 |

Hard Pruning(한 번 제거하면 끝)과의 차이:
- Hard: 학습 중간에 아키텍처 변경 필요 → 복잡한 파이프라인
- **Soft: 학습 파이프라인 그대로, optimizer.step() 이후 한 줄만 추가**

---

## 2. 전체 파이프라인

```
1. 사전학습 모델 (pretrained checkpoint) 로드
        ↓
2. Soft Pruning 학습 (300 epoch)
   - 매 optimizer.step() 이후 pruning 함수 호출
   - weight가 0/non-zero를 반복하며 수렴
        ↓
3. checkpoint_best.pth 저장 (원본 아키텍처 shape 그대로)
        ↓
4. Reducing 실행 (한 번만)
   - 0인 채널 물리적 제거
   - 작고 빠른 Dense 모델 생성
        ↓
5. 압축률 검증 (B - A) / B × 100 ≥ target
```

---

## 3. Phase 1 — Soft Pruning (학습 중)

### 3.1 삽입 위치

`optimizer.step()` 직후, 다음 `forward()` 이전:

```python
# engine.py (학습 루프)
for samples, targets in data_loader:
    outputs = model(samples)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()

    # backward + optimizer.step() (loss_scaler 래핑 또는 직접 호출)
    loss_scaler(loss, optimizer, ...)   # ← 여기 안에서 backward + step 수행

    # ★ 이 위치에 pruning 삽입 ★
    if pruner is not None:
        pruner.apply(model)             # ← L2 norm 하위 X% → 0 리셋

    # 이후 model_ema 업데이트 등 진행
```

**왜 이 위치인가:**
- backward 이후 → gradient 정보가 weight에 반영된 직후
- 다음 forward 이전 → 0으로 만든 weight가 다음 forward에 영향
- model_ema 업데이트 이전 → EMA는 pruning 후 weight를 추적

### 3.2 L2 Norm 기반 중요도 측정

```python
# Conv2d: 출력 필터(output channel) 기준
weight = conv.weight      # shape: [out_ch, in_ch, kH, kW]
norms  = torch.norm(weight.view(out_ch, -1), dim=1)   # [out_ch]

# Linear: 출력 행(row) 기준
weight = linear.weight    # shape: [out_features, in_features]
norms  = torch.norm(weight, dim=1)                    # [out_features]
```

하위 `num_pruning`개 인덱스 선택:
```python
num_pruning = round(num_filters * sparsity)
_, prune_idx = torch.topk(norms, num_pruning, largest=False)
```

### 3.3 마스킹

```python
# Conv2d 출력 필터 마스킹 (BN 포함 필수)
with torch.no_grad():
    conv.weight.data[prune_idx] = 0.0
    bn.weight.data[prune_idx]   = 0.0
    bn.bias.data[prune_idx]     = 0.0
    bn.running_mean[prune_idx]  = 0.0
    bn.running_var[prune_idx]   = 1.0   # ← 0이면 분모 0 문제 발생!
```

> **BN도 반드시 같이 마스킹해야 한다.** Conv를 0으로 해도 BN이 살아있으면  
> BN이 학습되면서 정보를 살려낼 수 있어 reducing 후 불일치 발생.

---

## 4. Phase 2 — Reducing (학습 완료 후)

### 4.1 핵심 아이디어

```
Soft Pruning 완료 모델:
  Conv.weight[dead_ch] ≈ 0.0  (L2 norm ≈ 0)
  Conv.weight[live_ch] ≠ 0.0

Reducing:
  survived_idx = where(norm(conv.weight) > 0)
  new_conv = Conv2d(n_in, len(survived_idx), ...)  ← 작은 dense 레이어
  new_conv.weight = old_conv.weight[survived_idx]
```

### 4.2 Survived Index 추출

```python
def get_survived_idx(conv: nn.Conv2d) -> torch.Tensor:
    n = conv.weight.shape[0]
    norms = torch.norm(conv.weight.view(n, -1), dim=1)
    return torch.where(norms != 0)[0]
```

> **실용적 팁:** `!= 0` 대신 `> 1e-6` 임계값을 쓰는 경우도 있으나,  
> Soft Pruning은 정확히 0으로 마스킹하므로 `!= 0`이 더 정확하다.

### 4.3 Expand-Shrink 쌍 처리

FFN처럼 expand → shrink 구조는 **expand의 출력 인덱스 = shrink의 입력 인덱스**:

```python
survived = get_survived_idx(ffn.expand)   # expand 출력 기준

# expand: out_ch 축소
new_expand = Conv2d(in_ch, len(survived), ...)
new_expand.weight = old_expand.weight[survived]

# shrink: in_ch 축소 (출력 ch는 고정!)
new_shrink = Conv2d(len(survived), out_ch, ...)
new_shrink.weight = old_shrink.weight[:, survived]
```

### 4.4 Cascade 처리

레이어가 연쇄적으로 연결된 경우 (예: PE1 → PE2 → PE3 → PE4):

```python
s1 = get_survived_idx(pe1.conv)
s2 = get_survived_idx(pe2.conv)
s3 = get_survived_idx(pe3.conv)

# PE2: 입력 s1 축소 + 출력 s2 축소
new_pe2 = Conv2d(len(s1), len(s2), ...)
new_pe2.weight = pe2.weight[s2][:, s1]   # 출력 먼저, 입력 나중

# PE3: 입력 s2 + 출력 s3
# PE4: 입력 s3, 출력은 embed_dim 고정 → 건드리지 않음!
```

### 4.5 Reducing 후 검증

```python
with torch.no_grad():
    out = reduced_model(torch.zeros(1, 3, 224, 224))
assert out.shape == (1, num_classes)

B = original_param_count
A = sum(p.numel() for p in reduced_model.parameters())
rate = 100 * (B - A) / B
assert rate >= target_compression * 100, f"압축률 미달: {rate:.1f}%"
```

---

## 5. Sparsity 계산 — 이진탐색 방법

### 5.1 왜 이진탐색이 필요한가

단순 선형 계산:
```python
# 잘못된 방법
sparsity = target_compression * total_params / prunable_params
```

이 방법은 **secondary effect(이차 효과)를 무시**한다:
- Expand layer를 s% 제거 → shrink의 입력도 s% 제거 (연동)
- DWConv는 채널 수 = group 수 → hid 축소 시 자동 축소
- SE(Squeeze-Excite): hid 축소 → rd도 비례 축소 (2s - s² 비율)
- Chain 레이어: PE1 축소 → PE2 입력 측도 함께 축소

결과: 선형 계산은 **실제 압축률을 5~9% 과소 추정**하여 더 많이 압축됨.

### 5.2 이진탐색 구조

```python
def find_sparsity(model, target_compression, max_s=0.95):
    total         = count_total_params(model)
    target_remove = target_compression * total
    lo, hi        = 0.0, max_s

    for _ in range(64):          # 64회 반복 → 1e-19 정밀도
        mid = (lo + hi) / 2
        estimated = estimate_total_removed(model, mid)   # ← 이차 효과 포함
        if estimated < target_remove:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
```

### 5.3 estimate_total_removed 작성 원칙

새 아키텍처에 적용 시, 이 함수가 핵심:

```
estimate_total_removed(model, s) = Σ (각 prunable group의 실제 제거량)
```

각 그룹별 제거량 계산 시 포함해야 할 요소:

| 구조 | 제거량 계산 |
|------|------------|
| Conv (단독) | `p × (in_ch × kH × kW) + p × 2` (weight + BN) |
| Expand-Shrink 쌍 | expand rows + shrink cols (동일 p) |
| DWConv (groups=ch) | expand로부터 연동되므로 `p × (kH × kW + 2)` |
| SE | `old_rd × hid - new_rd × n` (이차, new_rd = round(n × old_rd / hid)) |
| Chain (PE1→PE2→...) | 입출력 양방향 축소 → `n_old × m_old - n_new × m_new` |

---

## 6. Prunable / Non-Prunable 레이어 분류 원칙

### Prunable (제거해도 안전한 레이어)

| 레이어 역할 | 이유 |
|------------|------|
| FFN Expand | 내부 hidden dim, 외부 채널과 무관 |
| Attention Q, K | QK^T 계산의 중간 dim, 외부 채널과 무관 |
| Conv chain (intermediate) | 최종 output dim이 고정이면 중간은 자유 |
| PatchMerging expand (1×1) | 내부 hidden dim |

### Non-Prunable (절대 건드리지 말 것)

| 레이어 역할 | 이유 |
|------------|------|
| **Output Projection** | 다음 레이어의 입력 채널과 일치 필요 |
| **Stage 경계 레이어** | blocks1 → blocks2 → blocks3 연결 채널 고정 |
| **DWConv (depthwise)** | groups = channels → 채널 변경 시 구조 파괴 |
| **Attention V** | proj(V) 출력 dim = embed_dim과 일치 필요 |
| **Classifier Head** | num_classes에 고정 |
| **최종 embed_dim 레이어** | 전체 채널 backbone에 영향 |

### 분류 방법 (새 아키텍처)

1. 모든 레이어를 열거하고 **입력/출력 채널이 외부 연결에 의존하는지** 확인
2. 양쪽 다 외부 연결이 있으면 Non-Prunable
3. 한쪽(주로 내부 hidden)이 자유로우면 Prunable

```python
# 구조 파악용 코드
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"{name}: {module.weight.shape}")
```

---

## 7. 연동 Pruning — coupled 레이어 처리

### 원칙: 차원이 맞닿는 레이어는 반드시 동일 인덱스 사용

**FFN (Expand-Shrink)**:
```
expand.out_ch [idx] → shrink.in_ch [idx]
→ expand row pruning 후, 반드시 shrink column도 동일 idx로 pruning
```

**Attention Q-K**:
```
Q.out_dim [idx] → K.out_dim [동일 idx] (QK^T 계산)
→ Q 기준으로 idx 계산 → K에 동일 idx 강제 적용
→ 각각 독립적으로 idx 계산하면 QK^T 차원 불일치로 런타임 에러!
```

**PatchEmbed Chain**:
```
PE1.out [idx1] → PE2.in [idx1], PE2.out [idx2] → PE3.in [idx2], ...
→ 각 레이어의 out idx는 독립적이나, 다음 레이어의 in은 반드시 앞의 out idx를 따름
```

**DWConv (연동)**:
```
DWConv의 채널 수 = groups → hid 축소 시 DWConv도 동일 idx로 축소
```

---

## 8. 주의사항 및 흔한 실수

### ⚠️ BN running_var를 1.0으로 설정

```python
# 잘못된 코드
bn.running_var[idx] = 0.0   # ← 나중에 BN 정규화 시 분모가 0

# 올바른 코드
bn.running_var[idx] = 1.0   # ← BN(x) = (x - 0) / sqrt(1 + eps) = x/(1+eps) ≈ x
```

### ⚠️ int() 대신 round() 사용

```python
# 소규모 레이어에서 문제 발생
num_prune = int(16 * 0.30)    # = 4 (4.8 버림)
num_prune = round(16 * 0.30)  # = 5 (더 정확)

# 최소 생존 채널 보장
num_prune = min(num_prune, total_ch - MIN_SURVIVE)
# MIN_SURVIVE = 4 권장 (너무 작으면 정보 손실, 너무 크면 압축률 저하)
```

### ⚠️ Q-K 인덱스 반드시 동일하게

```python
# 잘못된 코드 (각각 독립 계산)
q_idx = topk_smallest(q_norms, k)
k_idx = topk_smallest(k_norms, k)   # ← 절대 안 됨!

# 올바른 코드 (Q 기준으로 K 강제)
q_idx = topk_smallest(q_norms, k)
k_idx = q_idx   # 또는 q_idx + key_dim (같은 weight tensor 내 offset)
```

### ⚠️ DDP (DistributedDataParallel) 처리

```python
# DDP로 래핑된 경우 실제 모델은 .module 안에 있음
actual_model = model.module if hasattr(model, 'module') else model
pruner.apply(actual_model)  # ← .module 전달
```

### ⚠️ Reducing 후 state_dict 저장 방식

```python
# 방법 1: state_dict만 저장 (나중에 아키텍처 재현 필요)
torch.save({'model': model.state_dict()}, path)

# 방법 2: full model 저장 (즉시 사용 가능)
torch.save(model, full_path)    # torch.load()로 바로 사용

# Reducing 전 soft-pruned state_dict로도 복원 가능하게 하려면:
# → soft-pruned ckpt 로드 → reducing 적용 순서로 재현
```

### ⚠️ Sparsity 상한 클리핑

```python
# QK는 채널이 매우 작으므로 너무 높은 sparsity 금지
sparsity_qk = min(sparsity, 0.90)   # 90% 이상이면 key_dim이 너무 작아짐
sparsity_ffn = min(sparsity, 0.95)  # FFN은 상대적으로 안전
```

### ⚠️ `torch.no_grad()` 필수

```python
# 마스킹은 반드시 no_grad 안에서 수행
with torch.no_grad():
    conv.weight.data[idx] = 0.0
    # .data 접근 또는 no_grad 컨텍스트 중 하나는 반드시 필요
```

### ⚠️ 압축률 계산 공식

```python
# 파라미터 수 기준 (파일 크기 기준이 아님!)
B = original_param_count      # 원본
A = reduced_param_count        # 축소 후
compression_rate = (B - A) / B * 100   # 76% = 76% 제거됨
```

---

## 9. 새 아키텍처 적용 체크리스트

### Step 1: 아키텍처 분석

- [ ] 모든 레이어 이름/shape 출력 (`model.named_modules()`)
- [ ] 각 레이어의 입/출력 채널이 외부(다른 레이어)와 연결되는지 확인
- [ ] Prunable 그룹 목록 작성 (FFN-like, Attention QK, Intermediate Conv 등)
- [ ] 절대 Prunable 불가 레이어 목록 작성 (output proj, stage boundary 등)
- [ ] Coupled 관계 목록 작성 (expand-shrink 쌍, Q-K 쌍, chain 등)

### Step 2: estimate_total_removed 구현

- [ ] 각 Prunable 그룹의 제거량 계산 함수 구현
- [ ] Secondary effect (SE, DWConv, cascade) 포함 여부 확인
- [ ] `s=0`, `s=0.5`, `s=0.95`로 sanity check

### Step 3: Pruning 함수 구현

- [ ] `get_pruning_idx(layer, sparsity)` → L2 norm 하위 topk 인덱스
- [ ] 각 Prunable 그룹별 `_prune_XXX(module, sparsity)` 구현
- [ ] Coupled 레이어 연동 확인 (동일 인덱스 사용)
- [ ] BN 마스킹 포함 여부 확인

### Step 4: Pruner 클래스

- [ ] `__init__`: 이진탐색으로 sparsity 계산, 로그 출력
- [ ] `apply(model)`: 전체 모델에 pruning 적용
- [ ] `log_sparsity(model)`: 실제 zero 비율 반환 (검증용)

### Step 5: engine.py / 학습 루프 수정

- [ ] `pruner=None` 인자 추가
- [ ] `optimizer.step()` 직후 `pruner.apply(model)` 삽입
- [ ] DDP 환경이면 `model.module` 전달 처리

### Step 6: Reducing 함수 구현

- [ ] `get_survived_idx(layer)` → norm > 0 인덱스
- [ ] 각 그룹별 `_reduce_XXX(module)` in-place 구현
- [ ] Chain 레이어의 cascade 순서 확인 (앞에서 뒤로)
- [ ] Forward pass 검증 (`torch.zeros(1, C, H, W)`)
- [ ] 압축률 검증 (`rate >= target`)

### Step 7: 검증

- [ ] Epoch 1 후 실제 zero 비율 확인 (`log_sparsity`)
- [ ] Epoch 10 후 정확도 확인 (너무 급격히 하락하면 sparsity 조정)
- [ ] Reducing 후 forward pass OK
- [ ] Reducing 후 압축률 target 달성

---

## 10. EfficientViT 구현 요약 (레퍼런스)

### 파일 구조

```
classification/pruning/
├── efficientvit_pruning.py   ← Soft Pruning (학습 중)
└── efficientvit_reducing.py  ← Dense 변환 (학습 후)
```

### Prunable 그룹 (M4 기준)

| 그룹 | 레이어 | 파라미터 비중 |
|------|--------|--------------|
| G_FFN | FFN pw1(expand) + pw2(shrink) | ~67% |
| G_INV | PatchMerging conv1 + conv3 | ~10% |
| G_QK | CGA qkvs Q+K + dws | ~1% |
| G_PE | PatchEmbed PE1, PE2, PE3 | ~0.3% |

### 이진탐색 결과 (M4)

| target | bisection sparsity | 실제 압축률 |
|--------|-------------------|------------|
| 30% | ≈ 0.295 | ≈ 30% |
| 50% | ≈ 0.505 | ≈ 50% |
| 70% | ≈ 0.750 | ≈ 70% |
| 75% | ≈ 0.820 | ≈ 75% |

### 학습 명령어

```bash
python main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/.../imagenet \
  --resume /workspace/.../efficientvit_m4.pth \
  --pruning --target-compression 0.30 \
  --output_dir /workspace/.../output/pruning_30pct \
  --device cuda:0
```

### Reducing 명령어

```bash
python -m classification.pruning.efficientvit_reducing \
  --checkpoint /workspace/.../output/pruning_30pct/checkpoint_best.pth \
  --output /workspace/.../reduced_m4_30pct.pth
```

---

*문서 작성 기준: EfficientViT M4 구현 (2026-04)*
