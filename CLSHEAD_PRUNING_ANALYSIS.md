# ClsHead Pruning 상세 분석

> EfficientViT-B1 기준 ClsHead 아키텍처 분석, Prunable 그룹 정의,
> 마스킹/Reducing 로직, 파라미터 추정식, 위험도 분석을 포함한다.
> PRUNING_IMPLEMENTATION_REPORT.md §12.1 의 확장 문서.

---

## 1. 왜 ClsHead를 Pruning해야 하는가

PRUNING_IMPLEMENTATION_REPORT.md §8.5 실측 기준:

```
group        #mod         numel          MB        %
----------------------------------------------------
G_STEM          1         1,027       0.004    0.01%
G_HEAD          1     4,461,161      17.845   48.94%
G_LITEMLA       7     1,735,303       6.941   19.04%
G_MBCONV       14     2,917,976      11.672   32.01%
```

**Head가 전체 파라미터의 48.94%** 를 차지한다. 현재 구현(backbone-only pruning)에서
G_MBCONV + G_STEM(= 32.02%)만 대상으로 하면 나머지 67.98%는 그대로 남는다.
target_compression=0.30 이라도 실질적으로는 prunable 32%의 94%만 제거되므로
전체 대비 실제 제거율은 ~30%에 그친다.

Head까지 포함하면 prunable 영역이 ~80.96%로 늘어나 동일 sparsity에서
훨씬 효율적인 압축이 가능하다.

---

## 2. ClsHead 아키텍처

### 2.1 클래스 정의

파일: [`efficientvit/models/efficientvit/cls.py`](efficientvit/models/efficientvit/cls.py)

```python
class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,       # backbone 출력 채널 (B1: 256)
        width_list: list[int],  # [hidden0, hidden1] (B1: [1536, 1600])
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)
        self.fid = fid

    def forward(self, feed_dict):
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)
```

### 2.2 B1 모델 기준 op_list 구성

모델: `efficientvit-b1` / `in_channels=256` / `width_list=[1536, 1600]` / `n_classes=1000`

| 인덱스 | 레이어 타입 | in | out | 커널 | norm | act |
|--------|-----------|-----|-----|-----|------|-----|
| `op_list[0]` | `ConvLayer` | 256 | **1536** | 1×1 | BN2d | HSwish |
| `op_list[1]` | `AdaptiveAvgPool2d` | 1536 | 1536 | — | — | — |
| `op_list[2]` | `LinearLayer` | **1536** | **1600** | — | LayerNorm | HSwish |
| `op_list[3]` | `LinearLayer` | **1600** | 1000 | — | None | None |

모델 접근 경로: `model.head.op_list[0~3]`

### 2.3 텐서 흐름 (B1, 입력 224×224)

```
backbone 출력
  feed_dict['stage_final'] = (B, 256, 7, 7)
          ↓
[op_list[0]] ConvLayer(256 → 1536, k=1)
  conv.weight:  (1536, 256, 1, 1)
  BN2d(1536)
  HSwish
  출력: (B, 1536, 7, 7)
          ↓
[op_list[1]] AdaptiveAvgPool2d(1)
  출력: (B, 1536, 1, 1)      ← 공간 정보 제거, 채널 순서 보존
          ↓
[op_list[2]] LinearLayer(1536 → 1600, bias=False)
  _try_squeeze: (B, 1536, 1, 1) → (B, 1536)   ← 채널 i = feature i
  linear.weight: (1600, 1536)
  LayerNorm(1600)
  HSwish
  출력: (B, 1600)
          ↓
[op_list[3]] LinearLayer(1600 → 1000, bias=True)
  linear.weight: (1000, 1600)
  linear.bias:   (1000,)
  출력: (B, 1000)   ← 최종 logits, n_classes 고정
```

### 2.4 파라미터 수 정확한 분해 (B1)

| 레이어 | 구성 요소 | 파라미터 수 |
|--------|---------|------------|
| `op_list[0].conv` | Conv2d(256, 1536, 1): weight | 256 × 1536 = **393,216** |
| `op_list[0].norm` | BN2d(1536): weight + bias | 1536 × 2 = **3,072** |
| `op_list[0].norm` | BN2d(1536): running_mean + running_var (buffer) | 1536 × 2 = **3,072** |
| `op_list[2].linear` | Linear(1536, 1600, bias=False): weight | 1536 × 1600 = **2,457,600** |
| `op_list[2].norm` | LayerNorm(1600): weight + bias | 1600 × 2 = **3,200** |
| `op_list[3].linear` | Linear(1600, 1000, bias=True): weight | 1600 × 1000 = **1,600,000** |
| `op_list[3].linear` | Linear(1600, 1000): bias | **1,000** |
| **합계 (학습 파라미터)** | | **4,458,088** |
| **합계 (buffer 포함)** | | **4,461,160** ≈ 측정값 4,461,161 ✓ |

---

## 3. Prunable 그룹 정의

ClsHead에서 Prunable한 "내부 채널"은 두 그룹이다.
외부 채널(`in_channels=256`, `n_classes=1000`)은 backbone/태스크와 직결되므로 고정.

### G_HEAD0 — 1×1 Conv 확장 그룹 (dim: 1536)

```
op_list[0].conv.weight  (1536, 256, 1, 1)   ← 출력 필터 (L2 norm 기준 인덱스 산정)
    ↓ 같은 인덱스
op_list[0].norm  BN2d(1536)
    ↓ 채널 순서 보존 (AdaptiveAvgPool2d + _try_squeeze)
op_list[2].linear.weight  (1600, 1536)      ← 입력 컬럼 ([:, idx])
```

**인덱스 산정 기준**: `op_list[0].conv.weight` 출력 필터 L2 norm 하위 k개.

### G_HEAD1 — FC 히든 그룹 (dim: 1600)

```
op_list[2].linear.weight  (1600, 1536)      ← 출력 행 (L2 norm 기준 인덱스 산정)
    ↓ 같은 인덱스
op_list[2].norm  LayerNorm(1600)
    ↓ 같은 인덱스
op_list[3].linear.weight  (1000, 1600)      ← 입력 컬럼 ([:, idx])
```

**인덱스 산정 기준**: `op_list[2].linear.weight` 출력 행 L2 norm 하위 k개.

### Non-Prunable 이유

| 요소 | 이유 |
|------|------|
| `op_list[0]` 입력 (256채널) | backbone `stage_final` 출력 채널 = 외부 hyperparameter |
| `op_list[3]` 출력 (1000채널) | n_classes = 태스크 고정값 |
| `AdaptiveAvgPool2d` | 파라미터 없음, 채널 순서만 전달 |

### Coupled 요약

```
G_HEAD0 인덱스 적용 위치
  op_list[0].conv.weight[idx]          (출력 필터 마스킹)
  op_list[0].norm  BN2d[idx]           (BN weight, bias, running_mean, running_var)
  op_list[2].linear.weight[:, idx]     (입력 컬럼 마스킹)

G_HEAD1 인덱스 적용 위치
  op_list[2].linear.weight[idx]        (출력 행 마스킹)
  op_list[2].norm  LayerNorm[idx]      (LN weight, bias)
  op_list[3].linear.weight[:, idx]     (입력 컬럼 마스킹)
  ※ op_list[3].linear.bias 는 출력(=n_classes)에 속하므로 건드리지 않음
```

---

## 4. AdaptiveAvgPool2d가 채널 순서를 보존하는 이유

```python
# AdaptiveAvgPool2d: (B, C, H, W) → (B, C, 1, 1)
# 공간 차원(H, W)만 평균 풀링. 채널 축은 독립적으로 처리.
# 채널 i 의 값: mean(input[:, i, :, :])  — 다른 채널과 섞이지 않음.

# LinearLayer._try_squeeze: (B, C, 1, 1) → (B, C)
# torch.flatten(x, start_dim=1) = reshape(B, C*1*1) = reshape(B, C)
# 채널 i → feature position i 로 1:1 매핑.
```

따라서 `op_list[0].conv`의 출력 채널 인덱스와
`op_list[2].linear`의 입력 feature 인덱스가 **정확히 일치**한다.
G_HEAD0 마스킹 시 동일 `idx`를 양쪽에 적용해도 안전.

---

## 5. 마스킹 로직 (Soft Pruning)

### 5.1 BN vs LayerNorm 처리

기존 `_zero_bn_` 함수가 두 경우를 모두 처리한다.

```python
def _zero_bn_(bn, idx):
    if bn is None:
        return
    with torch.no_grad():
        if getattr(bn, "weight", None) is not None:
            bn.weight.data[idx] = 0.0       # BN 과 LN 모두 weight 있음
        if getattr(bn, "bias", None) is not None:
            bn.bias.data[idx] = 0.0         # BN 과 LN 모두 bias 있음
        if isinstance(bn, _BatchNorm):      # BN 만 running stats 있음
            bn.running_mean.data[idx] = 0.0
            bn.running_var.data[idx] = 1.0  # 0 이면 분모 0 위험
```

- `BN2d` (op_list[0].norm): running_mean/var 도 처리됨 ✓
- `LayerNorm` (op_list[2].norm): `_BatchNorm` 이 아니므로 weight/bias 만 처리됨 ✓
- 추가 함수 불필요, 기존 helper 재사용 가능.

### 5.2 G_HEAD0 마스킹 의사 코드

```python
def _prune_head0(head: ClsHead, sparsity: float) -> None:
    weight = head.op_list[0].conv.weight       # (1536, 256, 1, 1)
    n_total = weight.shape[0]                  # 1536
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)

    with torch.no_grad():
        # op_list[0]: 출력 필터 + BN
        head.op_list[0].conv.weight.data[idx] = 0.0
        _zero_bn_(head.op_list[0].norm, idx)   # BN2d

        # op_list[2]: 입력 컬럼
        head.op_list[2].linear.weight.data[:, idx] = 0.0
```

### 5.3 G_HEAD1 마스킹 의사 코드

```python
def _prune_head1(head: ClsHead, sparsity: float) -> None:
    weight = head.op_list[2].linear.weight     # (1600, 1536)
    n_total = weight.shape[0]                  # 1600
    n_prune = _calc_n_prune(n_total, sparsity)
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(weight, n_prune)

    with torch.no_grad():
        # op_list[2]: 출력 행 + LayerNorm
        head.op_list[2].linear.weight.data[idx] = 0.0
        _zero_bn_(head.op_list[2].norm, idx)   # LayerNorm (weight/bias만)

        # op_list[3]: 입력 컬럼 (bias는 출력=n_classes 소속, 건드리지 않음)
        head.op_list[3].linear.weight.data[:, idx] = 0.0
```

### 5.4 EfficientViTPruner.apply() 연동

```python
def apply(self, model: nn.Module) -> None:
    if self.sparsity <= 0:
        return
    # 기존 backbone pruning
    for kind, mod in _iter_prunable_modules(model):
        ...
    _prune_input_stem(model, self.sparsity)

    # ★ 신규: head pruning
    head = getattr(model, "head", None)
    if head is not None:
        _prune_head0(head, self.sparsity)
        _prune_head1(head, self.sparsity)
```

### 5.5 모델 접근 경로 주의사항

DDP 학습 시 `pruner.apply(self.network)` 에서 `self.network` 는
`model.module` (DDP wrapper 벗긴 raw 모델). `model.module.head.op_list[0]`
처럼 접근해야 하며, 기존 코드에서 `_get_stem_op_seq(model)` 이 이미
`model.backbone` 으로 접근하는 것과 동일 패턴이라 호환됨.

---

## 6. Reducing 로직

### 6.1 G_HEAD0 Reducing

```python
def _reduce_head0(head: ClsHead) -> None:
    weight = head.op_list[0].conv.weight           # (1536, 256, 1, 1)
    n_filt = weight.shape[0]
    norms = torch.norm(weight.detach().reshape(n_filt, -1), dim=1)
    survived = torch.where(norms != 0)[0]
    n_new = len(survived)
    if n_new == n_filt:
        return

    old_conv = head.op_list[0].conv
    # Conv2d(256 → n_new, 1, 1)
    new_conv = nn.Conv2d(
        old_conv.in_channels, n_new, 1,
        bias=old_conv.bias is not None
    )
    new_conv.weight.data = old_conv.weight.data[survived]
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data[survived]
    head.op_list[0].conv = new_conv

    # BN2d(n_new)
    old_bn = head.op_list[0].norm
    new_bn = nn.BatchNorm2d(n_new, eps=old_bn.eps, momentum=old_bn.momentum)
    new_bn.weight.data    = old_bn.weight.data[survived]
    new_bn.bias.data      = old_bn.bias.data[survived]
    new_bn.running_mean   = old_bn.running_mean[survived]
    new_bn.running_var    = old_bn.running_var[survived]
    head.op_list[0].norm  = new_bn

    # Linear(n_new → 1600) — 입력 컬럼만 축소, 출력 / LN 은 그대로
    old_linear = head.op_list[2].linear
    new_linear = nn.Linear(n_new, old_linear.out_features, bias=old_linear.bias is not None)
    new_linear.weight.data = old_linear.weight.data[:, survived]
    if old_linear.bias is not None:
        new_linear.bias.data = old_linear.bias.data
    head.op_list[2].linear = new_linear
```

### 6.2 G_HEAD1 Reducing

```python
def _reduce_head1(head: ClsHead) -> None:
    weight = head.op_list[2].linear.weight         # (1600, 1536 or n_new after head0)
    n_filt = weight.shape[0]
    norms = torch.norm(weight.detach(), dim=1)
    survived = torch.where(norms != 0)[0]
    n_new = len(survived)
    if n_new == n_filt:
        return

    old_linear2 = head.op_list[2].linear
    # Linear(현재_in → n_new)
    new_linear2 = nn.Linear(old_linear2.in_features, n_new, bias=old_linear2.bias is not None)
    new_linear2.weight.data = old_linear2.weight.data[survived]
    if old_linear2.bias is not None:
        new_linear2.bias.data = old_linear2.bias.data[survived]
    head.op_list[2].linear = new_linear2

    # LayerNorm(n_new)
    old_ln = head.op_list[2].norm
    new_ln = nn.LayerNorm(n_new, eps=old_ln.eps)
    new_ln.weight.data   = old_ln.weight.data[survived]
    new_ln.bias.data     = old_ln.bias.data[survived]
    head.op_list[2].norm = new_ln

    # Linear(n_new → 1000) — 입력 컬럼만 축소, bias / 출력 1000 은 그대로
    old_linear3 = head.op_list[3].linear
    new_linear3 = nn.Linear(n_new, old_linear3.out_features, bias=old_linear3.bias is not None)
    new_linear3.weight.data = old_linear3.weight.data[:, survived]
    if old_linear3.bias is not None:
        new_linear3.bias.data = old_linear3.bias.data  # bias는 출력(1000) 소속, 유지
    head.op_list[3].linear = new_linear3
```

### 6.3 Reducing 순서 주의

**반드시 G_HEAD0 → G_HEAD1 순서로 실행해야 한다.**

G_HEAD0 을 먼저 Reduce 하면 `op_list[2].linear.in_features` 가
1536 → n_new_h0 로 변한다. G_HEAD1 의 `_reduce_head1` 은
`op_list[2].linear.weight.shape[1]` (= n_new_h0) 를 그대로 사용하므로
순서가 바뀌어도 수치는 맞지만, 개념적으로 G_HEAD0 → G_HEAD1 순서가 자연스럽다.

```python
def reduce_head(model: nn.Module) -> None:
    head = getattr(model, "head", None)
    if head is None:
        return
    _reduce_head0(head)   # 1st: 1536 차원 축소
    _reduce_head1(head)   # 2nd: 1600 차원 축소
```

---

## 7. 파라미터 추정식 (이진탐색용)

### 7.1 G_HEAD0

```
n_prune_h0 = _calc_n_prune(1536, sparsity)

removed_h0
    += n_prune_h0 * 256                     # op_list[0] Conv 출력 필터 (in=256)
    += n_prune_h0 * 2    (BN weight+bias)   # op_list[0] BN
    += n_prune_h0 * 1600                    # op_list[2] Linear 입력 컬럼 (out=1600)
    ※ cross-term(-n_prune_h0 * n_prune_h1) 무시 → 약간 over-estimate (안전 방향)
```

### 7.2 G_HEAD1

```
n_prune_h1 = _calc_n_prune(1600, sparsity)

removed_h1
    += n_prune_h1 * 1536                    # op_list[2] Linear 출력 행 (현재_in=1536)
    += n_prune_h1 * 2    (LN weight+bias)   # op_list[2] LayerNorm
    += n_prune_h1 * 1000                    # op_list[3] Linear 입력 컬럼 (out=1000)
    ※ op_list[3] bias(1000)는 출력 소속이라 제거 대상 아님
```

### 7.3 전체 합산 (기존 코드 수정 위치)

```python
def _estimate_total_removed(model, sparsity):
    total = 0
    for kind, mod in _iter_prunable_modules(model):
        ...
    total += _estimate_removed_input_stem(model, sparsity)
    # ★ 신규
    head = getattr(model, "head", None)
    if head is not None:
        total += _estimate_removed_head0(head, sparsity)
        total += _estimate_removed_head1(head, sparsity)
    return total
```

### 7.4 B1 기준 per-group sparsity 재추정

Head 포함 시 prunable 파라미터가 크게 늘어나므로
동일 target_compression에서 per-group sparsity가 낮아진다.

| target | 기존 per-group sparsity | Head 포함 시 예상 |
|--------|------------------------|-----------------|
| 0.10   | ≈ 0.31                 | ≈ 0.15~0.18     |
| 0.15   | ≈ 0.47                 | ≈ 0.22~0.27     |
| 0.20   | ≈ 0.63                 | ≈ 0.30~0.36     |
| 0.30   | ≈ 0.93~0.94            | ≈ 0.47~0.55     |

> 이 수치는 추정이며, 이진탐색이 정확한 값을 계산한다.
> 실제 실행 시 `[EfficientViTPruner] per-group sparsity=X.XXXX` 로 확인.

---

## 8. 위험도 및 주의사항

### 8.1 Head는 왜 더 위험한가

| 항목 | Backbone MBConv | ClsHead |
|------|----------------|---------|
| 역할 | 중간 표현 학습 | 직접 logits 생성 |
| redundancy | FFN hidden은 overparameterized | FC 행은 상대적으로 적음 |
| 회복 가능성 | fine-tuning으로 잘 회복 | 분류 직전 레이어라 민감 |

### 8.2 높은 sparsity에서의 정확도 리스크

- G_HEAD1의 1600→1000 Linear는 압축 비율이 이미 낮다 (1.6배).
  여기서 1600을 800으로 줄이면 800→1000 Linear가 되어 over-complete가 아닌
  under-complete 매핑이 된다. → 표현력 급감 위험.
- 권장: G_HEAD1 sparsity 상한을 `MIN_SURVIVE / 1600 ≈ 0.4` 이하로 제한하거나,
  G_HEAD1만 별도 낮은 sparsity를 적용.

### 8.3 G_HEAD0는 상대적으로 안전

- 256 → 1536 확장은 6배 overparameterized.
- 1536을 768로 줄여도 3배 확장으로 충분한 표현력 유지 가능.
- Backbone MBConv의 FFN과 동일한 성질.

### 8.4 별도 sparsity 적용 옵션

현재 구현은 모든 그룹에 동일 sparsity를 적용한다.
Head를 별도로 제어하려면:

```python
class EfficientViTPruner:
    def __init__(self, model, target_compression, max_sparsity=0.95,
                 head_sparsity_scale=0.5):  # ← head sparsity 를 backbone의 절반으로
        ...
        self.head_sparsity = self.sparsity * head_sparsity_scale
```

이 경우 이진탐색도 분리하거나 scale factor를 반영해야 한다.
**1차 구현에서는 동일 sparsity를 권장** — 학습 후 실제 정확도를 확인하고 조정.

### 8.5 LayerNorm running stats 없음

BN과 달리 LayerNorm은 running stats가 없다.
`_zero_bn_` 의 `isinstance(bn, _BatchNorm)` 분기가 자동으로 건너뛰므로
추가 처리 불필요. 기존 코드 재사용 가능.

---

## 9. log_sparsity 연동

```python
@torch.no_grad()
def log_sparsity(self, model):
    ...  # 기존 backbone 집계

    # ★ head 집계 추가
    head = getattr(model, "head", None)
    if head is not None:
        # G_HEAD0: op_list[0] 출력 필터
        w0 = head.op_list[0].conv.weight
        n0 = w0.shape[0]
        norms0 = torch.norm(w0.detach().reshape(n0, -1), dim=1)
        n_total += n0
        n_zero  += int((norms0 == 0).sum().item())

        # G_HEAD1: op_list[2] 출력 행
        w1 = head.op_list[2].linear.weight
        n1 = w1.shape[0]
        norms1 = torch.norm(w1.detach(), dim=1)
        n_total += n1
        n_zero  += int((norms1 == 0).sum().item())

    return {
        "prunable_filters": n_total,
        "zero_filters": n_zero,
        "actual_sparsity": n_zero / max(n_total, 1),
        "target_sparsity": self.sparsity,
    }
```

---

## 10. 구현 체크리스트

| 항목 | 파일 | 변경 내용 |
|------|------|---------|
| `_prune_head0` 추가 | `efficientvit_pruning.py` | 신규 함수 |
| `_prune_head1` 추가 | `efficientvit_pruning.py` | 신규 함수 |
| `_estimate_removed_head0` 추가 | `efficientvit_pruning.py` | 신규 함수 |
| `_estimate_removed_head1` 추가 | `efficientvit_pruning.py` | 신규 함수 |
| `EfficientViTPruner.apply()` 수정 | `efficientvit_pruning.py` | head 호출 추가 (~3줄) |
| `_estimate_total_removed()` 수정 | `efficientvit_pruning.py` | head 호출 추가 (~3줄) |
| `EfficientViTPruner.log_sparsity()` 수정 | `efficientvit_pruning.py` | head 집계 추가 (~10줄) |
| `_reduce_head0` 추가 | `efficientvit_reducing.py` | 신규 함수 |
| `_reduce_head1` 추가 | `efficientvit_reducing.py` | 신규 함수 |
| `reduce_efficientvit_cls_model()` 수정 | `efficientvit_reducing.py` | `reduce_head(model)` 호출 추가 (~3줄) |
| `ClsHead` import 추가 | `efficientvit_pruning.py` / `reducing.py` | `from efficientvit.models.efficientvit.cls import ClsHead` |

---

## 11. 모델별 ClsHead 파라미터 비교

| 모델 | in_ch | width_list | head params | 전체 대비 |
|------|-------|-----------|------------|---------|
| B0 | 128 | [1024, 1280] | ~2.7M | ~52% |
| **B1** | **256** | **[1536, 1600]** | **~4.46M** | **~49%** |
| B2 | 384 | [2304, 2560] | ~9.6M | ~45% |
| B3 | 512 | [2304, 2560] | ~10.5M | ~38% |
| L1 | 512 | [3072, 3200] | ~19.5M | ~49% |
| L2 | 512 | [3072, 3200] | ~19.5M | ~42% |
| L3 | 1024 | [6144, 6400] | ~78M | ~55% |

모든 모델에서 head 비율이 매우 높으며, 구현 로직은 `in_channels` / `width_list` 를
동적으로 읽으므로 **B/L 시리즈 공통 사용 가능**.

---

*작성일: 2026-05-01. 기준 모델: EfficientViT-B1. 참고: PRUNING_IMPLEMENTATION_REPORT.md §12.1.*
