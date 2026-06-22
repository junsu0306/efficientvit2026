# EfficientViT Soft Pruning 프로젝트 보고서

> **작성 기준**: 2026년 6월 / 프로젝트 기간: 2026년 4월 ~ 2026년 6월  
> **담당**: JunsuHA  
> **목적**: 이 문서는 EfficientViT 분류 모델에 Soft Pruning을 적용하는 전 과정을 기록한 최종 보고서다.  
> 진행한 내용, 설계 근거, 구현 결과, 얻으려 했던 것을 담는다.

---

## 목차

1. [프로젝트 배경 및 목적](#1-프로젝트-배경-및-목적)
2. [대상 모델: EfficientViT](#2-대상-모델-efficientvit)
3. [방법론: Soft Pruning](#3-방법론-soft-pruning)
4. [아키텍처 분석 및 Prunable 그룹 정의](#4-아키텍처-분석-및-prunable-그룹-정의)
5. [구현 전체 구조](#5-구현-전체-구조)
6. [핵심 구현 상세](#6-핵심-구현-상세)
7. [훈련 루프 연동 및 WandB](#7-훈련-루프-연동-및-wandb)
8. [Reducing — Dense 변환](#8-reducing--dense-변환)
9. [메모리 프로파일링](#9-메모리-프로파일링)
10. [NPU 배포 대응](#10-npu-배포-대응)
11. [측정 기준 수치](#11-측정-기준-수치)
12. [실험 설계 및 목표 결과](#12-실험-설계-및-목표-결과)
13. [주요 의사결정 정리](#13-주요-의사결정-정리)
14. [구현 타임라인 (Changelog)](#14-구현-타임라인-changelog)
15. [향후 확장 가능 항목](#15-향후-확장-가능-항목)

---

## 1. 프로젝트 배경 및 목적

### 1.1 배경

EfficientViT는 경량 Vision Transformer로, 모바일/엣지 디바이스를 위해 설계된 모델이다.  
본 프로젝트는 이 모델을 **NPU(Neural Processing Unit) 에 배포**하는 것을 최종 목표로 한다.  
배포를 위해서는 모델 크기와 추론 속도를 줄이는 작업이 필요했으며, 그 수단으로 **구조적 프루닝(Structural Pruning)** 을 선택했다.

### 1.2 목적

| 목적 | 설명 |
|------|------|
| **파라미터 압축** | 원본 모델 대비 10~70% 파라미터 감소 |
| **정확도 보존** | 압축 후에도 baseline 대비 최소한의 정확도 하락 |
| **NPU 호환 모델 생성** | Reducing 완료 모델을 NPU 컴파일러(`qbcompiler`)에 입력 가능한 형태로 저장 |
| **방법론 일반화** | 동일한 방법론을 다른 아키텍처(ViT, DeiT 등)에 이식 가능하도록 문서화 |

### 1.3 선택 방법론 근거

| 후보 방법 | 선택 여부 | 이유 |
|---------|----------|------|
| Knowledge Distillation | 미채택 | Teacher 모델 필요, 파이프라인 복잡 |
| Quantization | 별도 (NPU 컴파일러 담당) | qbcompiler가 이미 INT8 양자화 수행 |
| Hard Pruning | 미채택 | 학습 중 아키텍처 변경 → 복잡한 파이프라인 |
| **Soft Pruning** | **✅ 채택** | 학습 파이프라인 최소 변경, Fine-tuning과 동시 수행 |

---

## 2. 대상 모델: EfficientViT

### 2.1 모델 계열

프로젝트에서 다룬 모델은 EfficientViT **B/L 시리즈 분류 모델**이다.  
메인 실험 기준 모델은 **EfficientViT-B1 (224×224)** 이며, B2 모델도 실험 명령어까지 준비되었다.

| 계열 | 모델명 | 파라미터 | 주요 블록 |
|------|--------|---------|---------|
| B | b0, b1, b2, b3 | 3.4M ~ 33.7M | MBConv (inverted + DW + point) |
| L | l1, l2, l3 | 53M ~ 246M | FusedMBConv (spatial + point) |

### 2.2 아키텍처 개요 (B-series 기준)

```
입력 (3, H, W)
    ↓
input_stem: Conv(3→C0, s=2) + DSConv(C0→C0) × n
    ↓
stages[0..1]: MBConv × depth  (채널: width_list[1], [2])
    ↓
stages[2..3]: MBConv downsample + EfficientViTBlock × depth
              EfficientViTBlock = LiteMLA(attention) + MBConv(local)
    ↓
ClsHead: Conv(256→1536) → AvgPool → Linear(1536→1600) → Linear(1600→1000)
    ↓
출력 logits (1000)
```

### 2.3 핵심 빌딩 블록 채널 흐름

| 블록 | 채널 흐름 | 내부 hidden |
|------|----------|------------|
| `MBConv` | in → mid (expand 1×1) → mid (DWConv) → out (shrink 1×1) | **mid_channels** ← Pruning 대상 |
| `FusedMBConv` | in → mid (k×k spatial) → out (shrink 1×1) | **mid_channels** ← Pruning 대상 |
| `LiteMLA` | multi-scale linear attention | 복잡한 결합, **이번 구현에서 제외** |
| `DSConv` | in → in (DW) → out (point) | 채널 수 고정 (입출력 동일) |

### 2.4 Pretrained Weight

| 모델 | 파일명 | ImageNet Top-1 (공식) |
|------|--------|----------------------|
| efficientvit-b1 | `efficientvit_b1_r224.pt` | 79.4% |
| efficientvit-b2 | `efficientvit_b2_r288.pt` | 82.7% |

---

## 3. 방법론: Soft Pruning

### 3.1 핵심 원리

Soft Pruning은 weight를 **완전히 삭제하지 않고 0으로 마스킹**한 채로 Fine-tuning을 계속하는 방식이다.

```
[일반 학습]     forward → loss → backward → optimizer.step()

[Soft Pruning]  forward → loss → backward → optimizer.step()
                                                      ↓
                                    L2 norm 하위 X% weight를 0으로 리셋
                                                      ↓
                                    (다음 step에서 gradient로 살아날 수도 있음)
                                                      ↓
                                    다시 0으로 리셋 → 반복 → 수렴
```

### 3.2 전체 파이프라인

```
[1] Pretrained weight 로드 (mit-han-lab 공식 checkpoint)
        ↓
[2] Soft Pruning Fine-tuning
    - 매 optimizer.step() 직후 pruner.apply() 호출
    - L2 norm 하위 X% 채널을 0으로 강제
    - EMA가 pruning 후 weight를 추적 (raw weight는 0이지만 EMA shadow에 성능 보존)
        ↓
[3] model_best.pt 저장 (원본 아키텍처 shape 그대로, 0인 채널 포함)
        ↓
[4] Reducing 실행 (학습 완료 후 한 번)
    - L2 norm == 0인 채널을 물리적으로 삭제
    - 실제로 작은 Dense 모델 생성
        ↓
[5] NPU Export
    - reduced 모델을 nn.Module 객체(.pt)로 저장
    - qbcompiler backend="torch"로 NPU 컴파일
```

### 3.3 Soft Pruning이 Hard Pruning보다 유리한 이유

| 특성 | Soft Pruning | Hard Pruning |
|------|-------------|-------------|
| 학습 중 아키텍처 변경 | 없음 | 있음 (복잡) |
| DDP 호환성 | 완전 호환 | 별도 처리 필요 |
| 기존 코드 침투 | optimizer.step() 직후 1줄 추가 | 학습 루프 대폭 수정 |
| 채널 복구 가능성 | 있음 (gradient로 살아날 수 있음) | 없음 |
| 최종 Dense 변환 | Reducing 1회 실행 | 학습 중 매 단계 |

### 3.4 L2 Norm 기반 중요도 측정

```python
# Conv2d 출력 필터 기준 (out_channels 축)
weight = conv.weight      # shape: (out_ch, in_ch, kH, kW)
norms  = torch.norm(weight.view(out_ch, -1), dim=1)   # (out_ch,)

# 하위 num_pruning개 인덱스 선택
num_pruning = round(num_filters * sparsity)   # int() 아닌 round() — 소규모 편향 방지
_, prune_idx = torch.topk(norms, num_pruning, largest=False)
```

**왜 L2 Norm인가**: 채널의 "실질적 기여도"를 측정하는 표준 방법. 필터의 모든 weight를 종합하여 해당 채널이 출력에 미치는 영향을 단일 수치로 요약한다.

---

## 4. 아키텍처 분석 및 Prunable 그룹 정의

### 4.1 분류 기준

> **Prunable**: 내부 hidden dim — 양쪽 외부 채널(stage 경계)과 연결되지 않아 자유롭게 축소 가능  
> **Non-prunable**: 출력 projection, stage 경계 채널, classifier head 출력 등

### 4.2 Prunable 그룹 (최종 구현)

| 그룹명 | 위치 | 파라미터 비중 (B1) | 처리 방식 |
|--------|------|-------------------|---------|
| **G_MBCONV** | stage1~4 모든 MBConv의 mid_channels | 32.01% | inverted_conv 출력 → depth_conv → point_conv 입력 연동 |
| **G_FUSEDMB** | L-series stage1~2 FusedMBConv의 mid_channels | — (B-series=0%) | spatial_conv 출력 → point_conv 입력 연동 |
| **G_STEM** | input_stem chain 전체 (B: Conv+DSConv, L: Conv+ResBlock) | 0.01% | single index로 chain 전체 + 다음 stage 첫 conv 입력 컬럼 동기화 |
| **G_HEAD0** | ClsHead op_list[0] Conv (256→1536) 출력 필터 | ~4.3% | Conv 출력 → BN → Linear 입력 컬럼 |
| **G_HEAD1** | ClsHead op_list[2] Linear (1536→1600) 출력 행 | ~44.6% | Linear 출력 → LayerNorm → 최종 Linear 입력 컬럼 (sparsity 상한 0.40) |

### 4.3 Non-Prunable 그룹 (제외 이유)

| 그룹 | 제외 사유 |
|------|---------|
| **LiteMLA** (qkv/aggreg/proj) | qkv가 Q/K/V 인터리브 + multi-scale aggreg 결합도가 높음. 이번 구현에서 제외 |
| **Stage 경계 채널** (width_list[1..4]) | backbone 외부 hyperparameter. 변경 시 stage 간 channel mismatch |
| **ClsHead 출력** (n_classes=1000) | 태스크 고정값 |

### 4.4 Coupled 관계 (동일 인덱스 적용)

```
MBConv:
  inverted_conv.out_ch [idx]
    ↓ 동일 인덱스
  depth_conv.weight [idx]  (groups=mid이므로 채널==그룹)
    ↓ 동일 인덱스
  point_conv.in_ch [:, idx]

input_stem chain:
  stem.op_list[0].conv.weight [idx]    (첫 ConvLayer 출력)
    ↓ 동일 인덱스
  각 DSConv/ResBlock 양방향 마스킹
    ↓ 동일 인덱스
  stages[0].op_list[0].main.inverted_conv 입력 컬럼 [:, idx]

ClsHead G_HEAD0:
  op_list[0].conv.weight [idx]         (Conv 출력 필터)
    ↓ AdaptiveAvgPool2d 는 채널 순서 보존
  op_list[2].linear.weight [:, idx]    (Linear 입력 컬럼)

ClsHead G_HEAD1:
  op_list[2].linear.weight [idx]       (Linear 출력 행)
    ↓ 동일 인덱스
  op_list[2].norm (LayerNorm) [idx]
    ↓ 동일 인덱스
  op_list[3].linear.weight [:, idx]    (최종 Linear 입력)
```

### 4.5 Head Pruning을 추가한 배경

EfficientViT-B1 실측 기준으로 **Head가 전체 파라미터의 48.94%를 차지**함을 발견했다.  
Backbone만 pruning하면 prunable 영역이 32%에 불과하여, target=30% 달성을 위해 backbone sparsity를 **93~94%** 까지 올려야 했다. 이 수준에서는 MBConv hidden dim의 94%가 0이 되어 정확도가 급락한다.

Head를 포함하면 prunable 영역이 32% → 81%로 확대되어, 동일 target=30%에서 backbone sparsity를 **47~55%** 수준으로 낮출 수 있다.

```
B1 파라미터 분포:
  G_STEM:    1,027  ( 0.01%)  ← Prunable
  G_MBCONV: 2,917,976 (32.01%) ← Prunable
  G_HEAD:  4,461,161 (48.94%) ← Prunable (head 포함 시)
  G_LITEMLA: 1,735,303 (19.04%) ← Non-prunable
  TOTAL:   9,115,467 (100.00%)
```

---

## 5. 구현 전체 구조

### 5.1 파일 변화 요약

```
efficientvit2026/
├── efficientvit/
│   ├── apps/
│   │   └── trainer/
│   │       └── base.py                     ← MODIFIED (+5줄: pruner hook)
│   ├── clscore/
│   │   ├── trainer/
│   │   │   └── cls_trainer.py              ← MODIFIED (+3줄: pruner 인자)
│   │   └── pruning/                        ← NEW (패키지 전체 신규)
│   │       ├── __init__.py
│   │       ├── efficientvit_pruning.py     ← NEW (~830줄)
│   │       └── efficientvit_reducing.py    ← NEW (~503줄)
│   └── models/
│       └── nn/
│           └── ops.py                      ← MODIFIED (LiteMLA 수정, NPU 호환)
├── applications/
│   └── efficientvit_cls/
│       ├── train_efficientvit_cls_model.py ← MODIFIED (+19줄: pruning 옵션)
│       ├── eval_efficientvit_cls_model.py  ← MODIFIED (+wandb 인자)
│       ├── reduce_efficientvit_cls_model.py ← NEW
│       └── measure_memory.py               ← NEW
├── efficientvit/assets/compile_example/
│   └── export_torch_model.py               ← NEW (NPU export)
└── configs/imagenet/
    └── efficientvit_b2.yaml                ← MODIFIED (base_batch_size: 32)
```

**학습 코드 자체에 침투한 라인: 총 8줄** (나머지는 별도 모듈/CLI)

### 5.2 신규 파일별 역할

| 파일 | 역할 | 핵심 공개 API |
|------|------|-------------|
| `efficientvit_pruning.py` | Soft Pruning 본체 (이진탐색, 마스킹, sparsity 로깅) | `EfficientViTPruner` |
| `efficientvit_reducing.py` | Dense 변환 (0 채널 물리적 제거) + CLI | `reduce_efficientvit_cls_model` |
| `reduce_efficientvit_cls_model.py` | Reducing CLI 진입점 | `python ... reduce_...py` |
| `measure_memory.py` | 그룹별 파라미터 메모리 분해 + 원본/Reduced 비교 | `python ... measure_memory.py` |
| `export_torch_model.py` | NPU 컴파일용 nn.Module 객체 저장 | `python ... export_torch_model.py` |

---

## 6. 핵심 구현 상세

### 6.1 EfficientViTPruner 클래스

```python
class EfficientViTPruner:
    def __init__(
        self,
        model: nn.Module,
        target_compression: float,   # 목표 압축률 (e.g. 0.30 = 30% 파라미터 감소)
        max_sparsity: float = 0.95,  # 이진탐색 상한
        sparsity: float | None = None,
        head_sparsity_scale: float = 0.5,  # head sparsity = backbone × scale
        index_refresh_steps: int = 100,    # topk 재계산 간격 (성능 최적화)
    )

    def apply(self, model: nn.Module) -> None:
        """매 optimizer.step() 직후 호출. 모든 prunable 그룹을 0 마스킹."""

    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """실제 zero 비율 반환. WandB 로깅용."""
```

**주요 인자 설명:**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `target_compression` | (필수) | 전체 파라미터 기준 목표 감소율. 0.30 = 30% 감소 |
| `head_sparsity_scale` | 0.5 | head sparsity = backbone_sparsity × scale. 0.0이면 head pruning 비활성 |
| `index_refresh_steps` | 100 | 100 step마다 topk 인덱스 재계산. 0이면 매 step (느림) |

### 6.2 Sparsity 이진탐색

단순 선형 계산은 secondary effect를 무시하여 실제 압축률을 과소 추정한다.

**Secondary effect란:**
- inverted_conv mid를 n_prune개 제거 → depth_conv(groups=mid)도 같이 제거 → point_conv 입력 컬럼도 제거
- input_stem chain: point_conv가 in=out=C0이어서 `C0² - (C0-n)²` 이차 효과 발생

따라서 **이진탐색**으로 정확한 per-group sparsity를 결정한다:

```python
def _find_sparsity_by_bisection(model, target_compression, max_sparsity=0.95, iters=64):
    total_params  = count_params(model)
    target_remove = target_compression * total_params
    lo, hi = 0.0, max_sparsity
    for _ in range(64):    # 64회 → 약 1e-19 정밀도
        mid = (lo + hi) / 2
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
```

`_estimate_total_removed`는 각 MBConv의 secondary effect (DWConv 연동, point_conv 입력 컬럼)를 모두 포함하여 제거량을 정확히 추산한다.

### 6.3 성능 최적화 (2026-05-30 추가)

B2(24M 파라미터) 모델에서 매 step `pruner.apply()`의 CPU 병목이 발생했다. 세 가지 최적화를 적용했다.

| 병목 | 원인 | 해결 방법 |
|------|------|---------|
| `model.modules()` 매 step 순회 | Python 루프 오버헤드 | init 시 `_PruneGroup`으로 텐서 레퍼런스 1회 수집 |
| topk L2 norm 매 step 재계산 | CUDA 커널 N번 launch | `index_refresh_steps`마다 한 번만 재계산 (기본 100 step) |
| fancy-index scatter (`weight[idx]=0`) | CPU-GPU sync | 사전 할당 `(n,)` mask로 `weight.mul_(mask)` 벡터화 처리 |

```python
@dataclass
class _PruneGroup:
    criterion: torch.Tensor          # ranking 기준 weight
    sparsity:  float
    targets:   List[Tuple[...]]      # 마스킹 대상 (tensor, dim, fill_value)
    _mask:     torch.Tensor | None   # 캐시된 마스크

    def refresh(self): ...           # topk 재계산
    def apply(self): ...             # mask *= tensor 벡터화 마스킹
```

### 6.4 BN / LayerNorm 처리

**BatchNorm** (MBConv, input_stem, ClsHead G_HEAD0):
```python
bn.weight.data[idx] = 0.0
bn.bias.data[idx]   = 0.0
bn.running_mean[idx] = 0.0
bn.running_var[idx]  = 1.0    # 반드시 1.0 — 0이면 분모 0 문제
```

**LayerNorm** (ClsHead G_HEAD1):
```python
ln.weight.data[idx] = 0.0
ln.bias.data[idx]   = 0.0
# running_mean/var 없음 — 추가 처리 불필요
```

### 6.5 안전장치

| 장치 | 값 | 이유 |
|------|----|------|
| `MIN_SURVIVE` | 4 | 어떤 그룹도 최소 4채널 생존 보장 (정보 완전 단절 방지) |
| `max_sparsity` | 0.95 | 이진탐색 상한. 1.0이면 그룹 전체 소멸 위험 |
| `round()` 사용 | — | `int()`의 버림으로 소규모 그룹에서 발생하는 과소 pruning 방지 |
| G_HEAD1 sparsity 상한 | 0.40 | 1600→1000 FC는 1.6× near-complete. 800 미만이면 under-complete |
| Lazy init | — | `__init__` 시점(CPU)이 아닌 첫 `apply()` 시점(CUDA)에 텐서 수집 → device mismatch 방지 |

---

## 7. 훈련 루프 연동 및 WandB

### 7.1 Hook 삽입 위치

```python
# efficientvit/apps/trainer/base.py — after_step()
def after_step(self) -> None:
    self.scaler.unscale_(self.optimizer)
    if self.run_config.grad_clip is not None:
        torch.nn.utils.clip_grad_value_(self.model.parameters(), ...)
    self.scaler.step(self.optimizer)
    self.scaler.update()

    # ★ Soft Pruning Hook — optimizer.step() 직후, EMA 이전 ★
    pruner = getattr(self, "pruner", None)
    if pruner is not None:
        pruner.apply(self.network)    # self.network = DDP unwrap된 실제 모델

    self.lr_scheduler.step()
    self.run_config.step()
    if self.ema is not None:
        self.ema.step(self.network, self.run_config.global_step)
```

**이 위치가 중요한 이유:**
- `optimizer.step()` 이후: gradient가 weight에 반영된 직후 마스킹
- `ema.step()` 이전: EMA shadow가 pruning 후 weight를 추적하도록
- EMA shadow weight가 실제 학습 성능을 보존 (raw weight는 매 step 0으로 강제되므로)

### 7.2 ClsTrainer 변경 (최소 침투)

```python
class ClsTrainer(Trainer):
    def __init__(self, ..., pruner: Optional[Any] = None):
        super().__init__(...)
        self.pruner = pruner    # base.after_step이 getattr로 참조
```

`getattr(self, "pruner", None)` 패턴 덕분에 **다른 Trainer (Seg/SAM/AE 등)는 영향 없음**.

### 7.3 학습 진입점 변경

```bash
# Pruning 없는 Fine-tuning (기본값)
python train_efficientvit_cls_model.py config.yaml --target_compression 0.0

# Pruning Fine-tuning (target=30%, head 포함)
python train_efficientvit_cls_model.py config.yaml \
    --target_compression 0.30 \
    --pruning_head_sparsity_scale 0.5 \
    --prune_refresh_steps 100
```

추가된 argparse 인자:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--target_compression` | 0.0 | 0.0이면 pruning 비활성 |
| `--pruning_max_sparsity` | 0.95 | 이진탐색 상한 |
| `--pruning_head_sparsity_scale` | 0.5 | head sparsity 비율 |
| `--prune_refresh_steps` | 100 | topk 갱신 주기 |

### 7.4 WandB 연동

```bash
# Fine-tune + Pruning + WandB 전체 명령어 예시
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=12345 \
  applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --path /workspace/.../output/b1_prune30_head \
    --init_from assets/checkpoints/efficientvit_cls/efficientvit_b1_r224.pt \
    --amp bf16 \
    --target_compression 0.30 \
    --pruning_head_sparsity_scale 0.5 \
    --wandb \
    --wandb_project efficientvit-pruning \
    --wandb_run_name b1_prune30_head05 \
    --data_provider.data_dir /workspace/.../data/imagenet
```

**WandB 로깅 항목:**

| 키 | 내용 |
|----|------|
| `train/loss` | epoch 평균 학습 손실 |
| `train/top1` | epoch 평균 학습 Top-1 |
| `train/lr` | 현재 learning rate |
| `val/top1` | 검증 Top-1 |
| `val/top5` | 검증 Top-5 |
| `val/top1_best` | 현재까지 최고 val top-1 |
| `val/loss` | 검증 손실 |
| `pruning/actual_sparsity` | 실제 zero 필터 비율 |
| `pruning/zero_filters` | zero 채널 수 |
| `pruning/layer/<name>` | 블록별 zero 비율 |

gradient histogram도 500 step마다 기록된다 (`wandb.watch(model, log='gradients', log_freq=500)`).

---

## 8. Reducing — Dense 변환

### 8.1 원리

Soft Pruning 학습이 끝나면 죽은 채널의 weight L2 norm이 정확히 0이다.  
Reducing은 이 0 채널들을 물리적으로 제거하여 실제로 작은 Dense 모델을 만든다.

```
Soft Pruning 완료 모델:
  inverted_conv.weight[dead_ch] == 0.0  (L2 norm = 0)
  inverted_conv.weight[live_ch] != 0.0

Reducing (MBConv 예시):
  survived = where(norm(inverted_conv.weight) != 0)
  new_inverted_conv = Conv2d(in, len(survived), ...)
  new_inverted_conv.weight = old.weight[survived]      ← 살아남은 필터만

  new_depth_conv = Conv2d(n_new, n_new, k, groups=n_new, ...)  ← DW: in==out==groups
  new_depth_conv.weight = old.weight[survived]

  new_point_conv = Conv2d(n_new, out, 1, ...)
  new_point_conv.weight = old.weight[:, survived]      ← 입력 컬럼만 축소
```

**`!= 0` 기준을 사용하는 이유**: Soft Pruning이 정확히 0으로 마스킹하므로 임계값 불필요.

### 8.2 Reducing 대상 그룹

| 그룹 | 변환 내용 |
|------|---------|
| MBConv mid_channels | inverted_conv 출력 필터 → depth_conv (groups=n_new) → point_conv 입력 컬럼 |
| FusedMBConv mid_channels | spatial_conv 출력 필터 → point_conv 입력 컬럼 |
| input_stem chain | stem[0] 출력 → 각 DSConv/ResBlock 양방향 → 다음 stage 첫 conv 입력 컬럼 |
| ClsHead G_HEAD0 | Conv 출력 필터+BN → Linear 입력 컬럼 (G_HEAD0 먼저 실행 필수) |
| ClsHead G_HEAD1 | Linear 출력 행 → LayerNorm normalized_shape 축소 → 최종 Linear 입력 컬럼 |

### 8.3 실행 명령어

```bash
# Step 1: Reducing
python applications/efficientvit_cls/reduce_efficientvit_cls_model.py \
    --model efficientvit-b1 \
    --checkpoint /workspace/.../output/b1_prune30/checkpoint/model_best.pt \
    --output    /workspace/.../output/b1_prune30/reduced_b1_30pct.pt \
    --save-full-model    # eval 스크립트 호환을 위해 필수

# Step 2: Eval
CUDA_VISIBLE_DEVICES=7 python applications/efficientvit_cls/eval_efficientvit_cls_model.py \
    --model efficientvit-b1 \
    --weight_url /workspace/.../output/b1_prune30/reduced_b1_30pct.pt \
    --path /workspace/.../data/imagenet/val \
    --wandb --wandb_project efficientvit-pruning --wandb_run_name b1_prune30_reduced_eval
```

**`--save-full-model` 이 필수인 이유**: `eval_efficientvit_cls_model.py`는 표준 B1 아키텍처(`width_list` 고정)로 모델을 생성하고 state_dict를 로드한다. Reducing 후 채널 수가 달라졌으므로 표준 아키텍처에는 로드가 불가능하다. 모델 객체 전체를 저장해야 `torch.load()`로 바로 사용할 수 있다.

### 8.4 EMA weight 우선 사용

Reducing CLI에서 checkpoint 로드 시 EMA weight를 우선 사용한다.

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
# EMA weights 우선: raw network는 매 step 0으로 강제 → degraded
# EMA.state_dict() 형식: {decay(float): shadows.state_dict()}
if "ema" in ckpt and isinstance(ckpt["ema"], dict):
    for v in ckpt["ema"].values():
        if isinstance(v, dict):
            return v  # EMA weights 반환
```

---

## 9. 메모리 프로파일링

### 9.1 measure_memory.py

그룹별 파라미터 메모리를 측정하는 도구. BN running stats(buffers)까지 포함한 실제 checkpoint 크기 기준.

```bash
# 기본 분석
python applications/efficientvit_cls/measure_memory.py --model efficientvit-b1 --per-stage

# Reducing 전후 비교
python applications/efficientvit_cls/measure_memory.py \
    --model efficientvit-b1 \
    --reduced /path/to/reduced_b1_30pct.pt \
    --per-stage
```

### 9.2 EfficientViT-B1 실측 기준 수치

**컴포넌트별 분해:**

```
=== [model] component-wise parameter memory ===
group        #mod         numel          MB        %
----------------------------------------------------
G_STEM          1         1,027       0.004    0.01%
G_HEAD          1     4,461,161      17.845   48.94%
G_LITEMLA       7     1,735,303       6.941   19.04%
G_MBCONV       14     2,917,976      11.672   32.01%
G_FUSEDMB       0             0       0.000    0.00%
----------------------------------------------------
TOTAL                 9,115,467      36.462  100.00%
```

**per-stage 분해:**

```
name           numel          MB        %
--------------------------------------------------
input_stem     1,027       0.004    0.01%
stages[0]     14,790       0.059    0.16%
stages[1]     89,481       0.358    0.98%
stages[2]    758,663       3.035    8.32%
stages[3]  3,790,345      15.161   41.58%
head       4,461,161      17.845   48.94%
```

**핵심 발견**: Head(48.94%)가 backbone보다 크다. 공식 모델 설명에서 "~9M 파라미터"로 알려진 것과 일치하지만, 내부 분포는 head 비중이 예상보다 훨씬 높다.

---

## 10. NPU 배포 대응

### 10.1 배포 파이프라인

```
reduced_model.pt (nn.Module)
        ↓
export_torch_model.py  →  efficientvit_b1_reduced.pt (nn.Module 객체)
        ↓
NPU 프로젝트 Docker:
qbcompiler.mxq_compile(model=..., backend="torch", ...)
        ↓
NPU 실행 파일
```

### 10.2 ONNX 경로를 포기한 이유

처음에는 `backend="onnx"` 경로를 시도했지만 qbcompiler에서 zeropoint overflow가 반복 발생했다.  
원인은 `LiteMLA.relu_linear_att` 내부의 `F.pad + Slice` 패턴이었다.

```python
# 변경 전 (문제 있음)
v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
vk  = torch.matmul(v, trans_k)
out = torch.matmul(vk, q)
out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
# ONNX에서 Pad + Slice(end=-1) → 양자화기에서 range 계산 실패 → zeropoint overflow

# 변경 후 (수학적으로 동일, ONNX 없이 torch backend로 우회)
vk      = torch.matmul(v, trans_k)
out     = torch.matmul(vk, q)           # 분자
k_sum   = k.sum(dim=-1, keepdim=True)
out_den = torch.matmul(k_sum.transpose(-1, -2), q)  # 분모
out     = out / (out_den + self.eps)
```

### 10.3 torch.reshape -1 제거

`backend="torch"` 경로에서도 추가 문제가 발견됐다:  
`torch.reshape(..., -1, ...)` 의 `-1`이 qbcompiler에서 `CustomOpOptions`로 분류되어 컴파일 실패.

```python
# 변경 전
qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))

# 변경 후 (명시적 계산)
n_groups = C // (3 * self.dim)
qkv = torch.reshape(qkv, (B, n_groups, 3 * self.dim, H * W))
```

두 수정 모두 **수학적으로 완전히 동일**하며 정확도 영향 없음.

---

## 11. 측정 기준 수치

### 11.1 Baseline (측정 완료)

| 항목 | 값 |
|------|-----|
| EfficientViT-B1 공식 Top-1 | 79.4% |
| EfficientViT-B1 파라미터 수 | 9,115,467 |
| Prunable (G_STEM + G_MBCONV) | 2,919,003 (32.02%) |
| Non-prunable (G_HEAD + G_LITEMLA) | 6,196,464 (67.98%) |
| Head 포함 Prunable | ~7.38M (80.96%) |

### 11.2 target_compression별 예상 sparsity (이진탐색 추정)

**Backbone only (`head_sparsity_scale=0.0`):**

| target | backbone sparsity | 비고 |
|--------|------------------|------|
| 0.10 | ≈ 0.31 | |
| 0.15 | ≈ 0.47 | |
| 0.20 | ≈ 0.63 | |
| 0.25 | ≈ 0.79 | |
| **0.30** | **≈ 0.93~0.94** | ⚠️ 극단적 — 정확도 급락 위험 |

**Head 포함 (`head_sparsity_scale=0.5`, 권장):**

| target | backbone sparsity | head sparsity (×0.5) |
|--------|------------------|---------------------|
| 0.15 | ≈ 0.22~0.27 | ≈ 0.11~0.14 |
| 0.20 | ≈ 0.30~0.36 | ≈ 0.15~0.18 |
| 0.25 | ≈ 0.39~0.46 | ≈ 0.20~0.23 |
| **0.30** | **≈ 0.47~0.55** | **≈ 0.24~0.28** |

> backbone sparsity가 0.94 → 0.51로 완화되어 정확도 보존에 유리.

### 11.3 학습 환경

| 항목 | 값 |
|------|-----|
| 데이터셋 | ImageNet-1K (train ~1.28M, val 50K) |
| 학습 서버 | `/workspace/etri_iitp/JS/efficientvit2026/` |
| 사용 GPU | CUDA GPU 번호 3, 4, 5, 6, 7 (실험별 상이) |
| AMP | bf16 |
| 배치 사이즈 | B1: 128/GPU, B2: 32/GPU (VRAM 12GB 이하 대응) |
| 1 epoch steps (B1, GPU 1장) | ~5,004 steps |

---

## 12. 실험 설계 및 목표 결과

### 12.1 실험 구성

총 4가지 실험 조건을 설계했다:

| 실험 | 명칭 | 설명 | wandb run명 |
|------|------|------|------------|
| A | Fine-tune Baseline | Pruning 없이 pretrained에서 fine-tune | `b1_finetune_baseline` |
| B | Prune 15% | target=15%, head_scale=0.5 | `b1_prune15_head05` |
| C | Prune 30% | target=30%, head_scale=0.5 | `b1_prune30_head05` |
| D | B2 Prune 50%/70% | B2 모델 대용량 압축 | `b2_prune50_head05` |

### 12.2 목표 수치 (실험 결과 수집 예정)

실험이 완료되면 아래 표를 채운다:

| 실험 | Param (원본) | Param (Reduced) | 압축률 | val Top-1 | 기준 대비 차이 |
|------|------------|----------------|--------|-----------|-------------|
| A: Baseline | 9.12M | — | 0% | (측정 예정) | 기준 |
| B: Prune 15% | 9.12M | (측정 예정) | ~15% | (측정 예정) | (측정 예정) |
| C: Prune 30% | 9.12M | (측정 예정) | ~30% | (측정 예정) | (측정 예정) |
| D-50: B2 Prune 50% | 24.3M | (측정 예정) | ~50% | (측정 예정) | (측정 예정) |
| D-70: B2 Prune 70% | 24.3M | (측정 예정) | ~70% | (측정 예정) | (측정 예정) |

### 12.3 달성하려는 것

1. **15% 압축 + 정확도 손실 ≤1%**: 가장 현실적인 목표. `head_scale=0.5`로 backbone sparsity ≈ 27% 수준.
2. **30% 압축 + 정확도 손실 ≤2%**: Head 포함으로 backbone sparsity ≈ 51%. 2% 이내면 NPU 배포 가능 수준.
3. **B2 50~70% 압축**: 24.3M → 12M, 7.3M 수준. 더 큰 모델에서 공격적 압축 시 정확도 회복력 검증.

### 12.4 평가 방법

1. `eval_efficientvit_cls_model.py`로 Reduced 모델의 val Top-1 측정
2. WandB `efficientvit-pruning` 프로젝트에서 실험간 비교
3. `measure_memory.py --reduced`로 실제 압축률 확인
4. NPU 컴파일(`export_torch_model.py` → `mxq_compile`) 성공 여부 확인

---

## 13. 주요 의사결정 정리

| 결정 | 채택 방향 | 근거 |
|------|---------|------|
| Pruner hook 위치 | `base.py:after_step()` 4줄 추가 | EMA 갱신이 base 안에 있어 ClsTrainer에서 오버라이드만으로 "EMA 이전" 위치 확보가 까다로움 |
| Pruner 주입 방식 | `ClsTrainer.__init__`에 `pruner=None` 인자 | API 명시성 확보, 다른 Trainer 영향 없음 |
| LiteMLA 제외 | 이번 구현에서 제외 | qkv 인터리브 + multi-scale aggreg coupling으로 secondary effect 추정식이 복잡. 정확도 보존 우선 |
| input_stem idx 기준 | `op_list[0].conv` 출력 필터 L2 norm | inner block들이 C0 단일 채널 공간을 공유하므로 단일 layer ranking으로 충분 |
| BN running_var | 1.0으로 설정 | 0이면 `(x-mean)/sqrt(0+eps)` → 수치 불안정 |
| Reducing 임계값 | `norm != 0` (정확 0) | Soft Pruning이 정확히 0으로 마스킹. 부동소수점 임계값 불필요 |
| G_HEAD1 sparsity 상한 | 0.40 | 1600→1000 FC는 1.6× near-complete. 0.40 이상이면 under-complete 위험 |
| EMA weight 우선 | Reducing 시 EMA 사용 | raw weight는 매 step 0으로 강제 → degraded. EMA shadow가 실제 성능 보존 |
| ONNX 포기 | torch backend로 전환 | LiteMLA의 Pad+Slice 패턴이 qbcompiler 양자화기에서 zeropoint overflow |
| index_refresh_steps | 기본 100 | 수렴 후 pruned 채널 순위는 수백 step 동안 거의 변하지 않음. CPU topk 병목 완화 |
| `--save-full-model` | Reducing 출력에 필수 | eval 스크립트가 standard 아키텍처로 모델 생성 후 state_dict 로드 → shape 불일치 오류 |

---

## 14. 구현 타임라인 (Changelog)

### 2026-04-27 — 초기 구현

- `EfficientViTPruner` 기본 클래스 작성
- MBConv/FusedMBConv `mid_channels` pruning 구현
- 이진탐색으로 sparsity 결정 (64회, secondary effect 포함)
- `efficientvit_reducing.py` 기본 구현
- `base.py:after_step()` hook 삽입 (4줄)
- M-series 레퍼런스 구현과 대조, 방법론 일치 확인

### 2026-04-27 (rev 2) — input_stem chain + measure_memory

- `_prune_input_stem`: B/L-series 자동 처리. single index로 chain 전체 + 다음 stage 연동
- `_estimate_removed_input_stem`: 이차 효과 포함 정확한 추정식
- `_reduce_input_stem`: DSConv groups=n_new 처리, 양방향 축소
- `measure_memory.py` 신규: 그룹별 / per-stage 파라미터 분해, 원본/Reduced 비교

### 2026-04-29 (rev 3) — 실행 명령어 정비

- `CUDA_VISIBLE_DEVICES` + `--nproc_per_node` 불일치 문제 수정
- `--data_provider.data_dir=...` → 공백 분리 방식으로 수정 (`parse_unknown_args` 형식)

### 2026-04-29 (rev 4) — WandB 연동

- `ClsTrainer`에 `wandb_project`, `wandb_run_name` 인자 추가
- 에폭별 `train/loss`, `val/top1`, `pruning/*` 자동 로깅
- gradient histogram 500 step마다 기록

### 2026-05-02 (rev 5) — eval 스크립트 WandB 지원 + Base Model Validation

- `eval_efficientvit_cls_model.py`에 `--wandb` 인자 추가
- Base Model Validation 명령어 정의 (pretrained 단발 eval)
- eval 스크립트 경로 오류 수정 (M-series → B-series 경로)

### 2026-05-03 (rev 6) — ClsHead Pruning 구현

- Head가 48.94%를 차지함을 실측으로 확인
- `_prune_head0`, `_prune_head1` 구현 (G_HEAD0/1)
- `_reduce_head0`, `_reduce_head1` 구현 (순서 G_HEAD0 → G_HEAD1 필수)
- `head_sparsity_scale=0.5` 기본값: head sparsity = backbone × 0.5
- G_HEAD1 sparsity 상한 0.40 설정 (under-complete 방지)
- 이진탐색에 head 제거량 합산

### 2026-05-30 (rev 7) — B2 모델 지원

- `efficientvit_b2.yaml`에 `base_batch_size: 32` 추가 (VRAM 12GB 이하 대응)
- B2 전용 실험 명령어 정의 (target=50%, 70%)

### 2026-05-30 (rev 8) — CPU 병목 최적화

- `_PruneGroup` dataclass 도입: init 시 텐서 레퍼런스 1회 수집
- `index_refresh_steps` 인자: 기본 100 step마다 topk 재계산
- `weight *= mask` 벡터화 마스킹: scatter 대신 elementwise mul

### 2026-06 — NPU 대응 & 타 모델 이식 준비

- `LiteMLA.relu_linear_att` 수정: F.pad+Slice → 분자/분모 명시적 분리 (ONNX 양자화 우회)
- `torch.reshape -1` → 명시적 계산 (qbcompiler CustomOpOptions 오류 해결)
- `export_torch_model.py`: reduced 모델 nn.Module 객체 저장
- `TIMM_PRUNING_GUIDE.md`: ViT/DeiT 적용 가이드 작성

---

## 15. 향후 확장 가능 항목

### 15.1 LiteMLA Q/K Pruning (미구현)

- 현재: LiteMLA 완전 제외
- 가능성: Q, K의 head_dim 축소
- 난이도: qkv가 Q/K/V 인터리브 + multi-scale aggreg coupling 복잡
- 구현 시 필요: Q 기준으로 K에 동일 인덱스 강제, aggreg 채널 슬라이싱

### 15.2 GLUMBConv 지원 (미구현)

- `local_module="GLUMBConv"` 사용 시에만 필요
- inverted_conv 출력이 `mid*2`로 data/gate 분리 → 쌍으로 인덱스 처리 필요
- 기본 config에서는 사용하지 않으므로 현재 영향 없음

### 15.3 타 모델 적용

- **timm ViT/DeiT**: `TIMM_PRUNING_GUIDE.md` 참조. G_FFN이 메인 타겟 (전체의 ~67%)
- **방법론 공통**: `PRUNING_METHODOLOGY.md`에 범용 체크리스트 정리

### 15.4 정확도 회복 기법 (미탐색)

현재는 Soft Pruning + Standard Fine-tuning만 수행했다.  
정확도 손실이 큰 경우 아래를 추가로 탐색할 수 있다:
- Knowledge Distillation (pruned 모델이 원본을 teacher로 학습)
- Cosine Annealing with Warm Restarts (sparsity 증가 스케줄)
- Progressive Pruning (낮은 sparsity → 점진적 증가)

---

*보고서 작성일: 2026-06-22*  
*기준 코드: `efficientvit2026` master branch*  
*관련 문서: `PRUNING_METHODOLOGY.md`, `PRUNING_IMPLEMENTATION_REPORT.md`, `CLSHEAD_PRUNING_ANALYSIS.md`, `TIMM_PRUNING_GUIDE.md`*
