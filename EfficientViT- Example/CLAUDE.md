# EfficientViT Soft Pruning — 새 레포지토리 구현 가이드

> **프로젝트**: RS-2024-00339187 | 고려대학교 | 3차년도 ViT 확장 연구
> **목표**: EfficientViT M4에 Soft Pruning 적용 → **압축률 76% 이상** 달성
> **압축률 공식**: `100 × (B - A) / B` (B: 원본 파라미터 수, A: 압축 후 파라미터 수)

---

## 작업 환경 (중요)

> **Claude(AI)는 로컬 컴퓨터에서 코드를 작성한다. 실제 학습 및 실행은 별도 GPU 서버에서 진행된다.**

| 환경 | 역할 | 경로 |
|------|------|------|
| 로컬 (macOS) | Claude가 코드 작성 / 파일 편집 | `/Users/junsu/Projects/EfficientVIT_Compression` |
| GPU 서버 | 실제 학습 / 평가 실행 | `/workspace/etri_iitp/JS/EfficientViT` |

작업 흐름:
```
로컬에서 코드 작성 (Claude)
    ↓
서버로 파일 복사 (scp / rsync / git 등)
    ↓
서버에서 학습 실행
```

### 서버 환경

```
서버 절대 경로 (기준):  /workspace/etri_iitp/JS/EfficientViT
conda 환경 이름:        efficientvit
GPU:                    cuda:0 (기본)

데이터셋 경로:
  /workspace/etri_iitp/JS/EfficientViT/data/imagenet/
  ├── train/    ← ImageNet train set
  └── val/      ← ImageNet val set
```

명령어 작성 시 항상 서버 절대 경로 기준으로 작성:
```bash
# 데이터 경로
--data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet

# 출력 경로 예시
--output_dir /workspace/etri_iitp/JS/EfficientViT/output/soft_pruning_76pct

# pretrained 체크포인트 예시
--resume /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth
```

---

## 0. 레포 구조 (새 레포 시작 기준)

새 레포는 다음 두 폴더만 존재:
```
/
├── classification/          ← EfficientViT 원본 코드 (그대로 가져옴)
│   ├── model/
│   │   ├── efficientvit.py
│   │   └── build.py
│   ├── data/
│   ├── engine.py            ← 원본 최대한 유지, backward 이후에 pruning 추가
│   ├── main.py              ← 원본 최대한 유지, pruning 인자만 추가
│   ├── losses.py
│   └── utils.py
│
└── YOLO pruning/            ← Reference 코드 (수정 없이 참고용)
    ├── pruning common.py    ← get_filter_pruning_idx, filter_pruning 패턴 참고
    ├── reducing_common.py   ← conv_reduce, fc_reduce, bn_reduce 패턴 참고
    ├── compression.py       ← yolov8_pruning/reducing 전체 흐름 참고
    └── trainer.py           ← optimizer.step() 이후 pruning 삽입 위치 참고
```

구현할 새 파일:
```
classification/pruning/
├── efficientvit_pruning.py  ← 핵심: soft pruning 적용 함수
└── efficientvit_reducing.py ← 마지막: sparse → dense 모델 변환
```

---

## 1. Pruning 철학 (YOLO 방식 동일)

### 핵심 아이디어

```
매 optimizer.step() 이후:
  1. 압축률로부터 "살려야 할 파라미터 수" 계산
  2. L2 norm 기준으로 하위 X%를 0으로 설정
  3. 다음 학습 iteration에서 0이었던 값이 조금 살아날 수 있음
  4. 하지만 계속 0으로 리셋 → 결국 남은 파라미터들이 그 크기에 맞게 최적화됨
  5. 학습 완료 후, 0인 파라미터를 물리적으로 제거 (reducing)
```

YOLO pruning/trainer.py에서의 패턴:
```python
# Backward
self.scaler.scale(self.loss).backward()

# Optimize
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()  # optimizer.step() 내부에서 수행
    ########################################################################
    ################## SM Insert ############################################
    if self.args.pruning_ratio == 0:
        continue
    else:
        yolov8_pruning(self.model.model, self.args.pruning_ratio)  # ← 여기!
    ########################################################################
```

EfficientViT engine.py에 동일하게 적용:
```python
# loss_scaler 내부에서 backward + optimizer.step() 수행
loss_scaler(loss, optimizer, ...)

# ← 여기에 pruning 삽입 (backward 이후, 다음 forward 이전)
if pruner is not None:
    pruner.apply_pruning(model)
```

---

## 2. Pruning 대상 (EfficientViT 구조)

### 2.1 아키텍처 요약

```
Input [B, 3, H, W]
  └─ OverlapPatchEmbed   (3× Conv-BN-ReLU)
  └─ Stage 1 ~ 3
       EfficientViTBlock × L
            DWConv → FFN → CGA → DWConv → FFN
  └─ AvgPool + Linear → logits
```

### 2.2 Pruning 대상 레이어

| 대상 | 레이어 구조 | Pruning 방식 |
|------|------------|-------------|
| **G_FFN** | expand: Linear(C → C*r) + shrink: Linear(C*r → C) | expand의 출력 뉴런 (row) 기준 L2 norm, shrink의 입력 채널 (col)도 동일 인덱스 |
| **G_QK** | Q: Linear(C/h → d_qk) + K: Linear(C/h → d_qk) | Q의 출력 dim (row) 기준 L2 norm, K도 반드시 동일 인덱스 적용 |
| **G_V** | V: Linear(C/h → C/h) | (선택적, 보수적으로) |

### 2.3 절대 Pruning 금지

- **Output projection** (CGA의 마지막 Linear): 채널 얼라인먼트 필수
- **DWConv**: 제거 시 -1.4% 정확도 하락
- **SubsampleBlock**: stage 경계 차원 정의

---

## 3. 구현: efficientvit_pruning.py

### 3.1 전체 흐름

```python
# YOLO pruning common.py 패턴을 EfficientViT Linear에 맞게 변환
# Conv filter pruning → Linear row/column pruning

def get_linear_pruning_idx(layer, sparsity):
    """
    Linear 레이어의 row(출력 dim) 기준 L2 norm 계산 후
    하위 sparsity 비율의 인덱스 반환.
    YOLO의 get_filter_pruning_idx와 동일한 패턴.
    """
    with torch.no_grad():
        weight = layer.weight        # [out_features, in_features]
        num_rows = weight.shape[0]
        num_pruning = int(num_rows * sparsity)
        row_norms = torch.norm(weight, dim=1)   # [out_features]
        _, pruning_idx = torch.topk(row_norms, num_pruning, largest=False)
    return pruning_idx

def linear_row_pruning(layer, pruning_idx):
    """
    Linear 레이어의 지정된 row를 0으로 마스킹.
    YOLO의 filter_pruning과 동일한 패턴.
    """
    with torch.no_grad():
        layer.weight[pruning_idx, :] = 0.0
        if layer.bias is not None:
            layer.bias[pruning_idx] = 0.0

def linear_col_pruning(layer, pruning_idx):
    """
    Linear 레이어의 지정된 col을 0으로 마스킹 (shrink의 입력 dim).
    """
    with torch.no_grad():
        layer.weight[:, pruning_idx] = 0.0
```

### 3.2 Sparsity 계산 (핵심)

```python
def compute_sparsity_from_compression(model, target_compression):
    """
    목표 압축률로부터 각 그룹의 sparsity를 계산.

    Args:
        model: EfficientViT 모델
        target_compression: 0.76이면 76% 파라미터 제거, 24% 유지

    Returns:
        sparsity_ffn: FFN 그룹에 적용할 sparsity
        sparsity_qk: QK 그룹에 적용할 sparsity
    """
    # 전체 prunable 파라미터 수 계산
    total_ffn_params = count_ffn_params(model)
    total_qk_params = count_qk_params(model)
    total_prunable = total_ffn_params + total_qk_params

    # 제거할 파라미터 수
    target_remove = int(total_prunable * target_compression)

    # FFN : QK 비율은 중요도에 따라 조정 (FFN이 더 많이 제거됨)
    # 예: FFN 80% 제거, QK 70% 제거 (QK는 이미 작으므로)
    sparsity_ffn = target_compression * 1.05   # FFN 약간 더 aggressively
    sparsity_qk = target_compression * 0.90    # QK 조금 보수적
    # 값 클리핑
    sparsity_ffn = min(sparsity_ffn, 0.95)
    sparsity_qk = min(sparsity_qk, 0.90)

    return sparsity_ffn, sparsity_qk
```

> **단순화 대안**: sparsity_ffn = sparsity_qk = target_compression으로 시작해서 결과 보고 조정

### 3.3 모델 전체에 Pruning 적용

```python
def efficientvit_pruning(model, sparsity_ffn, sparsity_qk):
    """
    EfficientViT 전체 모델에 soft pruning 적용.
    매 optimizer.step() 이후 호출.

    M4 blocks 구조:
      model.blocks1: [EVBlock]
      model.blocks2: [SubPreDWFFN, PatchMerging, SubPostDWFFN, EVBlock, EVBlock]
      model.blocks3: [SubPreDWFFN, PatchMerging, SubPostDWFFN, EVBlock, EVBlock, EVBlock]

    EVBlock 내부:
      block.mixer  → CGA (attention)
      block.ffn    → FFN (expand + shrink)
    """
    # 모든 블록 순회
    for block_list in [model.blocks1, model.blocks2, model.blocks3]:
        for block in block_list:
            block_type = type(block).__name__

            if block_type == 'EfficientViTBlock':
                # FFN pruning
                _prune_ffn(block.ffn, sparsity_ffn)
                # CGA (QK) pruning
                _prune_cga_qk(block.mixer, sparsity_qk)

            elif block_type == 'ResidualDrop':
                # SubPreDWFFN / SubPostDWFFN 내의 FFN
                inner = block.m
                if hasattr(inner, 'ffn'):
                    _prune_ffn(inner.ffn, sparsity_ffn)


def _prune_ffn(ffn_module, sparsity):
    """FFN의 expand + shrink를 연동하여 pruning."""
    # ffn_module: Sequential 또는 FFN class
    # expand: 첫 번째 Linear, shrink: 마지막 Linear
    expand = ffn_module.expand   # Linear(C → C*r)
    shrink = ffn_module.shrink   # Linear(C*r → C)

    # Q: expand의 out_features 기준으로 sparsity 계산
    pruning_idx = get_linear_pruning_idx(expand, sparsity)

    # expand의 row를 0으로
    linear_row_pruning(expand, pruning_idx)
    # shrink의 col을 동일 인덱스로 0으로 (연결 일관성)
    linear_col_pruning(shrink, pruning_idx)


def _prune_cga_qk(cga_module, sparsity):
    """CGA의 Q와 K를 동일 인덱스로 pruning (QK^T 차원 일치 필수)."""
    for head in cga_module.heads:  # 각 attention head
        q_proj = head.q   # Linear(C/h → d_qk)
        k_proj = head.k   # Linear(C/h → d_qk)

        # 반드시 Q 기준으로 인덱스 계산 → K에 동일 적용
        pruning_idx = get_linear_pruning_idx(q_proj, sparsity)
        linear_row_pruning(q_proj, pruning_idx)
        linear_row_pruning(k_proj, pruning_idx)   # CRITICAL: 동일 인덱스!
```

> **주의**: 실제 EfficientViT 코드에서 FFN/CGA 내부 레이어 이름은 `efficientvit.py`를 직접 읽고 확인 후 맞게 수정할 것

---

## 4. engine.py 수정 (최소한으로)

원본 `engine.py`의 `train_one_epoch` 함수에서 **딱 한 곳만** 수정:

```python
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, clip_grad=0, clip_mode='norm',
                    model_ema=None, mixup_fn=None,
                    set_training_mode=True, set_bn_eval=False,
                    pruner=None):    # ← pruner 인자 하나만 추가
    ...
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        ...
        # Forward + Loss (원본 그대로)
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        optimizer.zero_grad()

        # Backward + optimizer.step() (원본 그대로)
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), ...)

        # ============================================================
        # [추가] Soft Pruning: optimizer.step() 직후 weight masking
        # YOLO trainer.py의 yolov8_pruning() 삽입 위치와 동일
        if pruner is not None:
            pruner.apply(model)
        # ============================================================

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    ...
```

---

## 5. main.py 수정 (최소한으로)

인자 추가 (pruning 관련 3개만):
```python
# 기존 args에 추가
parser.add_argument('--pruning', action='store_true',
                    help='Enable soft pruning')
parser.add_argument('--target-compression', type=float, default=0.76,
                    help='Target compression rate (e.g., 0.76 = remove 76%)')
parser.add_argument('--pruning-ffn-ratio', type=float, default=1.05,
                    help='FFN pruning ratio multiplier vs target (default 1.05)')
parser.add_argument('--pruning-qk-ratio', type=float, default=0.90,
                    help='QK pruning ratio multiplier vs target (default 0.90)')
```

학습 루프에서:
```python
# pruner 초기화 (main 함수 내, 모델 생성 후)
pruner = None
if args.pruning:
    from classification.pruning.efficientvit_pruning import EfficientViTPruner
    pruner = EfficientViTPruner(
        target_compression=args.target_compression,
        ffn_ratio=args.pruning_ffn_ratio,
        qk_ratio=args.pruning_qk_ratio,
    )

# train_one_epoch 호출 시
train_stats = train_one_epoch(
    model, criterion, data_loader_train,
    optimizer, device, epoch, loss_scaler,
    args.clip_grad, args.clip_mode, model_ema, mixup_fn,
    set_training_mode=True,
    set_bn_eval=args.set_bn_eval,
    pruner=pruner,   # ← 추가
)
```

---

## 6. Pruner 클래스 설계

```python
class EfficientViTPruner:
    """
    EfficientViT용 Soft Pruner.
    매 optimizer.step() 이후 apply() 호출.

    동작 방식:
      - 첫 호출 시 sparsity 계산 (target_compression 기반)
      - 이후 매 호출 시 L2 norm 기준으로 하위 sparsity% weight를 0으로 설정
      - 0이었던 값이 gradient로 살아나도 다음 step에서 다시 0으로
    """
    def __init__(self, target_compression=0.76, ffn_ratio=1.05, qk_ratio=0.90):
        self.sparsity_ffn = min(target_compression * ffn_ratio, 0.95)
        self.sparsity_qk = min(target_compression * qk_ratio, 0.90)

    def apply(self, model):
        efficientvit_pruning(model, self.sparsity_ffn, self.sparsity_qk)
```

---

## 7. efficientvit_reducing.py (학습 완료 후 실행)

학습 완료 후 0인 weight를 물리적으로 제거하여 작은 dense 모델 생성.
`YOLO pruning/reducing_common.py`의 `fc_reduce`, `bn_reduce` 함수를 그대로 활용.

```python
from YOLO_pruning.reducing_common import fc_reduce, get_survived_filter_idx

def get_survived_linear_idx(layer):
    """
    row(출력 dim) 기준으로 norm이 0이 아닌 인덱스 반환.
    reducing_common.py의 get_survived_filter_idx와 동일 패턴.
    """
    weight = layer.weight   # [out_features, in_features]
    row_norms = torch.norm(weight, dim=1)
    survived_idx = torch.where(row_norms != 0)[0]
    return survived_idx


def reduce_ffn(ffn_module, reduced_ffn_module):
    """FFN expand + shrink를 물리적으로 축소."""
    survived_idx = get_survived_linear_idx(ffn_module.expand)

    # fc_reduce 패턴 적용
    # expand: out_features 축소 (survived rows)
    # shrink: in_features 축소 (survived cols)
    ...


def reduce_cga_qk(cga_module, reduced_cga_module):
    """CGA Q + K를 동일 survived_idx로 축소."""
    survived_idx = get_survived_linear_idx(cga_module.heads[0].q)
    # Q와 K 모두 동일 인덱스 적용
    ...


def efficientvit_reducing(model, reduced_model):
    """
    Pruning된 모델 → Dense 축소 모델 변환.
    학습 완료 후 한 번만 실행.

    Args:
        model: soft pruning 학습 완료 모델 (0인 weight 존재)
        reduced_model: 새로 생성한 작은 EfficientViT 모델 인스턴스
    """
    # blocks1, blocks2, blocks3 순회하며 각 block 축소
    ...

    # 검증
    _ = reduced_model(torch.zeros(1, 3, 224, 224))
    opt_rate = compute_optimization_rate(model, reduced_model)
    print(f"Optimization rate: {opt_rate:.1f}%")
    assert opt_rate >= 76.0, f"Target not met: {opt_rate:.1f}%"
```

---

## 8. 학습 명령어

### M4 원본 훈련 하이퍼파라미터 (기준)

```bash
# 원본 M4 학습 (8 GPU, 서버에서 실행)
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  classification/main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --dist-eval \
  --batch-size 256 \
  --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/m4_baseline
```

### Soft Pruning 학습 (동일 하이퍼파라미터 + pruning 인자 추가)

```bash
# Single GPU 예시 (서버에서 실행)
python -m classification.main \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --resume /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 \
  --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning \
  --target-compression 0.76 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/soft_pruning_76pct \
  --device cuda:0

# Multi GPU (권장, 서버에서 실행)
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env \
  classification/main.py \
  --model EfficientViT_M4 \
  --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
  --resume /workspace/etri_iitp/JS/EfficientViT/efficientvit_m4.pth \
  --batch-size 256 \
  --epochs 300 \
  --opt adamw --lr 1e-3 --weight-decay 0.025 \
  --clip-grad 0.02 --clip-mode agc \
  --model-ema --model-ema-decay 0.99996 \
  --pruning \
  --target-compression 0.76 \
  --output_dir /workspace/etri_iitp/JS/EfficientViT/output/soft_pruning_76pct \
  --dist-eval
```

### 학습 완료 후 Reducing

```bash
python -m classification.pruning.efficientvit_reducing \
  --model EfficientViT_M4 \
  --checkpoint /workspace/etri_iitp/JS/EfficientViT/output/soft_pruning_76pct/checkpoint_best.pth \
  --output /workspace/etri_iitp/JS/EfficientViT/output/reduced_m4_76pct.pth
```

---

## 9. 구현 순서 (체크리스트)

### Step 1: 모델 구조 파악
- [ ] `classification/model/efficientvit.py` 열어서 EfficientViTBlock 내부 FFN 레이어 이름 확인
- [ ] CGA 내 Q, K, V projection 레이어 이름 확인
- [ ] SubPreDWFFN / SubPostDWFFN 구조 확인

### Step 2: efficientvit_pruning.py 구현
- [ ] `get_linear_pruning_idx()` - YOLO `get_filter_pruning_idx` 패턴
- [ ] `linear_row_pruning()` - YOLO `filter_pruning` 패턴
- [ ] `linear_col_pruning()` - shrink 연동용
- [ ] `_prune_ffn()` - expand + shrink 연동 pruning
- [ ] `_prune_cga_qk()` - Q + K 동일 인덱스 pruning
- [ ] `efficientvit_pruning()` - 전체 모델 순회
- [ ] `EfficientViTPruner` 클래스

### Step 3: engine.py 수정
- [ ] `pruner=None` 인자 추가
- [ ] `loss_scaler(...)` 이후 `if pruner: pruner.apply(model)` 추가

### Step 4: main.py 수정
- [ ] pruning 인자 3개 추가 (`--pruning`, `--target-compression`, etc.)
- [ ] `EfficientViTPruner` 초기화 코드 추가
- [ ] `train_one_epoch` 호출에 `pruner=pruner` 전달

### Step 5: 학습 실행 및 검증
- [ ] 초기 sparsity 확인 (epoch 1 이후 0인 weight 비율 출력)
- [ ] 10 epoch 후 정확도 확인
- [ ] 300 epoch 완료

### Step 6: efficientvit_reducing.py 구현 및 실행
- [ ] `get_survived_linear_idx()` 구현
- [ ] `reduce_ffn()`, `reduce_cga_qk()` 구현
- [ ] `efficientvit_reducing()` 전체 함수 구현
- [ ] forward pass 검증
- [ ] 압축률 확인 (`opt_rate >= 76.0%`)

---

## 10. 핵심 제약사항 (절대 위반 금지)

```
1. QK pruning 시 Q와 K는 반드시 동일한 인덱스로 제거
   이유: QK^T 계산 시 차원이 일치해야 함

2. FFN pruning 시 expand.out_features == shrink.in_features 유지
   이유: expand → shrink 연결 차원 일치

3. Output projection은 pruning 금지
   이유: 전체 블록 채널 얼라인먼트 핵심

4. DWConv는 pruning 금지
   이유: 제거 시 -1.4% 정확도 하락 (실험 확인)

5. Reducing 시 Q 기준 survived_idx를 K에도 동일 적용
   이유: Pruning 시와 동일한 인덱스 일관성 유지
```

---

## 11. 참고: 모델 크기 계산

```python
def get_model_size_mb(model):
    """모델 파라미터 크기를 MB로 반환."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / 1e6

# M4 원본: ~35.2 MB
# 76% 압축 후: ~35.2 * 0.24 = ~8.4 MB

def compute_optimization_rate(original_model, reduced_model):
    B = get_model_size_mb(original_model)
    A = get_model_size_mb(reduced_model)
    return 100 * (B - A) / B
```

---

## 12. M4 모델 구조 요약 (블록 인덱스)

```
model.patch_embed   ← OverlapPatchEmbed (Conv 3개)
model.blocks1       ← [EVBlock(C=128, H=4)]
model.blocks2       ← [SubPreDWFFN, PatchMerging(128→256), SubPostDWFFN,
                        EVBlock(C=256, H=4), EVBlock(C=256, H=4)]
model.blocks3       ← [SubPreDWFFN, PatchMerging(256→384), SubPostDWFFN,
                        EVBlock(C=384, H=4), EVBlock(C=384, H=4), EVBlock(C=384, H=4)]
model.head          ← AvgPool + Linear(384 → num_classes)
```

M4 Q/K/V 크기:
| Stage | C | H | C/H | d_qk |
|-------|---|---|-----|------|
| 1 | 128 | 4 | 32 | 16 |
| 2 | 256 | 4 | 64 | 16 |
| 3 | 384 | 4 | 96 | 16 |

FFN expansion ratio r=2이므로:
- Stage 1 FFN: 128 → 256 → 128
- Stage 2 FFN: 256 → 512 → 256
- Stage 3 FFN: 384 → 768 → 384
