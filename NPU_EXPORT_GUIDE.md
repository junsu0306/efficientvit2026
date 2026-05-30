# Pruned EfficientViT → NPU 컴파일 가이드

## 배경 및 핵심 결론

### 왜 기존 .pt는 NPU 컴파일러에서 실패했는가

기존 파이프라인에서 실패 원인은 **timm 형식이 달라서가 아니라 NHWC 변환 도구** 때문이었다.

```
실패한 파이프라인:
이 프로젝트 .pt → NHWC 변환 도구 → NHWC ONNX → mxq_compile
                       ↑
              Slice 노드 상수가 FLOAT64로 재파싱됨
              → 양자화 시 zeropoint 오버플로우 발생

동작한 파이프라인 (timm):
timm.create_model() → torch.onnx.export 직접 → NCHW ONNX → mxq_compile
                                  ↑
                    NHWC 변환 없이 바이패스 → Slice 상수가 INT64으로 정상 저장
```

**결론: NHWC 변환 도구를 완전히 건너뛰고 `torch.onnx.export`를 직접 호출하면 pruned 모델도 동일하게 동작한다.**

---

## 전체 파이프라인

```
[서버 A: 학습/프루닝 서버]                [서버 B: NPU 장착 서버]
                                          │
Soft Pruning 학습                          │  (Docker 외부)
    ↓                                     │  export_pruned_to_onnx.py 실행
reduce_efficientvit_cls_model()            │      ↓
    ↓                                     │  pruned_model.onnx 생성
pruned_checkpoint.pt 저장                  │      ↓
    ↓                                     │  calib 데이터 준비
서버 B로 파일 전송 ──────────────────────→ │  (Docker 내부)
  - pruned_checkpoint.pt                  │  compile_pruned.py 실행
  - export_pruned_to_onnx.py              │      ↓
  - compile_pruned.py                     │  pruned_model.mxq 생성
```

---

## 서버 B에서 실행 순서

### 1단계: ONNX Export (Docker 외부)

의존성 설치:
```bash
pip install timm onnx torch
# 이 프로젝트도 설치되어 있어야 함
pip install -e /path/to/efficientvit2026
```

`export_pruned_to_onnx.py` 작성 및 실행:

```python
"""
Pruned EfficientViT → NCHW ONNX export

주의: NHWC 변환 도구 사용 금지.
      torch.onnx.export 직접 호출해야 Slice 상수가 INT64으로 저장됨.
"""

import os
import sys
import torch
import onnx
import numpy as np

sys.path.insert(0, "/path/to/efficientvit2026")
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.clscore.pruning.efficientvit_reducing import reduce_efficientvit_cls_model

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "efficientvit-b2"        # pruning 대상 모델명
INPUT_SIZE   = 288                      # 학습 때 사용한 해상도
PRUNED_CKPT  = "pruned_checkpoint.pt"  # soft pruning 후 reduce된 체크포인트
ONNX_OUT     = "efficientvit_b2_r288_pruned_nchw.onnx"
CALIB_DIR    = "calib_nchw"
CALIB_TXT    = "calib_nchw.txt"
N_CALIB      = 100
# ──────────────────────────────────────────────────────────────────────────────

# 1. 모델 로드
print(f"=> 모델 생성: {MODEL_NAME}")
model = create_efficientvit_cls_model(MODEL_NAME, pretrained=False)

ckpt = torch.load(PRUNED_CKPT, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict):
    # EMA 가중치 우선 사용
    if "ema" in ckpt and isinstance(ckpt["ema"], dict):
        for v in ckpt["ema"].values():
            if isinstance(v, dict):
                print("=> EMA 가중치 사용")
                ckpt = v
                break
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

# module. prefix 제거 (DDP 체크포인트 대응)
ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)

# 2. Structural reduce (이미 reduced된 .pt라면 이 줄 생략)
# reduce_efficientvit_cls_model(model)

n_params = sum(p.numel() for p in model.parameters())
print(f"=> 파라미터 수: {n_params:,}")

# 3. ONNX export (NCHW, NHWC 변환 없음)
model.eval()
dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

print(f"=> ONNX export: {ONNX_OUT}")
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

# 4. ONNX 검증 및 FLOAT64 상수 확인
onnx_model = onnx.load(ONNX_OUT)
onnx.checker.check_model(onnx_model)

float64_consts = [
    i.name for i in onnx_model.graph.initializer
    if i.data_type == 11  # DOUBLE = 11
]
if float64_consts:
    print(f"[WARNING] FLOAT64 상수 {len(float64_consts)}개 — NHWC 변환이 끼어든 것 같음")
else:
    print("[OK] FLOAT64 상수 없음 — 양자화 안전")

print(f"=> ONNX 저장 완료: {ONNX_OUT}")

# 5. 캘리브레이션 데이터 생성 (CHW npy 형식)
print(f"=> 캘리브레이션 데이터 생성")
os.makedirs(CALIB_DIR, exist_ok=True)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

calib_paths = []
for i in range(N_CALIB):
    arr = (np.random.randn(3, INPUT_SIZE, INPUT_SIZE).astype(np.float32) * std) + mean
    path = os.path.join(CALIB_DIR, f"{i:04d}.npy")
    np.save(path, arr)
    calib_paths.append(os.path.abspath(path))

with open(CALIB_TXT, "w") as f:
    f.write("\n".join(calib_paths) + "\n")

print(f"=> 캘리브레이션 {N_CALIB}개 저장: {CALIB_DIR}/")
print(f"=> txt 저장: {CALIB_TXT}")
print()
print("다음 단계:")
print(f"  1. {ONNX_OUT} 와 {CALIB_DIR}/ 를 Docker 컨테이너에서 접근 가능한 경로에 배치")
print(f"  2. Docker 안에서 compile_pruned.py 실행")
```

실행:
```bash
python3 export_pruned_to_onnx.py
```

---

### 2단계: NPU 컴파일 (Docker 내부)

`compile_pruned.py`:

```python
"""
Pruned EfficientViT NCHW ONNX → MXQ compile
Mobilint qbcompiler Docker 컨테이너 내부에서 실행.

Docker 실행 (호스트 경로를 컨테이너에 동일 경로로 마운트):
  docker run -it --ipc=host --name qbcompiler \
    -v /home/airlab_compression/npu:/home/airlab_compression/npu \
    mobilint/qbcompiler:v0.9.0.2 /bin/bash
"""

import os
from qbcompiler import mxq_compile

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CALIB_TXT = os.path.join(BASE_DIR, "calib_nchw.txt")

MODELS = [
    (
        os.path.join(BASE_DIR, "efficientvit_b2_r288_pruned_nchw.onnx"),
        os.path.join(BASE_DIR, "efficientvit_b2_r288_pruned_nchw.mxq"),
    ),
]

for onnx_path, mxq_path in MODELS:
    print(f"\n{'='*60}")
    print(f"Compiling: {os.path.basename(onnx_path)}")
    print(f"  calib : {CALIB_TXT}")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")

    mxq_compile(
        model=onnx_path,
        calib_data_path=CALIB_TXT,
        save_path=mxq_path,
        backend="onnx",
        in_dformats={"input": "NCHW"},  # NCHW ONNX이므로 변환 없음
        cpu_offload=True,               # EfficientViT unsupported ops 대응
        quantize_method="Percentile",
        quantize_percentile=0.99995,
        is_quant_ch=True,
    )

    print(f"Saved: {mxq_path}")

print("\nAll done.")
```

실행:
```bash
# Docker 내부에서
python3 /path/to/compile_pruned.py
```

---

## 체크리스트

### Export 전 확인
- [ ] `pruned_checkpoint.pt`가 이미 `reduce_efficientvit_cls_model()` 적용된 상태인지 확인
  - reduced 상태면 스크립트 내 `reduce_efficientvit_cls_model(model)` 줄 주석 처리
  - soft pruning 체크포인트라면 해당 줄 활성화
- [ ] `MODEL_NAME`, `INPUT_SIZE`가 학습 때와 일치하는지 확인
- [ ] NHWC 변환 도구(`nhwc_converter` 등)를 **절대 사용하지 않음**

### Export 후 확인
- [ ] `[OK] FLOAT64 상수 없음` 메시지 확인 (핵심)
- [ ] ONNX checker 통과 확인

### 컴파일 설정 확인
- [ ] `backend="onnx"` (pytorch 아님)
- [ ] `in_dformats={"input": "NCHW"}` (NHWC 변환 없음)
- [ ] `cpu_offload=True` (EfficientViT의 일부 ops가 NPU 미지원)

---

## 기술 메모

### LiteMLA의 ONNX 호환성

`LiteMLA.forward()`에 입력 크기 기반 분기가 있다:

```python
if H * W > self.dim:   # self.dim = 8 (기본값)
    out = self.relu_linear_att(qkv)
else:
    out = self.relu_quadratic_att(qkv)
```

`torch.onnx.export`는 trace 방식이므로 export 시 dummy input의 크기로 한 경로만 고정된다.
실제 추론 시 input_size = 224~288이면 feature map이 항상 H*W >> 8이므로 선형 경로로 고정되며 문제없다.

### Pruned 모델과 timm 모델의 차이

| 항목 | 이 프로젝트 pruned 모델 | timm 모델 |
|------|----------------------|-----------|
| 아키텍처 구현 | 원저자 코드 | timm 재구현 |
| state_dict key 이름 | `backbone.stages.0.0...` | `stages.0.blocks.0...` |
| 가중치 형식 | `.pt` (state_dict) | `.safetensors` |
| 채널 수 | pruning 후 불규칙 | 원본 고정값 |
| ONNX export | 직접 가능 | 직접 가능 |
| mxq_compile 호환 | NHWC 변환 없이 동일 | ✓ 확인됨 |

timm으로 변환할 필요 없음. 동일한 `torch.onnx.export` 경로로 처리 가능.
