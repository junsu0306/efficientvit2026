#!/usr/bin/env python3
"""
EfficientViT-B0/B1 ONNX → MXQ compile script
Runs inside the Mobilint qbcompiler Docker container.

Key settings:
  - backend="onnx"
  - in_dformats="NCHW" : ONNX 모델 입력이 NCHW이므로 캘리브레이션도 CHW 포맷 사용.
      컴파일러가 CPU/NPU 경계에서 NCHW↔NHWC 변환을 자동 삽입함.
  - cpu_offload=True    : required for EfficientViT unsupported ops (attention/GELU)
  - calib_nchw/         : CHW-ordered calibration tensors (shape: [3, 224, 224])

Directory layout (relative to npu/):
  compile/compile_nchw.py   ← this file
  compile/calib_nchw.txt    ← calibration file list (paths inside Docker)
  assets/calib_nchw/        ← CHW source calibration tensors
  assets/onnx/*.onnx        ← input models
  assets/mxq/*.mxq          ← output compiled models

Docker run (mount host npu/ so that calib_nchw.txt absolute paths resolve):
  docker run -it --ipc=host --name qbcompiler \
    -v /home/airlab_compression/npu:/home/airlab_compression/npu \
    mobilint/qbcompiler:v0.9.0.2 /bin/bash

Then inside container:
  python3 /home/airlab_compression/npu/compile/compile_nchw.py
"""

import glob
import os
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # .../npu/compile
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

CALIB_CHW_DIR = os.path.join(ASSETS_DIR, "calib_nchw")

chw_files = sorted(glob.glob(os.path.join(CALIB_CHW_DIR, "*.npy")))
if not chw_files:
    raise FileNotFoundError(f"No CHW calibration .npy files found in {CALIB_CHW_DIR}")

CALIB_TXT = os.path.join(BASE_DIR, "calib_nchw.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(chw_files) + "\n")
print(f"Calibration list: {len(chw_files)} CHW files -> {CALIB_TXT}")

MXQ_DIR = os.path.join(ASSETS_DIR, "mxq")
os.makedirs(MXQ_DIR, exist_ok=True)

MODELS = [
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_timm_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b0_r224_timm_nchw.mxq"),
    ),
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b1_r224_timm_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b1_r224_timm_nchw.mxq"),
    ),
    # 원본(mit-han-lab) repo 가중치 → export_original_to_onnx.py 로 생성한 ONNX
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_original_r224_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b0_original_r224_nchw.mxq"),
    ),
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b1_original_r224_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b1_original_r224_nchw.mxq"),
    ),
]

for onnx_path, mxq_path in MODELS:
    if not os.path.exists(onnx_path):
        print(f"\n[SKIP] ONNX 없음: {onnx_path}")
        continue

    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path} (재컴파일하려면 파일 삭제)")
        continue

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
        # in_dformats 생략: 컴파일러가 HWC 기본값으로 cpu_offload 경계마다
        # NCHW↔NHWC 변환 op을 자동 삽입하도록 함. "NCHW" 지정 시 경계 변환이
        # 생략되어 NPU에 NCHW 데이터가 그대로 전달되어 결과가 깨짐.
        cpu_offload=True,
        quantize_method="Percentile",
        quantize_percentile=0.99999,
        is_quant_ch=True,
    )

    print(f"Saved: {mxq_path}")

print("\nAll done.")
