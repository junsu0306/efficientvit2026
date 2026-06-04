"""
timm EfficientViT → ONNX 변환 + 캘리브레이션 데이터 생성 스크립트

목적:
  기존 NHWC 변환 경로(PyTorch → NHWC ONNX → 패치 → MXQ)를 우회하고,
  timm pretrained 모델을 표준 NCHW ONNX로 직접 export.

배경:
  기존 NHWC ONNX 파일에서 Slice 노드의 상수가 FLOAT64로 잘못 저장되어
  양자화 시 zeropoint 오버플로우 발생. 이는 NHWC 변환 도구의 재파싱이 원인.
  표준 torch.onnx.export는 Slice 상수를 INT64으로 정상 저장함.

실행 환경:
  Docker 컨테이너 외부 (timm, torch, onnx 설치된 로컬 환경)

출력:
  efficientvit_b0_r224_nchw.onnx
  efficientvit_b1_r224_nchw.onnx
  calib_nchw/ 디렉토리 (캘리브레이션 데이터 CHW 형식)
  calib_nchw.txt

실행:
  pip install timm onnx
  python3 export_timm_to_onnx.py
"""

import os
import torch
import timm
import onnx
import numpy as np

# ─── 설정 ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # .../npu/compile
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

MODELS = [
    ("efficientvit_b0.r224_in1k", os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_timm_nchw.onnx")),
    ("efficientvit_b1.r224_in1k", os.path.join(ASSETS_DIR, "onnx", "efficientvit_b1_r224_timm_nchw.onnx")),
]
CALIB_SAVE_DIR = os.path.join(ASSETS_DIR, "calib_nchw")
CALIB_TXT      = os.path.join(BASE_DIR, "calib_nchw.txt")

# 기존 HWC 캘리브레이션 txt (없으면 랜덤 생성)
EXISTING_CALIB_TXT = os.path.join(ASSETS_DIR, "imagenet_calib", "imagenet_nhwc.txt")

# ─── 1단계: ONNX Export ───────────────────────────────────────────────────────
dummy_input = torch.randn(1, 3, 224, 224)

for model_name, onnx_path in MODELS:
    print(f"\n{'='*60}")
    print(f"Exporting {model_name}")
    print(f"{'='*60}")

    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

    # 저장된 ONNX 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX 검증 통과")
    print(f"  입력 노드 이름: {[i.name for i in onnx_model.graph.input]}")
    print(f"  입력 shape: {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")

    # Slice 상수 dtype 확인 (핵심 진단)
    float64_consts = [
        init.name for init in onnx_model.graph.initializer
        if init.data_type == 11  # DOUBLE = 11
    ]
    if float64_consts:
        print(f"  [WARNING] FLOAT64 상수 {len(float64_consts)}개 발견 — 패치 필요할 수 있음")
        for name in float64_consts[:5]:
            print(f"    {name}")
    else:
        print(f"  [OK] FLOAT64 상수 없음 — 양자화 안전 예상")

    print(f"  저장 완료: {onnx_path}")

# ─── 2단계: 캘리브레이션 데이터 txt 생성 ────────────────────────────────────
print(f"\n{'='*60}")
print(f"캘리브레이션 데이터 확인")
print(f"{'='*60}")

import glob as _glob
os.makedirs(CALIB_SAVE_DIR, exist_ok=True)

existing = sorted(_glob.glob(os.path.join(CALIB_SAVE_DIR, "*.npy")))
if existing:
    # calib_nchw/ 에 이미 CHW .npy 파일이 있으면 그대로 사용
    calib_paths = existing
    print(f"기존 CHW 캘리브레이션 데이터 사용: {len(calib_paths)}개 ({CALIB_SAVE_DIR})")
elif os.path.exists(EXISTING_CALIB_TXT):
    # HWC txt가 있으면 CHW로 변환
    print(f"기존 HWC 데이터 변환: {EXISTING_CALIB_TXT}")
    with open(EXISTING_CALIB_TXT) as f:
        src_paths = [l.strip() for l in f if l.strip()]
    calib_paths = []
    for i, src_path in enumerate(src_paths):
        arr_hwc = np.load(src_path)
        arr_chw = arr_hwc.transpose(2, 0, 1)
        save_path = os.path.join(CALIB_SAVE_DIR, f"{i:04d}.npy")
        np.save(save_path, arr_chw)
        calib_paths.append(save_path)
    print(f"  변환 완료: {len(calib_paths)}개")
else:
    # 둘 다 없으면 랜덤 생성 (정확도 낮음)
    print(f"캘리브레이션 데이터 없음 — 랜덤 데이터 생성 (정확도 낮을 수 있음)")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    calib_paths = []
    for i in range(100):
        arr = (np.random.randn(3, 224, 224).astype(np.float32) * std) + mean
        save_path = os.path.join(CALIB_SAVE_DIR, f"{i:04d}.npy")
        np.save(save_path, arr)
        calib_paths.append(save_path)
    print(f"  생성 완료: {len(calib_paths)}개")

# txt 파일 작성
with open(CALIB_TXT, "w") as f:
    for p in calib_paths:
        f.write(os.path.abspath(p) + "\n")
print(f"  txt 저장: {CALIB_TXT}")

print(f"\n완료! 다음 단계:")
print(f"  1. 생성된 .onnx 파일과 calib_nchw/ 를 Docker 컨테이너에 복사")
print(f"  2. Docker 안에서 compile_nchw.py 실행")
