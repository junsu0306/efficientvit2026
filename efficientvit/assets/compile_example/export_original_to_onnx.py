"""
════════════════════════════════════════════════════════════════════════════
 EfficientViT(.pt) → NCHW ONNX 변환 스크립트  (Mobilint NPU 배포용)
════════════════════════════════════════════════════════════════════════════

[이 파일이 왜 여기 있나]
  compile_example/ 디렉토리에서 바로 실행할 수 있도록 둔 사본.
  실제 구현은 ../export_original_to_onnx.py 와 동일하다.

[절대 지켜야 할 제약 — 어기면 NPU 양자화가 실패한다]
  • 입력 레이아웃은 반드시 NCHW (1, 3, 224, 224). NHWC 로 바꾸지 마라.
  • 입력 텐서 이름은 반드시 "input", 출력은 "output".
  • opset_version=11, do_constant_folding=True 유지.
  • ★ NHWC 변환/그래프 재작성 도구(onnxsim 포함)를 쓰지 마라. ★

[성공 기준]
  • "ONNX 검증 통과" 출력, 입력 shape == [1, 3, 224, 224], 입력 이름 == "input"
  • "[OK] FLOAT64 상수 없음" 출력  ← [WARNING] 이 뜨면 실패로 간주

[실행]
  # compile_example/ 디렉토리에서 바로 실행 가능
  python export_original_to_onnx.py

  # 특정 모델 1개만
  python export_original_to_onnx.py \
      --model efficientvit-b1 \
      --weight assets/checkpoints/efficientvit_cls/efficientvit_b1_original_r224.pt \
      --output assets/export_models/efficientvit_b1_original_r224_nchw.onnx
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys

import torch

# compile_example/ 는 repo 루트에서 3단계 아래: efficientvit/assets/compile_example/
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import onnx  # noqa: E402

from efficientvit.cls_model_zoo import create_efficientvit_cls_model  # noqa: E402

DEFAULT_RESOLUTION = 224
DEFAULT_OPSET = 11
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "assets", "export_models")

DEFAULT_MODELS = [
    (
        "efficientvit-b0",
        "assets/checkpoints/efficientvit_cls/efficientvit_b0_original_r224.pt",
        os.path.join(DEFAULT_OUT_DIR, "efficientvit_b0_original_r224_nchw.onnx"),
    ),
    (
        "efficientvit-b1",
        "assets/checkpoints/efficientvit_cls/efficientvit_b1_original_r224.pt",
        os.path.join(DEFAULT_OUT_DIR, "efficientvit_b1_original_r224_nchw.onnx"),
    ),
]


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_model(source: str, model_name: str, weight_path: str) -> torch.nn.Module:
    weight_path = _abspath(weight_path)

    if source == "factory":
        if model_name is None:
            raise SystemExit("--source factory 에는 --model (registered 모델명)이 필요합니다.")
        model = create_efficientvit_cls_model(
            name=model_name, pretrained=True, weight_url=weight_path
        )
    elif source == "full-model":
        model = torch.load(weight_path, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise SystemExit(
                f"--source full-model 인데 nn.Module 이 아닙니다: {type(model)}\n"
                "  → efficientvit_reducing.py 를 --save-full-model 로 저장했는지 확인하세요."
            )
    else:
        raise SystemExit(f"알 수 없는 --source: {source} (factory | full-model)")

    model.eval()
    return model


def export_to_onnx(model: torch.nn.Module, onnx_path: str, resolution: int, opset: int) -> bool:
    onnx_path = _abspath(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    dummy_input = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

    # shape inference: 중간 텐서 shape 메타데이터 추가 (그래프 구조 변경 없음)
    onnx_model = onnx.load(onnx_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, onnx_path)

    ok = True
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    in_name = onnx_model.graph.input[0].name
    in_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"  ONNX 검증 통과 | 입력 '{in_name}' shape={in_shape}")
    if in_name != "input" or in_shape != [1, 3, resolution, resolution]:
        print(f"  [WARNING] 입력 이름/shape 가 기대값(input, [1,3,{resolution},{resolution}])과 다릅니다!")
        ok = False

    float64_consts = [i.name for i in onnx_model.graph.initializer if i.data_type == 11]
    if float64_consts:
        print(f"  [WARNING] FLOAT64 상수 {len(float64_consts)}개 발견 — 양자화 실패 위험.")
        for n in float64_consts[:5]:
            print(f"    {n}")
        ok = False
    else:
        print("  [OK] FLOAT64 상수 없음 — 양자화 안전 예상")

    print(f"  저장 완료: {onnx_path}")
    return ok


def run_one(source: str, model_name: str, weight_path: str, onnx_path: str, resolution: int, opset: int) -> bool:
    print(f"\n{'='*60}")
    print(f"Exporting [{source}] {model_name or '(full-model)'}")
    print(f"  weight: {weight_path}")
    print(f"  onnx  : {onnx_path}")
    print(f"{'='*60}")

    if not os.path.exists(_abspath(weight_path)):
        print(f"  [SKIP] 가중치 파일 없음: {weight_path}")
        return False

    model = load_model(source, model_name, weight_path)
    return export_to_onnx(model, onnx_path, resolution, opset)


def main():
    parser = argparse.ArgumentParser(
        description="EfficientViT .pt → NCHW ONNX (Mobilint NPU 배포용). "
                    "인자 없이 실행하면 원본 b0/b1 을 모두 export 한다.",
    )
    parser.add_argument("--source", choices=["factory", "full-model"], default="factory")
    parser.add_argument("--model", type=str, default=None,
                        help="registered 모델명 (예: efficientvit-b1). --source factory 에 필요.")
    parser.add_argument("--weight", type=str, default=None, help="로드할 .pt 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 .onnx 경로")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    if args.weight or args.output:
        if not (args.weight and args.output):
            raise SystemExit("단일 모델 모드에서는 --weight 와 --output 을 함께 지정해야 합니다.")
        ok = run_one(args.source, args.model, args.weight, args.output, args.resolution, args.opset)
        sys.exit(0 if ok else 1)

    results = []
    for name, weight_path, onnx_path in DEFAULT_MODELS:
        results.append(run_one("factory", name, weight_path, onnx_path, args.resolution, args.opset))

    print(f"\n완료! 생성된 ONNX 를 NPU 프로젝트로 가져가 qbcompiler 로 .mxq 컴파일하세요.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
