"""
════════════════════════════════════════════════════════════════════════════
 EfficientViT(.pt) → NCHW ONNX 변환 스크립트  (Mobilint NPU 배포용)
════════════════════════════════════════════════════════════════════════════

[이 스크립트의 역할]
  이 repo(원본 mit-han-lab/efficientvit 정의)의 .pt 가중치를 NCHW ONNX 로
  export 한다. 생성된 ONNX 는 NPU 프로젝트의 Docker 안에서 qbcompiler 로
  .mxq 컴파일된다. compile 쪽은 입력 텐서를 in_dformats={"input": "NCHW"} 로
  직접 참조하므로, 입력 이름/레이아웃 규약을 반드시 지켜야 한다.

[NHWC 변환을 하지 않는 이유 — 과거 onnx_nhwc_convert.py 를 제거한 배경]
  과거에는 export 후 ONNX 앞에 Transpose 노드를 붙여 NHWC 로 재작성했는데,
  그 그래프 재작성 과정에서 Slice 노드의 INT64 상수가 FLOAT64(double)로
  잘못 저장 → qbcompiler 양자화기가 float 텐서로 인식 → range=0 → scale=0
  → zeropoint = -min/scale = ±inf → INT32 오버플로우 →
  "zeropoint=-2147483648 is out of range" 로 양자화가 항상 깨졌다.
  ⇒ 그래서 NHWC 재작성을 버리고, NCHW 그대로 export 한 뒤
     compile 단계에서 in_dformats 로 NCHW 임을 선언하는 방식으로 바꿨다.
  torch.onnx.export 직접 경로는 상수를 INT64 로 올바르게 저장하므로 안전하다.

[절대 지켜야 할 제약 — 어기면 NPU 양자화가 실패한다]
  • 입력 레이아웃은 반드시 NCHW (1, 3, 224, 224). NHWC 로 바꾸지 마라.
  • 입력 텐서 이름은 반드시 "input", 출력은 "output".
  • opset_version=13, do_constant_folding=True 유지.
  • ★ NHWC 변환/그래프 재작성 도구(onnxsim 포함)를 쓰지 마라. ★
    이 스크립트는 export 후 FLOAT64(data_type==11) 상수가 있으면 경고한다.
    그 경고가 뜨면 ONNX 가 오염된 것이니 멈추고 원인을 찾아라.

[성공 기준]
  • "ONNX 검증 통과" 출력, 입력 shape == [1, 3, 224, 224], 입력 이름 == "input"
  • "[OK] FLOAT64 상수 없음" 출력  ← [WARNING] 이 뜨면 실패로 간주

[사용법]
  # 0) 환경 (repo 루트가 PYTHONPATH 에 없어도 이 스크립트가 자동으로 추가한다)
  pip install onnx        # torch / efficientvit 는 이미 설치돼 있다고 가정

  # 1) 원본 b0 / b1 둘 다 export (인자 없이 실행 → 기본 배치 모드)
  python efficientvit/assets/export_original_to_onnx.py

  # 2) 특정 원본 모델 1개만 export (registered 모델명 사용)
  python efficientvit/assets/export_original_to_onnx.py \
      --model efficientvit-b0 \
      --weight assets/checkpoints/efficientvit_cls/efficientvit_b0_original_r224.pt \
      --output assets/export_models/efficientvit_b0_original_r224_nchw.onnx

  # 3) [나중에] pruning 후 reduce 된 모델 export
  #    efficientvit_reducing.py 를 --save-full-model 로 저장한 경우(권장):
  #    채널 수가 줄어든 구조라 registered 팩토리로는 못 만들므로 full-model 로드.
  python efficientvit/assets/export_original_to_onnx.py \
      --source full-model \
      --weight assets/export_models/efficientvit_b1_reduced.pt \
      --output assets/export_models/efficientvit_b1_reduced_r224_nchw.onnx
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys

import torch

# ─── repo 루트를 sys.path 에 추가 (PYTHONPATH 없이도 import efficientvit 가능) ──
#   이 파일: <repo>/efficientvit/assets/export_original_to_onnx.py
#   REPO_ROOT = <repo>
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import onnx  # noqa: E402  (sys.path 설정 이후 import)

from efficientvit.cls_model_zoo import create_efficientvit_cls_model  # noqa: E402

# ─── 기본 설정 ────────────────────────────────────────────────────────────────
DEFAULT_RESOLUTION = 224
DEFAULT_OPSET = 13
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "assets", "export_models")

# 인자 없이 실행했을 때 export 할 원본 모델 목록.
#   (registered 모델명, .pt 경로, 출력 .onnx 경로)
#   출력 파일명은 NPU compile 쪽이 기대하는 이름이므로 바꾸지 말 것.
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
    """상대경로는 repo 루트 기준으로 절대경로화."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_model(source: str, model_name: str, weight_path: str) -> torch.nn.Module:
    """export 대상 모델을 로드한다.

    source:
      "factory"     원본/registered 모델. create_efficientvit_cls_model 로
                    구조를 만들고 .pt 의 state_dict 를 로드한다.
      "full-model"  reduce 후 `torch.save(model, ...)` (--save-full-model) 로
                    저장된 전체 모델 객체. 채널 수가 줄어든 구조라 팩토리로는
                    재현할 수 없으므로 객체째로 로드한다.
    """
    weight_path = _abspath(weight_path)

    if source == "factory":
        if model_name is None:
            raise SystemExit("--source factory 에는 --model (registered 모델명)이 필요합니다.")
        # create_efficientvit_cls_model 이 weight_url 의 {'state_dict': ...} 래퍼와
        # 순수 state_dict 를 모두 처리한다 (load_state_dict_from_file).
        model = create_efficientvit_cls_model(
            name=model_name, pretrained=True, weight_url=weight_path
        )
    elif source == "full-model":
        # reduce 된 모델은 채널 구조가 달라 registered 팩토리로 재현 불가 →
        # 저장된 nn.Module 객체를 그대로 로드한다. (신뢰할 수 있는 자체 체크포인트만)
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
    """모델을 NCHW ONNX 로 export 하고 성공 기준을 검증한다. 성공 시 True."""
    onnx_path = _abspath(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    dummy_input = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=opset,         # ← 유지 (13)
            input_names=["input"],       # ← 유지 (compile 이 참조)
            output_names=["output"],     # ← 유지
            do_constant_folding=True,    # ← 유지
        )

    # ── 성공 기준 검증 ────────────────────────────────────────────────────────
    ok = True
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    in_name = onnx_model.graph.input[0].name
    in_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"  ONNX 검증 통과 | 입력 '{in_name}' shape={in_shape}")
    if in_name != "input" or in_shape != [1, 3, resolution, resolution]:
        print(f"  [WARNING] 입력 이름/shape 가 기대값(input, [1,3,{resolution},{resolution}])과 다릅니다!")
        ok = False

    # FLOAT64(=11) 상수가 있으면 양자화 zeropoint 오버플로우로 컴파일이 깨진다.
    float64_consts = [i.name for i in onnx_model.graph.initializer if i.data_type == 11]
    if float64_consts:
        print(f"  [WARNING] FLOAT64 상수 {len(float64_consts)}개 발견 — 양자화 실패 위험. 멈추고 원인 확인.")
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
    parser.add_argument("--source", choices=["factory", "full-model"], default="factory",
                        help="factory: registered 원본 모델 / full-model: reduce 된 전체 모델 객체")
    parser.add_argument("--model", type=str, default=None,
                        help="registered 모델명 (예: efficientvit-b0). --source factory 에 필요.")
    parser.add_argument("--weight", type=str, default=None, help="로드할 .pt 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 .onnx 경로")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    # 단일 모델 모드: --weight 또는 --output 이 주어지면 그것만 export.
    if args.weight or args.output:
        if not (args.weight and args.output):
            raise SystemExit("단일 모델 모드에서는 --weight 와 --output 을 함께 지정해야 합니다.")
        ok = run_one(args.source, args.model, args.weight, args.output, args.resolution, args.opset)
        sys.exit(0 if ok else 1)

    # 기본 배치 모드: 원본 b0 / b1 둘 다 export.
    results = []
    for name, weight_path, onnx_path in DEFAULT_MODELS:
        results.append(run_one("factory", name, weight_path, onnx_path, args.resolution, args.opset))

    print(f"\n완료! 생성된 ONNX 를 NPU 프로젝트로 가져가 qbcompiler 로 .mxq 컴파일하세요.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
