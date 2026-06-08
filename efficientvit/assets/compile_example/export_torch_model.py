"""
════════════════════════════════════════════════════════════════════════════
 EfficientViT → PyTorch 모델 객체 저장 스크립트 (NPU 프로젝트용)
════════════════════════════════════════════════════════════════════════════

[목적]
  다른 프로젝트(Mobilint NPU 컴파일 파이프라인)의 compile_torch.py 가
  backend="torch" 로 mxq_compile 을 호출할 때 이 repo 의 모델을 그대로
  넘길 수 있도록, nn.Module 객체를 .pt 로 저장한다.

  timm 과 달리 이 repo 의 모델은 다른 프로젝트에서 import 할 수 없으므로
  torch.save(model, path) 로 객체째 저장한다.

[다른 프로젝트(npu)에서의 사용법]
  model = torch.load("efficientvit_b1.pt", map_location="cpu", weights_only=False)
  model.eval().cpu()
  mxq_compile(
      model=model,
      backend="torch",
      feed_dict={"x": torch.randn(1, 3, 224, 224)},
      calib_data_path=CALIB_TXT,
      save_path=mxq_path,
      ...
  )

[캘리브레이션 데이터 포맷 주의]
  backend="torch" 는 HWC (224, 224, 3) 포맷을 사용한다.
  backend="onnx" 의 CHW (3, 224, 224) 와 다르므로 calib 데이터를 분리해 관리할 것.

[실행]
  # 원본 b0
  python export_torch_model.py --source factory --model efficientvit-b0 \\
      --weight assets/checkpoints/efficientvit_cls/efficientvit_b0_original_r224.pt \\
      --output assets/torch_models/efficientvit_b0.pt

  # 원본 b1
  python export_torch_model.py --source factory --model efficientvit-b1 \\
      --weight assets/checkpoints/efficientvit_cls/efficientvit_b1_original_r224.pt \\
      --output assets/torch_models/efficientvit_b1.pt

  # reduced b1 (pruning 후 torch.save(model, ...) 로 저장된 전체 객체)
  python export_torch_model.py --source full-model \\
      --weight assets/export_models/efficientvit_b1_reduced.pt \\
      --output assets/torch_models/efficientvit_b1_reduced.pt

  # 인자 없이 실행 → 원본 b0/b1 배치 모드
  python export_torch_model.py
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from efficientvit.cls_model_zoo import create_efficientvit_cls_model  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "assets", "torch_models")

DEFAULT_MODELS = [
    (
        "efficientvit-b0",
        "assets/checkpoints/efficientvit_cls/efficientvit_b0_original_r224.pt",
        os.path.join(DEFAULT_OUT_DIR, "efficientvit_b0.pt"),
    ),
    (
        "efficientvit-b1",
        "assets/checkpoints/efficientvit_cls/efficientvit_b1_original_r224.pt",
        os.path.join(DEFAULT_OUT_DIR, "efficientvit_b1.pt"),
    ),
]


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_model(source: str, model_name: str, weight_path: str) -> torch.nn.Module:
    weight_path = _abspath(weight_path)

    if source == "factory":
        if model_name is None:
            raise SystemExit("--source factory 에는 --model 이 필요합니다.")
        model = create_efficientvit_cls_model(
            name=model_name, pretrained=True, weight_url=weight_path
        )
    elif source == "full-model":
        model = torch.load(weight_path, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise SystemExit(
                f"nn.Module 이 아닙니다: {type(model)}\n"
                "  → efficientvit_reducing.py 를 --save-full-model 로 저장했는지 확인하세요."
            )
    else:
        raise SystemExit(f"알 수 없는 --source: {source}")

    model.eval().cpu()
    return model


def run_one(source: str, model_name: str, weight_path: str, output_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"source : {source}")
    print(f"model  : {model_name or '(full-model)'}")
    print(f"weight : {weight_path}")
    print(f"output : {output_path}")
    print(f"{'='*60}")

    abs_weight = _abspath(weight_path)
    if not os.path.exists(abs_weight):
        print(f"  [SKIP] 가중치 파일 없음: {weight_path}")
        return

    model = load_model(source, model_name, weight_path)

    # 파라미터 수 출력
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {n_params / 1e6:.2f}M")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(model, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  저장 완료: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="EfficientViT → PyTorch 모델 객체 저장 (NPU 프로젝트용). "
                    "인자 없이 실행하면 원본 b0/b1 을 배치로 저장한다.",
    )
    parser.add_argument("--source", choices=["factory", "full-model"], default="factory")
    parser.add_argument("--model", type=str, default=None,
                        help="registered 모델명 (예: efficientvit-b1). --source factory 에 필요.")
    parser.add_argument("--weight", type=str, default=None, help="로드할 .pt 경로")
    parser.add_argument("--output", type=str, default=None, help="저장할 .pt 경로")
    args = parser.parse_args()

    if args.weight or args.output:
        if not (args.weight and args.output):
            raise SystemExit("단일 모드에서는 --weight 와 --output 을 함께 지정해야 합니다.")
        run_one(args.source, args.model, args.weight, args.output)
    else:
        for name, weight_path, output_path in DEFAULT_MODELS:
            run_one("factory", name, weight_path, output_path)

    print("\n완료! 저장된 .pt 파일을 NPU 프로젝트로 복사 후 compile_torch.py 에서 사용하세요.")


if __name__ == "__main__":
    main()
