"""
════════════════════════════════════════════════════════════════════════════
 mit-han-lab EfficientViT → HuggingFace/timm 형식 변환 스크립트
════════════════════════════════════════════════════════════════════════════

[배경]
  ONNX 경로(export → qbcompiler)는 Slice 노드 관련 양자화 에러가 발생한다.
  qbcompiler 는 HuggingFace 형식(config.json + 가중치 파일)을 직접 받으면
  에러 없이 .mxq 컴파일이 된다.

  이 스크립트는 이 repo 의 mit-han-lab 모델 가중치를 timm 의 key 이름 규칙으로
  변환하여 HuggingFace 형식으로 저장한다.

[key 매핑 규칙 — 원본 → timm]
  backbone.input_stem.op_list.0.*         → stem.in_conv.*
  backbone.input_stem.op_list.N.*  (N≥1) → stem.res{N-1}.*
  backbone.stages.N.op_list.M.*           → stages.N.blocks.M.*
  head.op_list.0.*                        → head.in_conv.*
  head.op_list.2.linear.weight            → head.classifier.0.weight
  head.op_list.2.norm.{weight,bias}       → head.classifier.1.{weight,bias}
  head.op_list.3.linear.{weight,bias}     → head.classifier.4.{weight,bias}
  그 외 내부 key (.conv., .norm., .main., .shortcut., .depth_conv., ...)는 동일.

[지원 케이스]
  1. original pretrained (--source factory)
     - create_efficientvit_cls_model 로 구조 생성 + .pt 로드
  2. reduced model       (--source full-model)
     - torch.save(model, ...) 로 저장된 전체 객체 로드
     - ★ qbcompiler 가 custom 채널 구성을 config.json 으로 받을 수 있는지
       확인이 필요하다. 채널 수가 기본 B1 과 다르면 qbcompiler 쪽 설정도
       조정이 필요할 수 있다. ★

[출력]
  <output_dir>/model.safetensors  (safetensors 없으면 pytorch_model.bin)
  <output_dir>/config.json

[실행]
  pip install safetensors  # (없어도 .bin 으로 fallback)

  # 원본 B0
  python export_to_hf_format.py --source factory --model efficientvit-b0 \\
      --weight /path/to/efficientvit_b0_original_r224.pt --output hf_b0

  # 원본 B1
  python export_to_hf_format.py --source factory --model efficientvit-b1 \\
      --weight /path/to/efficientvit_b1_original_r224.pt --output hf_b1

  # reduced B1 (채널 수가 줄어든 전체 객체)
  python export_to_hf_format.py --source full-model \\
      --weight /path/to/efficientvit_b1_reduced.pt --output hf_b1_reduced
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from efficientvit.cls_model_zoo import create_efficientvit_cls_model  # noqa: E402

# ── timm 모델명 추정 테이블 (last-stage 채널 수 → timm architecture 이름) ──────
# reduced 모델은 채널이 줄었으므로 가장 가까운 base 를 기준으로 한다.
_WIDTH_TO_ARCH = {
    128: "efficientvit_b0",
    256: "efficientvit_b1",
    384: "efficientvit_b2",
    512: "efficientvit_b3",
}

# ── 기본 pretrained_cfg (B 시리즈 공통) ──────────────────────────────────────
_BASE_PRETRAINED_CFG = {
    "custom_load": False,
    "input_size": [3, 224, 224],
    "fixed_input_size": False,
    "interpolation": "bicubic",
    "crop_pct": 0.95,
    "crop_mode": "center",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "num_classes": 1000,
    "pool_size": [7, 7],
    "first_conv": "stem.in_conv.conv",
    "classifier": "head.classifier.4",
}


def remap_key(key: str) -> str:
    """원본 state dict key 를 timm 형식으로 변환한다."""

    # 1) backbone.input_stem.op_list.0.* → stem.in_conv.*
    key = re.sub(r"^backbone\.input_stem\.op_list\.0\.", "stem.in_conv.", key)

    # 2) backbone.input_stem.op_list.N.* (N≥1) → stem.res{N-1}.*
    def _stem_res(m: re.Match) -> str:
        return f"stem.res{int(m.group(1)) - 1}.{m.group(2)}"

    key = re.sub(r"^backbone\.input_stem\.op_list\.([1-9]\d*)\.(.+)$", _stem_res, key)

    # 3) backbone.stages.N.op_list.M.* → stages.N.blocks.M.*
    key = re.sub(r"^backbone\.stages\.(\d+)\.op_list\.(\d+)\.", r"stages.\1.blocks.\2.", key)

    # 4) head.op_list.0.* → head.in_conv.*
    key = re.sub(r"^head\.op_list\.0\.", "head.in_conv.", key)

    # 5) head 두 번째 LinearLayer → head.classifier.0 (linear weight)
    key = re.sub(r"^head\.op_list\.2\.linear\.weight$", "head.classifier.0.weight", key)

    # 6) head 두 번째 LinearLayer norm → head.classifier.1 (LayerNorm)
    key = re.sub(r"^head\.op_list\.2\.norm\.weight$", "head.classifier.1.weight", key)
    key = re.sub(r"^head\.op_list\.2\.norm\.bias$", "head.classifier.1.bias", key)

    # 7) head 세 번째 LinearLayer → head.classifier.4
    key = re.sub(r"^head\.op_list\.3\.linear\.(weight|bias)$", r"head.classifier.4.\1", key)

    return key


def convert_state_dict(orig_sd: dict) -> dict:
    """전체 state dict 에 remap_key 를 적용한다."""
    new_sd = {}
    unmapped = []
    for k, v in orig_sd.items():
        new_k = remap_key(k)
        if new_k == k:
            unmapped.append(k)
        new_sd[new_k] = v

    if unmapped:
        print(f"  [주의] 매핑 규칙에 해당하지 않는 key {len(unmapped)}개 (그대로 유지):")
        for u in unmapped[:5]:
            print(f"    {u}")
        if len(unmapped) > 5:
            print(f"    ... 외 {len(unmapped) - 5}개")

    return new_sd


def infer_arch_and_features(model: torch.nn.Module) -> tuple[str, int, int]:
    """모델에서 timm architecture 이름, num_features, num_classes 를 추론한다."""
    num_features = model.backbone.width_list[-1]
    # head 마지막 LinearLayer 의 out_features → num_classes
    num_classes = model.head.op_list[-1].linear.out_features
    # 채널 수로 base 아키텍처 추정
    arch = _WIDTH_TO_ARCH.get(num_features)
    if arch is None:
        # 정확히 일치하는 것이 없으면 가장 가까운 것으로
        closest = min(_WIDTH_TO_ARCH.keys(), key=lambda w: abs(w - num_features))
        arch = _WIDTH_TO_ARCH[closest]
        print(f"  [주의] num_features={num_features} 가 표준 채널 수와 다릅니다.")
        print(f"         → {arch} 을 base 로 사용합니다 (reduced 모델).")
        print(f"         qbcompiler 가 커스텀 채널 구성을 받을 수 있는지 별도 확인 필요.")
    return arch, num_features, num_classes


def make_config(arch: str, num_features: int, num_classes: int) -> dict:
    cfg = _BASE_PRETRAINED_CFG.copy()
    cfg["num_classes"] = num_classes
    cfg["tag"] = "r224_in1k"
    return {
        "architecture": arch,
        "num_classes": num_classes,
        "num_features": num_features,
        "global_pool": "avg",
        "pretrained_cfg": cfg,
    }


def load_model(source: str, model_name: str | None, weight_path: str) -> torch.nn.Module:
    if source == "factory":
        if model_name is None:
            raise SystemExit("--source factory 에는 --model 이 필요합니다.")
        model = create_efficientvit_cls_model(
            name=model_name, pretrained=True, weight_url=weight_path
        )
    elif source == "full-model":
        model = torch.load(weight_path, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise SystemExit(f"nn.Module 이 아닙니다: {type(model)}")
    else:
        raise SystemExit(f"알 수 없는 --source: {source}")
    model.eval()
    return model


def save_weights(state_dict: dict, out_dir: str) -> str:
    """safetensors 우선으로 저장. 없으면 pytorch_model.bin."""
    try:
        from safetensors.torch import save_file

        path = os.path.join(out_dir, "model.safetensors")
        save_file(state_dict, path)
        return path
    except ImportError:
        path = os.path.join(out_dir, "pytorch_model.bin")
        torch.save(state_dict, path)
        return path


def run(source: str, model_name: str | None, weight_path: str, out_dir: str):
    print(f"\n{'='*60}")
    print(f"source : {source}")
    print(f"model  : {model_name or '(full-model)' }")
    print(f"weight : {weight_path}")
    print(f"output : {out_dir}")
    print(f"{'='*60}")

    if not os.path.exists(weight_path):
        raise SystemExit(f"가중치 파일이 없습니다: {weight_path}")

    os.makedirs(out_dir, exist_ok=True)

    model = load_model(source, model_name, weight_path)
    arch, num_features, num_classes = infer_arch_and_features(model)
    print(f"  arch={arch}, num_features={num_features}, num_classes={num_classes}")

    orig_sd = model.state_dict()
    print(f"  원본 key 수: {len(orig_sd)}")

    timm_sd = convert_state_dict(orig_sd)
    print(f"  변환 후 key 수: {len(timm_sd)}")

    weight_path_out = save_weights(timm_sd, out_dir)
    print(f"  가중치 저장: {weight_path_out}")

    cfg = make_config(arch, num_features, num_classes)
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  config.json 저장: {cfg_path}")
    print(f"  완료!")


def main():
    parser = argparse.ArgumentParser(
        description="mit-han-lab EfficientViT .pt → HuggingFace/timm 형식 변환"
    )
    parser.add_argument(
        "--source", choices=["factory", "full-model"], default="factory",
        help="factory: registered 원본 모델 / full-model: reduce 된 전체 객체"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="registered 모델명 (예: efficientvit-b1). --source factory 에 필요.")
    parser.add_argument("--weight", type=str, required=True, help="로드할 .pt 경로")
    parser.add_argument("--output", type=str, required=True, help="출력 디렉토리")
    args = parser.parse_args()

    run(args.source, args.model, args.weight, args.output)


if __name__ == "__main__":
    main()
