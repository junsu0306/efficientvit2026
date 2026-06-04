"""
════════════════════════════════════════════════════════════════════════════
 원본(mit-han-lab/efficientvit) EfficientViT .pt → NCHW ONNX 변환 스크립트
════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────┐
│  ▶▶▶ 이 repo 의 CLAUDE CODE 에게 보내는 당부 (반드시 먼저 읽을 것) ◀◀◀   │
└────────────────────────────────────────────────────────────────────────┘

[이 파일이 왜 여기 있나]
  이 스크립트는 다른 프로젝트(Mobilint NPU 배포 파이프라인)에서 가져온 것이다.
  거기서는 timm EfficientViT 를 NCHW ONNX 로 export → Mobilint qbcompiler 로
  .mxq 컴파일 → Aries2 NPU 추론까지 성공시켰다.
  그런데 우리가 배포하려는 가중치는 timm 이 아니라 "이 repo(원본)" 의 .pt 라서,
  timm 으로는 절대 못 읽는다(키 이름이 완전히 다름). 그래서 export 단계만
  "이 repo 안에서" 원본 모델 정의로 다시 만들어야 한다. 그게 이 파일의 역할이다.

[너(이 repo 의 Claude Code)가 해줘야 할 일]
  1. build_model() 안의 모델 생성 코드를 "이 repo 의 실제 API" 에 맞게 연동해라.
     - 아래 코드는 `from efficientvit.cls_model_zoo import create_cls_model` 를
       가정했지만, repo 버전마다 함수 이름/위치/시그니처가 다르다.
       (예: create_efficientvit_cls_model, efficientvit_cls_b0(), name="efficientvit-b0" 등)
     - repo 의 cls_model_zoo / model zoo 코드를 직접 열어 확인하고 맞춰라.
  2. WEIGHTS 경로를 이 repo 에서 .pt 가 실제 있는 위치로 바꿔라.
  3. 그대로 실행해서 ONNX 가 정상 생성되는지(아래 "성공 기준") 확인해라.
  4. 생성된 .onnx 파일명은 절대 바꾸지 마라 (compile 쪽이 이 이름을 기대함):
       efficientvit_b0_original_r224_nchw.onnx
       efficientvit_b1_original_r224_nchw.onnx

[절대 지켜야 할 제약 — 어기면 NPU 양자화가 실패한다]
  • 입력 레이아웃은 반드시 NCHW (1, 3, 224, 224). NHWC 로 바꾸지 마라.
  • 입력 텐서 이름은 반드시 "input", 출력은 "output".
    (compile 쪽이 in_dformats={"input": "NCHW"} 로 이 이름을 직접 참조한다.)
  • opset_version=13, do_constant_folding=True 유지.
  • ★ NHWC 변환 도구/그래프 재작성 도구를 절대 쓰지 마라. ★
    과거에 NHWC 변환 도구가 Slice 노드의 INT64 상수를 FLOAT64(double)로
    잘못 저장 → qbcompiler 양자화기가 그걸 float 텐서로 인식 →
    range=0 → scale=0 → zeropoint = -min/scale = ±inf → INT32 오버플로우 →
    "zeropoint=-2147483648 is out of range" 로 양자화가 항상 깨졌다.
    torch.onnx.export 직접 경로는 상수를 INT64 로 올바르게 저장하므로 안전하다.
  • 그래서 이 스크립트는 export 후 FLOAT64(data_type==11) 상수가 있으면 경고한다.
    그 경고가 뜨면 ONNX 가 오염된 것이니 멈추고 원인을 찾아라.

[성공 기준 (이 스크립트 실행 후 확인할 것)]
  • "ONNX 검증 통과" 출력, 입력 shape == [1, 3, 224, 224], 입력 이름 == "input"
  • "[OK] FLOAT64 상수 없음" 출력  ← [WARNING] 이 뜨면 실패로 간주
  • efficientvit_b0/b1_original_r224_nchw.onnx 2개 파일 생성

[이 다음 단계는 네(이 repo) 책임 밖이다 — 참고만]
  생성된 .onnx 2개를 NPU 프로젝트로 가져가서 Docker 안의 compile_nchw.py 가
  qbcompiler 로 .mxq 컴파일한다. 거기서 쓰는 캘리브레이션 데이터는 CHW (3,224,224)
  float32 .npy 이며 ImageNet 정규화(mean/std)된 전처리를 쓴다. 이 스크립트는
  ONNX 만 만들면 되고 캘리브레이션 데이터는 만들지 않아도 된다.

[사전 설치]
  pip install onnx
  # torch 와 이 repo(efficientvit) 는 이미 설치돼 있다고 가정

[실행]
  python3 export_original_to_onnx.py
════════════════════════════════════════════════════════════════════════════
"""

import os
import torch
import onnx

# ─── ▶ 연동 지점 1/2: 모델 팩토리 import ──────────────────────────────────────
# repo 버전에 따라 경로/이름이 다르다. 안 맞으면 이 repo 코드 보고 고쳐라.
try:
    from efficientvit.cls_model_zoo import create_cls_model
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "이 repo 의 cls 모델 팩토리를 import 하지 못했습니다.\n"
        "  → cls_model_zoo / model zoo 코드를 열어 올바른 함수명으로 바꾸세요.\n"
        f"(import error: {e})"
    )

# ─── 설정 ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "onnx_export")   # 생성된 ONNX 가 여기 저장됨
os.makedirs(OUT_DIR, exist_ok=True)

# ─── ▶ 연동 지점 2/2: 가중치 경로 & repo 모델 이름 ────────────────────────────
# WEIGHT 경로를 이 repo 에서 .pt 가 실제 있는 위치로 바꿔라.
# name 은 build_model() 의 팩토리가 받는 이름에 맞춰라(예: "b0" 또는 "efficientvit-b0").
# 출력 onnx 파일명은 절대 바꾸지 말 것(compile 쪽이 이 이름을 기대함).
MODELS = [
    ("b0",
     "assets/pt/efficientvit_b0_original_r224.pt",                  # ← 경로 수정
     os.path.join(OUT_DIR, "efficientvit_b0_original_r224_nchw.onnx")),
    ("b1",
     "assets/pt/efficientvit_b1_original_r224.pt",                  # ← 경로 수정
     os.path.join(OUT_DIR, "efficientvit_b1_original_r224_nchw.onnx")),
]

RESOLUTION = 224
dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)


def build_model(name, weight_path):
    """
    ▶ 이 repo 의 실제 API 에 맞게 고쳐라.
      - create_cls_model 은 weight_url 의 {'state_dict': ...} 래퍼/순수 state_dict
        둘 다 풀어 로드한다고 가정한 코드다. repo 구현을 확인하고 맞춰라.
      - 만약 팩토리가 가중치 로딩을 안 해주면 아래처럼 직접 로드:
            model = efficientvit_cls_b0()
            sd = torch.load(weight_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            model.load_state_dict(sd)
    """
    model = create_cls_model(name=name, pretrained=True, weight_url=weight_path)
    model.eval()
    return model


for name, weight_path, onnx_path in MODELS:
    print(f"\n{'='*60}")
    print(f"Exporting original efficientvit-{name}")
    print(f"  weight: {weight_path}")
    print(f"  onnx  : {onnx_path}")
    print(f"{'='*60}")

    if not os.path.exists(weight_path):
        print(f"  [SKIP] 가중치 파일 없음 — WEIGHT 경로를 수정하세요: {weight_path}")
        continue

    model = build_model(name, weight_path)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=13,            # ← 유지
            input_names=["input"],       # ← 유지 (compile 이 참조)
            output_names=["output"],     # ← 유지
            do_constant_folding=True,    # ← 유지
        )

    # ── 성공 기준 검증 ────────────────────────────────────────────────────────
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    in_name  = onnx_model.graph.input[0].name
    in_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"  ONNX 검증 통과 | 입력 '{in_name}' shape={in_shape}")
    if in_name != "input" or in_shape != [1, 3, RESOLUTION, RESOLUTION]:
        print("  [WARNING] 입력 이름/shape 가 기대값(input, [1,3,224,224])과 다릅니다!")

    # FLOAT64(=11) 상수가 있으면 양자화 zeropoint 오버플로우로 컴파일이 깨진다.
    float64_consts = [i.name for i in onnx_model.graph.initializer if i.data_type == 11]
    if float64_consts:
        print(f"  [WARNING] FLOAT64 상수 {len(float64_consts)}개 발견 — 양자화 실패 위험. 멈추고 원인 확인.")
        for n in float64_consts[:5]:
            print(f"    {n}")
    else:
        print(f"  [OK] FLOAT64 상수 없음 — 양자화 안전 예상")

    print(f"  저장 완료: {onnx_path}")

print(f"\n완료! 생성된 ONNX 를 NPU 프로젝트의 assets/onnx/ 로 복사한 뒤 compile_nchw.py 실행.")
