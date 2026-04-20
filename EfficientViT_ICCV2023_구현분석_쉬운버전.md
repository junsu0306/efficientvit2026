# EfficientViT (ICCV 2023) — 쉽게 풀어 쓴 구현 분석

이 문서는 [EfficientViT_ICCV2023_구현분석.md](EfficientViT_ICCV2023_구현분석.md) 의
**쉬운 버전**입니다. 원본이 간결한 요약 성격이라면, 이 문서는
**딥러닝/ViT 에 익숙하지 않은 사람도 따라갈 수 있도록** 용어 하나하나를
풀어 쓰고, "왜 이렇게 설계했는지"를 자세히 설명합니다.

논문: *"EfficientViT: Lightweight Multi-Scale Attention for
High-Resolution Dense Prediction"*, Cai et al., ICCV 2023.

---

## 0. 읽기 전에: 배경 지식 속성 정리

아래 용어들이 이 문서에 반복적으로 나옵니다. 이미 알고 있다면 건너뛰어도
됩니다.

### 0.1 CNN / ViT / 어텐션

- **CNN (Convolutional Neural Network)**: 이미지의 **작은 주변 영역**만 보고
  특징을 뽑는 네트워크. 커널이 움직이며 국소적 패턴(엣지, 텍스처 등)을 잡습니다.
  **장점**: 효율적, 국소 패턴에 강함. **단점**: 멀리 떨어진 픽셀끼리 관계를
  직접 보기 어렵습니다.
- **ViT (Vision Transformer)**: 이미지를 **패치**로 쪼개 문장의 단어처럼
  다루고, **Self-Attention** 으로 모든 패치 쌍의 관계를 계산합니다.
  **장점**: 전역(global) 정보를 잘 봅니다. **단점**: 패치 수 N 에 대해
  계산량이 **O(N²)**. 이미지가 커지면(= 고해상도) 폭발적으로 느려집니다.
- **Self-Attention(셀프 어텐션)**: 각 토큰이 "나는 다른 어떤 토큰과 얼마나
  관련 있나?"를 Softmax 가중치로 계산하고, 그 가중치로 값을 섞는 연산.
  수식적으로 `Attention(Q,K,V) = Softmax(QKᵀ / √d) · V`.
- **Q, K, V (Query, Key, Value)**: 입력 토큰을 세 가지 역할로 투영한 벡터.
  Q와 K의 내적이 "얼마나 관련 있는지(점수)", V는 "실제로 전달할 내용"입니다.

### 0.2 분류 vs 분할

- **Classification(분류)**: 이미지 한 장 → 라벨 하나 (예: "고양이").
  출력 크기가 작아도 OK. **ImageNet-1K** (1,000 클래스)가 표준 벤치.
- **Segmentation(분할)**: 이미지 **픽셀 하나하나**에 라벨 (예: "이 픽셀은
  도로, 저 픽셀은 사람"). 출력도 2D 맵. **고해상도 입력** 이 필요하고
  출력도 크므로 **어텐션 비용이 치명적**입니다.
  - **Cityscapes**: 도시 도로 장면, 19 클래스, `1024×2048` 해상도.
  - **ADE20K**: 다양한 장면, 150 클래스.
  - **mIoU(mean Intersection-over-Union)**: 분할 품질 지표. 클래스별로
    (예측과 정답의 겹친 영역) / (합친 영역) 을 계산해 평균.

### 0.3 효율적 컨볼루션 블록들

MobileNet 계열에서 나온 **경량 빌딩블록**이 이 저장소 전반에 쓰입니다.

- **Pointwise Conv (1×1 Conv)**: 공간은 그대로, **채널만 섞음**. 파라미터가
  `C_in × C_out` 으로 작고 빠릅니다.
- **Depthwise Conv (k×k, groups=C)**: **채널마다 독립적으로** 공간 필터링.
  일반 컨볼루션보다 연산량이 `C` 배 작음. 대신 채널 간 섞임은 없음.
- **Depthwise-Separable Conv (DSConv)**: `Depthwise → Pointwise` 조합.
  "공간은 따로, 채널은 따로" 처리해서 가볍게 일반 컨볼루션 흉내.
- **MBConv (MobileNetV2 Inverted Residual)**:
  `1×1 확장(expand) → k×k Depthwise → 1×1 축소(project)`.
  중간 채널을 잠깐 부풀렸다가(예: 4배) 다시 줄이는 "병목(bottleneck)" 구조.
  잔차(residual) 연결 포함.
- **FusedMBConv**: 앞의 `1×1 확장 + k×k Depthwise` 두 단계를 **하나의 일반
  k×k Conv 로 합친** 변형. 초반 스테이지처럼 채널이 작을 때 더 빠릅니다.
- **GLU (Gated Linear Unit)**: `output = A ⊙ σ(B)` 처럼 한쪽을 **게이트
  (0~1)** 로 써서 다른 쪽을 선택적으로 통과시키는 구조. Transformer FFN 에
  자주 쓰임. 본문의 **GLUMBConv** 는 MBConv 에 이 아이디어를 섞은 버전.

### 0.4 정규화·활성함수·Drop

- **BatchNorm(BN)** / **LayerNorm(LN)**: 특정 차원의 평균·분산으로 정규화.
  BN 은 배치, LN 은 채널 전체를 정규화. `bn2d`/`ln2d` 는 2D 맵 대상.
- **ReLU / ReLU6 / Hard-Swish(hswish) / SiLU(swish) / GELU**: 모두 활성함수.
  - `hswish(x) = x · ReLU6(x+3)/6`: swish 의 모바일 친화 근사. 속도 빠름.
  - `GELU`: `x · Φ(x)` 로 부드러운 활성. Transformer 에서 표준.
- **DropPath (Stochastic Depth)**: 학습 시 **잔차 블록을 확률적으로
  통째로 건너뜀**. 일반 Dropout 은 뉴런을 끄지만 DropPath 는 블록 자체를
  꺼서 얕은 서브네트워크를 앙상블하는 효과.
- **EMA (Exponential Moving Average)**: 파라미터 사본을 `θ̄ ← d·θ̄ + (1-d)·θ`
  로 천천히 갱신. 평가에는 EMA 사본을 쓰면 더 안정적·정확한 경향.

### 0.5 학습 트릭

- **Label Smoothing**: 원-핫 타겟 `[0,0,1,0,...]` 을 `[ε/K, ..., 1-ε+ε/K, ...]`
  로 살짝 흐려서 **과신(overconfidence) 방지**.
- **Mixup / CutMix**: 두 장의 이미지·라벨을 섞어(α-비율) **가상의 새로운
  샘플**을 만드는 증강.
  - Mixup: `x = λ·x1 + (1-λ)·x2`, `y = λ·y1 + (1-λ)·y2` (픽셀 선형 혼합).
  - CutMix: 한 이미지의 박스를 잘라 다른 이미지에 붙임, 라벨은 면적 비율.
- **BCEWithLogitsLoss (BCE)**: 일반적으로 분류는 CrossEntropy 를 쓰는데,
  EfficientViT 처럼 **BCE** 를 쓰면 각 클래스를 독립적 이진 분류로 다룹니다.
  멀티해상도·강한 증강과 궁합이 좋은 경향.
- **MESA (Masked/Multi-Epoch Self-distillation Adaptation)**: 학습 후반부에
  **EMA 모델의 예측**을 **소프트 타겟**으로 삼아 본 모델을 다시 맞추는
  **자기지식증류(self-distillation)**. 과적합을 줄이고 일반화를 올리는 용도.
- **AMP (Automatic Mixed Precision)**: `fp16`/`bf16` 으로 연산을 섞어 속도·메모리
  절감. 민감한 구간은 `fp32` 로 되돌리는 안전장치가 필요 (본 저장소는
  LiteMLA 에 명시적으로 적용).
- **Weight Decay + `no_wd_keys`**: L2 정규화를 걸되 **BN·bias 에는 걸지
  않음**. BN 스케일/bias 에 WD 를 걸면 학습이 불안정해지는 게 정설.

### 0.6 FPN / 멀티스케일

- **FPN (Feature Pyramid Network)**: 백본의 여러 스테이지(= 여러 해상도)
  피처를 위에서 아래로 연결해 **큰 물체, 작은 물체를 동시에** 잘 잡는
  고전적 패턴. 분할/검출 헤드에서 표준.
- 이 저장소의 SegHead 는 **경량 FPN**: 1×1 Conv 로 채널 맞춘 뒤 업샘플 →
  **덧셈(add)** 으로 합칩니다.

---

## 1. 저장소에서 우리가 볼 파일

이 저장소에는 SAM, DC-AE, Diffusion, GazeSAM 도 있지만, 이 문서는
**ICCV 2023 의 분류/분할** 만 다룹니다.

```
efficientvit/
├── cls_model_zoo.py              # 분류 모델 '카탈로그' + 체크포인트 경로
├── seg_model_zoo.py              # 분할 모델 '카탈로그'
├── models/
│   ├── efficientvit/
│   │   ├── backbone.py           # 실제 뼈대: B/L 시리즈 Backbone
│   │   ├── cls.py                # 분류 헤드 + 전체 분류 모델
│   │   └── seg.py                # 분할 헤드 + 전체 분할 모델
│   ├── nn/
│   │   ├── ops.py                # 모든 빌딩블록 (Conv, MBConv, LiteMLA …)
│   │   ├── act.py                # 활성함수 레지스트리
│   │   ├── norm.py               # 정규화 레지스트리
│   │   └── drop.py               # DropPath
│   └── utils/                    # 잡다한 유틸
└── clscore/                      # 분류 '학습 파이프라인'
    ├── data_provider/imagenet.py
    └── trainer/
        ├── cls_trainer.py        # 실제 학습 루프
        └── cls_run_config.py

applications/
├── efficientvit_cls/             # 분류: 학습+평가 스크립트 + YAML 설정
└── efficientvit_seg/             # 분할: 평가+데모 (※ 학습 스크립트 없음)
```

> **한 줄 요약**: 분류는 "학습부터 평가까지" 전부 있고, 분할은 **사전학습
> 가중치로 평가/추론만** 지원됩니다. 분할 학습을 재현하려면 mmSegmentation
> 같은 외부 프레임워크가 필요합니다.

---

## 2. 핵심 아이디어: 왜 **선형** 어텐션인가?

### 2.1 고해상도에서 ViT 가 느려지는 이유

일반 Softmax 어텐션은 `(N×D) · (D×N) = N×N` 짜리 **점수 행렬**을
만들어야 합니다. 여기서 `N = H·W` (토큰 개수).

- `224×224` 입력을 패치 16 으로 자르면 `N = 14·14 = 196`. 아직 할만 합니다.
- 분할용 `1024×2048` 에서 stride 8 피처맵이면 `N = 128·256 = 32,768`.
  점수 행렬은 `32,768 × 32,768 ≈ 10.7억 개`. GPU 가 비명을 지릅니다.

이게 바로 **O(N²) 문제**이고, ViT 가 고해상도 **밀집 예측(dense prediction)**
에 직접 쓰이기 어려운 이유입니다.

### 2.2 해법: "Softmax 를 쓰지 않는다"

Softmax 는 비선형이라 `Softmax(QKᵀ)·V` 의 계산 순서를 바꿀 수 없습니다.
그런데 만약 `ϕ(Q), ϕ(K)` 같은 **양의(positive) 커널**로 바꾸면:

```
Softmax(QKᵀ) · V          # O(N²·D)
↓   Softmax → ϕ(·)ϕ(·)ᵀ 로 근사
(ϕ(Q) · ϕ(K)ᵀ) · V
= ϕ(Q) · (ϕ(K)ᵀ · V)     # 결합법칙! O(N·D²)
```

마지막 줄에서 **N 이 사라집니다**. N 이 커도 D(채널 차원)만 한정적이면
연산량이 **N 에 선형**. 이게 "Linear Attention" 의 기본 원리입니다.

EfficientViT 는 이 `ϕ` 로 **아주 단순한 `ReLU`** 를 씁니다. `ReLU` 는
빠르고, 하드웨어 친화적이며, 양수 보장이 자연스럽게 되죠.

### 2.3 정규화 트릭 (코드에서 나오는 `F.pad(..., value=1)`)

Softmax 는 스스로 합이 1이 되지만 선형 어텐션은 그게 안 됩니다. 그래서
**분모로 나눠주는 정규화**가 따로 필요합니다. 코드에서 이렇게 합니다:

```python
# v 에 상수 1을 "한 줄 더" 붙여둔다 (차원 하나 증가)
v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)

# 확장된 v 로 (v·Kᵀ)·Q 를 계산하면,
# 출력의 "마지막 채널"에 자동으로 K·Q 의 '가중치 합'이 들어간다.
vk = torch.matmul(v, trans_k)
out = torch.matmul(vk, q)

# 마지막 채널로 나눠주면 정규화 완료
out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
```

수식 두 번 쓸 걸 텐서 조작 한 줄로 해치운 **깔끔한 트릭**입니다.

### 2.4 그런데 단순 선형 어텐션은 **수용 범위**가 약하다

`ReLU` 커널을 그냥 쓰면 Softmax 만큼 "특정 위치에 집중"하지 못합니다.
그래서 EfficientViT 는 **멀티스케일 집계**를 덧붙입니다:

```
Q,K,V 각각에 대해:
  원본
  + 5×5 depthwise conv  (이웃 25 픽셀 정보를 미리 섞어둠)
  + (필요시 더 큰 스케일)
  → concat
```

즉 Q/K/V 를 **여러 수용 범위의 버전**으로 확장한 뒤 선형 어텐션을
돌립니다. "약해진 집중력" 을 **공간적 컨텍스트를 미리 녹여 넣어** 보완.

### 2.5 `LiteMLA` 전체 그림

[efficientvit/models/nn/ops.py:518-668](efficientvit/models/nn/ops.py#L518-L668)

단계별로 정리:

1. **QKV 투영** — `1×1 Conv` 로 `in_ch → 3·heads·dim`.
2. **멀티스케일 집계** — `scales=(5,)` 이면 5×5 depthwise conv 한 가지.
   Q/K/V 를 (원본) + (5×5 로 섞은 버전) 으로 확장.
3. **ReLU 커널 선형 어텐션** — `(V·Kᵀ)·Q`, 마지막에 정규화 트릭.
4. **저해상도 자동 전환** — `H·W > dim` 이면 선형, 작으면 수치 안정을
   위해 **이차식(quadratic)** 형태로 자동 전환.
   ```python
   if H*W > dim:
       out = relu_linear_att(qkv)   # 큰 입력: 선형
   else:
       out = relu_quadratic_att(qkv) # 작은 입력: (Q·Kᵀ)·V 도 싸니까 정확히
   ```
5. **AMP 안전장치** — `@torch.autocast(..., enabled=False)` 로 LiteMLA 내부는
   **fp32 강제**. 선형 어텐션은 중간에 매우 작은 값이 곱해져서 fp16/bf16 에서
   쉽게 0 으로 떨어집니다. 이걸 막는 장치입니다.

### 2.6 EfficientViTBlock: "전역 + 국소"

[efficientvit/models/nn/ops.py:671-729](efficientvit/models/nn/ops.py#L671-L729)

```
          ┌─────── Residual ────────┐   ┌─────── Residual ────────┐
 x ──────►│ LiteMLA (전역 관계)     │──►│ (GLU)MBConv (국소 패턴) │───► out
          └──────────────────────────┘   └──────────────────────────┘
                 context_module                 local_module
```

- **context_module (LiteMLA)**: "멀리 있는 픽셀과의 관계"를 잡습니다.
- **local_module (MBConv/GLUMBConv)**: "가까운 영역의 패턴"을 잡습니다.

이 **두 가지를 직렬로 쌓고 각각 잔차 연결** 한 게 Metaformer 스타일이고,
EfficientViT 만의 장점은 전역 모듈이 **선형 비용**이라 고해상도에 쓸 수
있다는 점입니다.

---

## 3. 빌딩블록 상세 (efficientvit/models/nn/ops.py)

각 블록의 "**입력/출력 형상**"과 "**어디에 쓰이는지**"를 같이 적겠습니다.

### 3.1 ConvLayer

[ops.py:37-78](efficientvit/models/nn/ops.py#L37-L78)

단순히 `Conv2d + Norm + Dropout + Activation` 을 묶은 **편의 래퍼**.
`same padding` 을 커널 크기로 자동 계산. 거의 모든 곳에서 이 래퍼를 사용.

### 3.2 DSConv (Depthwise-Separable Conv)

[ops.py:270-309](efficientvit/models/nn/ops.py#L270-L309)

`Depthwise k×k → Pointwise 1×1` 의 두 단계.
`expand_ratio == 1` (채널 안 부풀림) 일 때 MBConv 대신 씁니다.

### 3.3 MBConv

[ops.py:312-364](efficientvit/models/nn/ops.py#L312-L364)

```
x ──► 1×1 Conv (C → C·t)   ← "expand"
   ──► k×k Depthwise       ← 공간 필터링
   ──► 1×1 Conv (C·t → C)  ← "project"
   + shortcut
```

`t = expand_ratio` (보통 4 또는 6). **중간에 한번 부풀렸다가 다시 줄이는**
게 Inverted Residual 의 핵심.

### 3.4 FusedMBConv

[ops.py:367-410](efficientvit/models/nn/ops.py#L367-L410)

`1×1 expand + k×k depthwise` → **한 번의 k×k 일반 conv** 로 합침.
채널이 작을 때 그룹 컨볼루션 오버헤드가 오히려 손해라서, **L 시리즈 초반
스테이지**에서 선호됩니다.

### 3.5 GLUMBConv

[ops.py:413-470](efficientvit/models/nn/ops.py#L413-L470)

중간 확장에서 **두 갈래(A, B)** 를 만들고 한쪽(B)을 시그모이드로
게이트 삼아 `A ⊙ σ(B)` 로 통과시키는 MBConv. 일부 L 시리즈 블록에서 사용.

### 3.6 LiteMLA

위 §2.5 에서 자세히 설명.

### 3.7 EfficientViTBlock

위 §2.6 에서 자세히 설명.

### 3.8 ResidualBlock

[ops.py:737-767](efficientvit/models/nn/ops.py#L737-L767)

```
out = main(x) + shortcut(x)
```

`shortcut` 이 `IdentityLayer` 면 일반 잔차 연결, `None` 이면 shortcut 없음.
`main` 이 반환하는 텐서에 **DropPath** 를 적용할 수도 있습니다.

### 3.9 DAGBlock

[ops.py:770-804](efficientvit/models/nn/ops.py#L770-L804)

"여러 입력 → merge(add 또는 cat) → middle → 여러 출력" 형태. **SegHead** 는
이 클래스를 상속해서 FPN 을 구현합니다.

### 3.10 OpSequential

[ops.py:807-819](efficientvit/models/nn/ops.py#L807-L819)

`None` 인 자식은 건너뛰는 `nn.Sequential` 변형. 옵션 모듈을 자연스럽게
처리하려고 도입.

### 3.11 활성함수·정규화 레지스트리

- `act.py`: 이름(문자열)으로 활성함수를 꺼내 쓰게 한 딕셔너리.
  - **B 시리즈** 는 기본 `hswish` (모바일 친화).
  - **L 시리즈 / 일부 분할 헤드** 는 `gelu` (Transformer 스타일).
- `norm.py`: `bn2d`, `ln2d`, `ln`, `trms2d` 등록. **모델마다 BN ε 값이
  다름**:
  - B 시리즈: `1e-5`.
  - L 시리즈: `1e-7` (더 정밀한 ε, 학습 안정성 튜닝 결과).
  `set_norm_eps(model, eps)` 로 모델 생성 후 일괄 교체합니다.

---

## 4. 백본 아키텍처

백본은 "이미지를 받아 여러 스케일의 피처맵을 만들어주는" 부분입니다.
분류 헤드도 분할 헤드도 **이 백본을 공유**합니다.

### 4.1 EfficientViTBackbone (B 시리즈 — 모바일 지향)

[efficientvit/models/efficientvit/backbone.py:33-157](efficientvit/models/efficientvit/backbone.py#L33-L157)

전체 파이프라인:

```
input (3ch)
  │
  │  ┌───── input_stem ─────┐
  │  ConvLayer(3 → w0, stride 2)           # 해상도 절반
  │  + [DSConv(w0→w0) × depth_list[0]]     # 초기 정제
  │  └───────────────────────┘
  │
  ├─► stage1: [MBConv × depth_list[1]]      # 첫 번째만 stride 2
  │                                         # w0 → w1, 해상도 /4
  │
  ├─► stage2: [MBConv × depth_list[2]]      # 첫 번째만 stride 2
  │                                         # w1 → w2, 해상도 /8
  │
  ├─► stage3: MBConv(stride 2, fewer_norm=True)    # w2 → w3, /16
  │         + EfficientViTBlock × depth_list[3]    # 여기서부터 어텐션!
  │
  └─► stage4: MBConv(stride 2, fewer_norm=True)    # w3 → w4, /32
            + EfficientViTBlock × depth_list[4]
```

주요 포인트를 하나씩 풀어보면:

- **`build_local_block`** ([backbone.py:119-148](efficientvit/models/efficientvit/backbone.py#L119-L148))
  이 `expand_ratio == 1` 일 때 `DSConv` 를, 아니면 `MBConv` 를 돌려줍니다.
  **expand 가 1 이면 "부풀릴 게 없으니까"** 더 가벼운 DSConv 를 고르는 거죠.
- **`fewer_norm=True`** 는 stage3/4 의 **다운샘플링 블록에서 일부 BN 을
  제거** 합니다. 왜냐하면:
  1. 바로 다음에 LiteMLA 가 따라오는데,
  2. LiteMLA 내부에 이미 Norm 이 충분히 있고,
  3. 다운샘플 직후의 BN 은 배치 내 통계 변동이 커서 학습을 불안정하게
     만들 수 있기 때문입니다.
- **어텐션은 stage3/4 에서만** 사용. 얕은 스테이지(해상도 큰 곳)는 Conv
  로만 처리해서 비용 절감.
- **forward 는 `dict` 반환**. 분할 헤드가 `stage2/3/4` 를 모두 써야 하므로
  단일 텐서로는 부족합니다:
  ```python
  return {
      "input": x0, "stage0": x1,
      "stage1": x2, "stage2": x3,
      "stage3": x4, "stage4": x5,
      "stage_final": x5,
  }
  ```

**B 시리즈 설정** ([backbone.py:159-196](efficientvit/models/efficientvit/backbone.py#L159-L196)):

| 모델 | width_list | depth_list | dim(head당) |
|---|---|---|---|
| B0 | [8, 16, 32, 64, 128] | [1, 2, 2, 2, 2] | 16 |
| B1 | [16, 32, 64, 128, 256] | [1, 2, 3, 3, 4] | 16 |
| B2 | [24, 48, 96, 192, 384] | [1, 3, 4, 4, 6] | 32 |
| B3 | [32, 64, 128, 256, 512] | [1, 4, 6, 6, 9] | 32 |

`width_list[i]` 는 i번째 스테이지의 **채널 수**, `depth_list[i]` 는 그
스테이지에 쌓는 **블록 개수**. 예를 들어 B2 의 stage4 는 채널 384,
EfficientViTBlock 6개.

### 4.2 EfficientViTLargeBackbone (L 시리즈 — 고성능 지향)

[efficientvit/models/efficientvit/backbone.py:199-338](efficientvit/models/efficientvit/backbone.py#L199-L338)

L 시리즈는 **스테이지별로 블록 타입을 바꿀 수 있게** 설계됐습니다.
`block_list=["res","fmb","fmb","mb","att"]` 이 기본값이고, 각 토큰이:

- `"res"` → **ResBlock** (가장 단순한 잔차 Conv 블록)
- `"fmb"` → **FusedMBConv** (초기 스테이지의 저지연 친화)
- `"mb"` → **MBConv**
- `"att"` → **EfficientViTBlock** (LiteMLA 포함)

각 스테이지별로 `expand_list`, `fewer_norm_list` 를 따로 지정하고,
활성함수는 **전역 `gelu`** 로 고정됩니다.

**L 시리즈 설정** ([backbone.py:341-374](efficientvit/models/efficientvit/backbone.py#L341-L374)):

| 모델 | width_list | depth_list |
|---|---|---|
| L0 | [32, 64, 128, 256, 512] | [1, 1, 1, 4, 4] |
| L1 | [32, 64, 128, 256, 512] | [1, 1, 1, 6, 6] |
| L2 | [32, 64, 128, 256, 512] | [1, 2, 2, 8, 8] |
| L3 | [64, 128, 256, 512, 1024] | [1, 2, 2, 8, 8] |

L 시리즈는 **뒤쪽 스테이지에 어텐션 블록을 더 많이** 넣어서 표현력을
끌어올렸고, 앞쪽은 FusedMBConv 로 계산량을 아낍니다.

---

## 5. 분류 (Classification)

### 5.1 ClsHead — 아주 단순한 구조

[efficientvit/models/efficientvit/cls.py](efficientvit/models/efficientvit/cls.py)

```
백본의 stage_final (예: B2→384ch, 7×7)
  │
  ├─► ConvLayer 1×1   (384 → 2304)    # 채널 크게 확장
  ├─► AdaptiveAvgPool2d(1)             # 공간 크기 1×1 로
  ├─► Flatten → LinearLayer (LN+act)   # 2304 → 2560
  └─► LinearLayer (dropout)            # 2560 → 1000 (로짓)
```

분류에서는 **`stage_final` 하나만** 사용합니다 — 전역 풀링하기 전에
충분히 채널을 부풀리는 **"fat classifier head"** 디자인 (B3/L3 는 6k 넘는
차원까지 확장). 덕분에 백본 자체는 상대적으로 가볍게 유지.

**Head 설정 요약:**

| 모델 | in_channels | MLP width_list | 활성함수 |
|---|---|---|---|
| B0 | 128 | [1024, 1280] | hswish |
| B1 | 256 | [1536, 1600] | hswish |
| B2 | 384 | [2304, 2560] | hswish |
| B3 | 512 | [2304, 2560] | hswish |
| L1/L2 | 512 | [3072, 3200] | gelu |
| L3 | 1024 | [6144, 6400] | gelu |

### 5.2 Model Zoo — 사전학습 가중치 불러오기

[efficientvit/cls_model_zoo.py](efficientvit/cls_model_zoo.py)

```python
REGISTERED_EFFICIENTVIT_CLS_MODEL = {
    "efficientvit-b0":      (efficientvit_cls_b0, 1e-5, None),
    "efficientvit-b0-r224": (efficientvit_cls_b0, 1e-5,
                             "assets/checkpoints/efficientvit_cls/efficientvit_b0_r224.pt"),
    ...
}
```

각 튜플은 `(생성 함수, BN ε, 체크포인트 경로)` 입니다.

```python
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
model = create_efficientvit_cls_model("efficientvit-l2-r384", pretrained=True)
```

- 이름 끝의 `-r{224,256,288,320,384}` 는 **해당 해상도로 학습된 가중치**.
  멀티해상도 사전학습의 결과로 해상도별 최적 가중치가 따로 배포됩니다.
- 공식 체크포인트는 **HuggingFace `han-cai/efficientvit-cls`** 에서 받아
  `assets/checkpoints/efficientvit_cls/` 에 그대로 두면 됩니다.

### 5.3 학습 파이프라인 한 번에 훑기

#### (a) 진입 스크립트

[applications/efficientvit_cls/train_efficientvit_cls_model.py](applications/efficientvit_cls/train_efficientvit_cls_model.py)

실행할 때:

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
    applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_l2.yaml \
    --amp bf16 \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_l2_r224/
```

진행 순서:

1. **CLI 파싱** — YAML 경로가 필수. `--amp`, `--path`, 그리고 `--X.Y=Z`
   형태로 YAML 값을 덮어쓸 수 있습니다.
2. **분산 환경 초기화** — `setup.setup_dist_env(args.gpu)` 가
   `torchrun` 환경변수를 받아 `DDP` 를 설정.
3. **설정 병합** — `default.yaml` + `efficientvit_{variant}.yaml` + CLI
   오버라이드를 **재귀 병합**. 한 파일에 모두 적는 대신 공통 베이스 + 차이만
   덮어쓰는 방식.
4. **데이터** — `ImageNetDataProvider` 생성.
5. **모델** — `create_efficientvit_cls_model(pretrained=False)` →
   `apply_drop_func` 로 **DropPath 주입**.
6. **트레이너** — `ClsTrainer.prep_for_training()` →
   `trainer.train()`.

#### (b) 데이터 프로바이더

[efficientvit/clscore/data_provider/imagenet.py](efficientvit/clscore/data_provider/imagenet.py)

눈여겨볼 특징:

- **`image_size` 를 리스트로도 받음**. 예: `[128, 160, 192, 224, 256, 288]`.
  매 에폭(또는 스텝)마다 해상도를 랜덤하게 바꿔 학습 → **하나의 모델이
  여러 해상도에 강해짐**. 이게 `-r{res}` 체크포인트 라인업의 토대.
- **`data_aug`**: `randaug(n=1~2, m=3~5)`, `erase(p=0.2)` 등. RandAugment 는
  "n 개의 증강을 강도 m 으로" 무작위 적용.
- 검증은 `bicubic` 보간 + `test_crop_ratio=1.0` (크롭 안 하고 리사이즈만).

#### (c) 트레이너 — mixup/cutmix/BCE/MESA

[efficientvit/clscore/trainer/cls_trainer.py](efficientvit/clscore/trainer/cls_trainer.py)

핵심 함수들:

- **`_validate`** ([L35-75](efficientvit/clscore/trainer/cls_trainer.py#L35-L75))
  — `CrossEntropyLoss` + top-1/top-5 정확도.
- **`before_step`** ([L77-105](efficientvit/clscore/trainer/cls_trainer.py#L77-L105))
  — 매 배치 전에:
  1. 라벨 스무딩으로 라벨을 부드럽게 만들고,
  2. `mixup_config["op"]` 에서 가중치 기반 랜덤 선택 (예: mixup 확률 50%,
     cutmix 확률 50%),
  3. `Beta(α,α)` 에서 `λ` 샘플링,
  4. `mixup_aug(x, y, λ)` 또는 `cutmix_aug(x, y, λ)` 적용.
- **`run_step`** ([L107-139](efficientvit/clscore/trainer/cls_trainer.py#L107-L139))
  — AMP 컨텍스트에서 forward→loss.
  **MESA** 가 켜져 있고 `run_config.progress >= thresh` 면:
  ```python
  mesa_loss = F.binary_cross_entropy_with_logits(
      model_logits, ema_model_logits.sigmoid().detach()
  )
  loss += run_config.mesa["ratio"] * mesa_loss
  ```
  즉 학습 후반부에 **EMA 모델의 soft prediction** 을 가짜 라벨로 써서
  한 번 더 정렬합니다. 과적합 완화 + 일반화 향상.
- **`train`** ([L188-230](efficientvit/clscore/trainer/cls_trainer.py#L188-L230))
  — `run_config.bce=True` 면 `BCEWithLogitsLoss`, 아니면 CE.
  `auto_restart_thresh` 설정되어 있으면 정확도가 **갑자기 급락**하는 경우
  best 체크포인트에서 **자동 재시작**. 학습 불안정에 대한 방어선.

#### (d) RunConfig

[efficientvit/clscore/trainer/cls_run_config.py](efficientvit/clscore/trainer/cls_run_config.py)

주요 필드: `base_lr`, `label_smooth`, `mixup_config`, `bce`, `mesa`,
`weight_decay`, `no_wd_keys`, `grad_clip`, `reset_bn`, …

#### (e) YAML 설정 (default.yaml 예시)

```yaml
data_provider:
  dataset: imagenet
  data_dir: /dataset/imagenet
  image_size: [128, 160, 192, 224]   # 멀티해상도
  base_batch_size: 128               # GPU 당
  data_aug: {name: randaug, n: 1, m: 3}

run_config:
  n_epochs: 300
  warmup_epochs: 20
  base_lr: 0.00025
  lr_schedule_name: cosine           # 코사인 감쇠
  optimizer_name: adamw
  optimizer_params: {eps: 1.0e-08, betas: [0.9, 0.999]}
  weight_decay: 0.1
  no_wd_keys: [norm, bias]           # BN / bias 에는 WD 안 걸기
  grad_clip: 2.0
  reset_bn: true                     # 학습 후 BN running stats 재계산
  reset_bn_size: 16000
  eval_image_size: [224]
  label_smooth: 0.1
  mixup_config:
    op:
      - [mixup,  0.1, 1.0]           # [종류, Beta 파라미터, 적용 확률]
      - [cutmix, 0.1, 1.0]
  bce: true

backbone_drop:
  name: droppath
  drop_prob: 0.05
  linear_decay: true                 # stage 깊어질수록 drop 확률 선형 증가

ema_decay: 0.9998                    # EMA 감쇠 계수 (1에 가까울수록 천천히)
```

**`efficientvit_l2.yaml` 오버라이드**:

```yaml
run_config:
  base_lr: 0.00015
  mixup_config:
    op:
      - [mixup,  0.4, 1.0]   # L2 는 더 강한 mixup/cutmix
      - [cutmix, 0.4, 1.0]
  mesa:
    thresh: 0.25             # 학습 진행률 25% 지나면 MESA 켜짐
    ratio: 2.75              # MESA 손실 가중치

backbone_drop:
  drop_prob: 0.1
  skip: 3                    # 앞 3블록은 DropPath 면제

net_config:
  name: efficientvit-l2
  dropout: 0
```

YAML 병합 순서:
```
default.yaml (공통)
  └─ efficientvit_l2.yaml (이 파일에 적은 것만 덮어씀)
        └─ CLI 오버라이드 (--data_provider.data_dir ...)
```

### 5.4 평가

```bash
python applications/efficientvit_cls/eval_efficientvit_cls_model.py \
    --model efficientvit-l2-r384 \
    --path ~/dataset/imagenet/val
```

내부적으로 `create_efficientvit_cls_model(..., pretrained=True)` 로 모델을
불러 top-1/top-5 를 계산합니다.

---

## 6. 분할 (Segmentation)

### 6.1 SegHead — 경량 FPN + Residual Stack

[efficientvit/models/efficientvit/seg.py:30-104](efficientvit/models/efficientvit/seg.py#L30-L104)

`SegHead` 는 `DAGBlock` 을 상속받은 **다중 입력/출력 구조**:

```
백본 출력 dict ─┐
               │
  stage4 (/32) ──► 1×1 Conv (→head_width) ──► UpSample ×4 ┐
  stage3 (/16) ──► 1×1 Conv (→head_width) ──► UpSample ×2 ├─► ADD ─► middle ─► [final_expand Conv] ─► 1×1 Conv (n_classes) ─► "segout"
  stage2 (/8)  ──► 1×1 Conv (→head_width) ──► Identity    ┘                                                                      (H/8, W/8)
```

구역별로 뜯어보면:

- **inputs** ([seg.py:47-58](efficientvit/models/efficientvit/seg.py#L47-L58))
  - `fid_list = ["stage4","stage3","stage2"]` 각각에 대해 `1×1 Conv` 로
    채널을 `head_width` 로 맞추고,
  - `stride > head_stride` 면 `UpSampleLayer(factor)` 로 공간을 맞춥니다.
    (여기서 head_stride=8 이니까 stage4(/32)는 ×4, stage3(/16)는 ×2 업샘플.)
- **merge = "add"** — concat 이 아니라 **덧셈**. 파라미터 없이 가볍게 합칩니다.
- **middle** ([seg.py:60-81](efficientvit/models/efficientvit/seg.py#L60-L81))
  - `middle_op="mbconv"` 면 `MBConv`, `"fmbconv"` 면 `FusedMBConv`.
  - `head_depth` 번 반복하여 `ResidualBlock` 으로 감싼 형태.
- **outputs** ([seg.py:83-102](efficientvit/models/efficientvit/seg.py#L83-L102))
  - `final_expand` 가 주어지면 채널 추가 확장 (ADE20K 처럼 클래스가 많은
    경우).
  - 마지막 `1×1 Conv → n_classes` 가 각 픽셀에 대한 로짓을 만들어 `"segout"`
    키로 반환.

### 6.2 EfficientViTSeg

[efficientvit/models/efficientvit/seg.py:107-117](efficientvit/models/efficientvit/seg.py#L107-L117)

```python
class EfficientViTSeg(nn.Module):
    def forward(self, x):
        feed_dict = self.backbone(x)    # {"stage2":…, "stage3":…, "stage4":…, …}
        feed_dict = self.head(feed_dict)
        return feed_dict["segout"]      # (B, n_classes, H/8, W/8)
```

출력 해상도가 입력의 **1/8** 이라, 실제 평가에서는 GT 마스크 크기로
`resize(..., size=mask.shape[-2:])` 해서 맞춥니다.

### 6.3 변형별 SegHead 설정

[efficientvit/models/efficientvit/seg.py:120-341](efficientvit/models/efficientvit/seg.py#L120-L341)

기본 공통:
- `fid_list = ["stage4","stage3","stage2"]`
- `stride_list = [32, 16, 8]`, `head_stride = 8`

| 모델 | 데이터셋 | in_channels | head_width | head_depth | expand | middle_op | final_expand | n_classes | act |
|---|---|---|---|---|---|---|---|---|---|
| B0 | cityscapes | [128,64,32] | 32 | 1 | 4 | mbconv | 4 | 19 | hswish |
| B1 | cityscapes | [256,128,64] | 64 | 3 | 4 | mbconv | 4 | 19 | hswish |
| B1 | ade20k | [256,128,64] | 64 | 3 | 4 | mbconv | None | 150 | hswish |
| B2 | cityscapes | [384,192,96] | 96 | 3 | 4 | mbconv | 4 | 19 | hswish |
| B2 | ade20k | [384,192,96] | 96 | 3 | 4 | mbconv | None | 150 | hswish |
| B3 | cityscapes | [512,256,128] | 128 | 3 | 4 | mbconv | 4 | 19 | hswish |
| B3 | ade20k | [512,256,128] | 128 | 3 | 4 | mbconv | None | 150 | hswish |
| L1 | cityscapes | [512,256,128] | 256 | 3 | 1 | **fmbconv** | None | 19 | **gelu** |
| L1 | ade20k | [512,256,128] | 128 | 3 | 4 | **fmbconv** | 8 | 150 | **gelu** |
| L2 | cityscapes | [512,256,128] | 256 | **5** | 1 | fmbconv | None | 19 | gelu |
| L2 | ade20k | [512,256,128] | 128 | 3 | 4 | fmbconv | 8 | 150 | gelu |

읽는 법:

- **B 시리즈는 `MBConv + hswish`**, **L 시리즈는 `FusedMBConv + gelu`** 로
  백본과 헤드의 스타일을 통일합니다.
- **Cityscapes Head** 는 `final_expand=4 (B) / None (L)`. 해상도가 커서
  마지막에 채널을 더 부풀리는 대신 내부 채널을 크게 유지.
- **ADE20K Head** 는 `final_expand=8` (L 기준) 로 **마지막에 크게 부풀려**
  150 클래스를 분류. 클래스가 많을수록 분류기 용량이 더 필요.

### 6.4 Seg Model Zoo

[efficientvit/seg_model_zoo.py](efficientvit/seg_model_zoo.py)

```python
"efficientvit-seg-l2-cityscapes": (
    partial(efficientvit_seg_l2, dataset="cityscapes"),
    1e-7,
    "assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_cityscapes.pt",
)
```

`functools.partial` 로 `dataset` 인자를 미리 바인딩해둬서, 같은 생성 함수를
cityscapes/ade20k 두 이름으로 각각 등록합니다.

등록 총 11종:
- **Cityscapes (6)**: `{b0, b1, b2, b3, l1, l2}`
- **ADE20K (5)**: `{b1, b2, b3, l1, l2}` — B0 는 cityscapes 전용.

### 6.5 평가 스크립트 해부

[applications/efficientvit_seg/eval_efficientvit_seg_model.py](applications/efficientvit_seg/eval_efficientvit_seg_model.py)

주요 컴포넌트:

- **`SegIOU`** ([L73-106](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L73-L106))
  - `torch.histc` 로 **클래스별 intersection/union** 을 배치마다 누적하고,
    마지막에 `mIoU = mean(I / U)`.
  - 히스토그램으로 계산하기 때문에 **큰 해상도에서도 메모리 효율적**.
- **`CityscapesDataset`** ([L109-237](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L109-L237))
  - `leftImg8bit/*/*.png` (이미지) + `gtFine/*/*_labelIds.png` (라벨).
  - `label_map` ([L152-189](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L152-L189))
    에서 **원본 34개 Cityscapes 라벨을 학습용 19 클래스로 매핑** (`-1` 은 void).
  - 기본 `crop_size=(1024, 2048)` 로 리사이즈.
- **`ADE20KDataset`** ([L240-602](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L240-L602))
  - 150 클래스의 이름/컬러 팔레트 내장.
  - **짧은 변을 512 로 맞추고 긴 변은 32 의 배수로 라운딩**
    ([L578-590](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L578-L590)).
    stride 32 백본이 깔끔히 통과하도록.
- **전처리**: `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`
  (표준 ImageNet 정규화).
- **추론**:
  ```python
  with torch.inference_mode():
      out = model(x)
      if out.shape[-2:] != mask.shape[-2:]:
          out = resize(out, size=mask.shape[-2:])
      pred = out.argmax(dim=1)
      iou.update(pred, mask)
  ```
- **시각화** ([L605-620](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L605-L620))
  — `--save_path` 주면 원본 + 예측 마스크를 반투명 합성해 PNG 저장.

실행 예시:

```bash
python applications/efficientvit_seg/eval_efficientvit_seg_model.py \
    --model efficientvit-seg-l2-cityscapes \
    --path ~/dataset/cityscapes/leftImg8bit/val \
    --dataset cityscapes --crop_size 1024
```

### 6.6 데모 스크립트

[applications/efficientvit_seg/demo_efficientvit_seg_model.py](applications/efficientvit_seg/demo_efficientvit_seg_model.py)

한 장의 이미지를 받아 클래스 팔레트로 시각화.

### 6.7 분할 학습 스크립트가 없는 이유

공식 저장소 방침상 **Segmentation 학습 코드는 공개하지 않습니다.**
`applications/efficientvit_seg/` 에 `train_*.py` 가 없는 이유이고, 이
저장소로는 **사전학습 가중치로 평가/추론만** 가능합니다.

**재현하려면**: mmSegmentation 같은 분할 프레임워크에서 이 저장소의
`EfficientViTSeg` 를 모델로 등록하고, 논문의 학습 설정(crop size, 학습
epoch, 옵티마이저 등) 을 따라 구성해야 합니다.

---

## 7. 분류 vs 분할 한눈 비교

| 항목 | Classification | Segmentation |
|---|---|---|
| 모델 클래스 | `EfficientViTCls` | `EfficientViTSeg` |
| Head 구조 | Conv → Pool → MLP → MLP (분류기) | 경량 FPN → MBConv/FMBConv stack → 1×1 Conv |
| 백본 출력 사용 | `stage_final` 1개 | `stage2/3/4` 3개 |
| 데이터셋 | ImageNet-1K | Cityscapes (19), ADE20K (150) |
| 학습 스크립트 | **제공** (`train_efficientvit_cls_model.py`) | **미제공** |
| 평가 스크립트 | 제공 | 제공 |
| 출력 해상도 | 1×1 (=벡터) | 입력의 1/8 |
| BN ε (B 시리즈) | 1e-5 | 1e-5 |
| BN ε (L 시리즈) | 1e-7 | 1e-7 |
| B 시리즈 활성 | hswish | hswish |
| L 시리즈 활성 | gelu | gelu |

---

## 8. 빠른 시작 예제

### 분류 추론

```python
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
model = create_efficientvit_cls_model("efficientvit-l2-r384", pretrained=True)
model.eval().cuda()

# images: (B, 3, 384, 384), ImageNet 정규화된 상태
logits = model(images)  # (B, 1000)
```

### 분할 추론

```python
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
model = create_efficientvit_seg_model("efficientvit-seg-l2-cityscapes", pretrained=True)
model.eval().cuda()

# images: (B, 3, 1024, 2048), ImageNet 정규화된 상태
logits = model(images)        # (B, 19, 128, 256) — stride 8
pred   = logits.argmax(dim=1) # (B, 128, 256)      — 클래스 인덱스 맵
```

### 분류 학습 (L2, 8-GPU)

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
    applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_l2.yaml \
    --amp bf16 \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_l2_r224/
```

---

## 9. 체크포인트 배치

`cls_model_zoo.py` / `seg_model_zoo.py` 가 기대하는 경로:

```
assets/checkpoints/
├── efficientvit_cls/
│   ├── efficientvit_b0_r224.pt
│   ├── efficientvit_b1_r{224,256,288}.pt
│   ├── efficientvit_b2_r{224,256,288}.pt
│   ├── efficientvit_b3_r{224,256,288}.pt
│   ├── efficientvit_l1_r224.pt
│   ├── efficientvit_l2_r{224,256,288,320,384}.pt
│   └── efficientvit_l3_r{224,256,288,320,384}.pt
└── efficientvit_seg/
    ├── efficientvit_seg_{b0,b1,b2,b3,l1,l2}_cityscapes.pt
    └── efficientvit_seg_{b1,b2,b3,l1,l2}_ade20k.pt
```

HuggingFace `han-cai/efficientvit-cls`, `han-cai/efficientvit-seg` 에서
받아 위 경로에 그대로 두면 `pretrained=True` 로 바로 로드됩니다.

---

## 10. 핵심 설계 포인트 (왜 이렇게 만들었는가)

1. **O(N) 선형 어텐션이 전부의 출발점**
   - 고해상도 밀집 예측에서 O(N²) 은 치명적. ReLU 커널 + 결합법칙 재배치로
     N 을 소거.
   - 그 대신 약해진 집중력은 **멀티스케일 depthwise 집계**로 보완.

2. **EfficientViTBlock = 전역(LiteMLA) + 국소((GLU)MBConv)**
   - 두 모듈을 각각 잔차로 감싼 Metaformer 스타일.
   - 장거리 의존성과 국소 패턴을 **분업**시켜 효율과 표현력을 모두 챙김.

3. **백본이 dict 를 반환**
   - 같은 백본으로 분류(`stage_final`), 분할(`stage2/3/4`)을 함께 지원.
   - 헤드 교체로 두 과제에 재사용.

4. **B(모바일) vs L(고성능) 분리**
   - B: MBConv + hswish + BN(ε=1e-5).
   - L: FusedMBConv 혼합 + gelu + BN(ε=1e-7). **스테이지별 블록 타입**
     을 YAML 수준에서 지정 가능해서 채널-지연시간 교환이 유연.

5. **멀티해상도 학습 + 해상도별 체크포인트**
   - 같은 모델을 `r224 ~ r384` 에 대해 각기 최적화한 가중치로 배포.
   - 배포 단계에서 속도/정확도 트레이드오프를 해상도로 조정 가능.

6. **학습 트릭 풀세트**
   - label smoothing, mixup/cutmix (가중치 기반 랜덤 선택),
     `BCEWithLogitsLoss`, DropPath 선형 감쇠, EMA, AMP bf16, MESA
     자기지식증류, **BN 재계산(`reset_bn`)**, **정확도 급락 시 자동 재시작**.
   - 각각이 경험적으로 효과 입증된 요소들의 조합.

7. **분할 헤드는 경량 FPN + 작은 residual stack**
   - `1×1 projection → upsample → add → (F)MBConv 반복 → 1×1 classifier`.
   - 데이터셋별 `final_expand` 로 분류기 용량을 튜닝.
   - stride 8 출력이라 최종 리사이즈 비용도 작음.

---

## 부록 A. 자주 막히는 개념 Q&A

**Q1. 왜 `LiteMLA` 안에서 `fp32` 로 강제 변환하나요?**
선형 어텐션은 `(V·Kᵀ)` 같은 행렬곱 중간에 **아주 작은 값**이 많이
생깁니다. fp16/bf16 에서는 이런 값이 쉽게 0으로 떨어져서 정규화 분모가
0 이 되거나 gradient 가 NaN 이 될 수 있습니다. `fp32` 로 구간 승격하면
AMP 의 속도 이득은 조금 손해보지만 **학습 안정성**을 얻습니다.

**Q2. `reset_bn` 이 뭐고 왜 필요한가요?**
학습 중 BN 은 running mean/var 을 지수이동평균으로 업데이트합니다. 그런데
학습 후반의 EMA 와 실제 데이터 분포가 어긋나 있으면 평가 성능이 떨어질 수
있습니다. `reset_bn` 은 학습이 끝난 뒤 **고정된 파라미터로 데이터를 다시
통과시키며 BN stats 를 새로 계산**하는 과정입니다. 보통 수천~수만 이미지면
충분.

**Q3. `no_wd_keys: [norm, bias]` 의 의도는?**
weight decay 는 L2 패널티입니다. BN 의 γ/β(스케일/시프트) 와 bias 는
**크기 자체가 모델의 표현력과 직접 연관** 되어 있어서, 여기 WD 를 걸면
학습이 불안정하거나 성능이 떨어지는 경향이 강합니다. 그래서 이름에
`norm` 또는 `bias` 가 들어간 파라미터는 제외.

**Q4. MESA 는 LoRA/KD 와 뭐가 다른가요?**
- **KD (Knowledge Distillation)**: 보통 "별개의 큰 선생 모델" 이 필요.
- **Self-Distillation**: 자기 자신(또는 자기 자신의 사본) 을 선생으로 사용.
- **MESA**: EMA 사본을 선생으로, 학습 **후반부에만** 켜서 과적합을 막는
  자기지식증류. 학습 초반엔 모델 자체가 아직 약해서 소프트 타겟이 의미
  없으므로 `thresh` 이후에만 활성.

**Q5. 왜 분할 헤드의 merge 가 `concat` 이 아닌 `add` 인가요?**
`concat` 은 채널이 늘어나 뒤이은 Conv 가 커집니다. `add` 는 **파라미터
0**, **메모리도 절약**. 단, 각 입력을 미리 **같은 채널 수(`head_width`)**
로 맞춰야 하는 제약이 생기는데, 앞단의 1×1 Conv 가 그걸 해줍니다. 경량화
목적에 잘 맞는 선택.

**Q6. 왜 `stage3/4` 에서만 어텐션을 쓰나요?**
어텐션 비용은 `N = H·W` 에 달려 있습니다. 얕은 스테이지는 해상도가 커서
(`stage1` 이면 `H/4·W/4`) 토큰 수가 많습니다. 반면 `stage3` 은 `H/16·W/16`,
`stage4` 는 `H/32·W/32` 로 토큰이 크게 줄어드는데, 여기서 어텐션이 특히
**의미 있는 전역 관계**를 잡을 수 있으므로 비용 대비 효과가 제일 큽니다.

---

## 참고

- 논문: Cai et al., *"EfficientViT: Lightweight Multi-Scale Attention for
  High-Resolution Dense Prediction"*, ICCV 2023, pp. 17302–17313.
- 원본 정리: [EfficientViT_ICCV2023_구현분석.md](EfficientViT_ICCV2023_구현분석.md)
- 저장소 README: [README.md](README.md)
- Classification README: [applications/efficientvit_cls/README.md](applications/efficientvit_cls/README.md)
- Segmentation README: [applications/efficientvit_seg/README.md](applications/efficientvit_seg/README.md)
