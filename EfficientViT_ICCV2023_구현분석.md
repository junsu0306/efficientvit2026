# EfficientViT (ICCV 2023) — Classification & Segmentation 구현 분석

본 문서는 저장소 [efficientvit2026](./) 에서 **ICCV 2023 EfficientViT 논문**의
**(1) 이미지 분류(Classification)** 와 **(2) 시맨틱 분할(Segmentation)** 파트가
어떻게 구현되어 있는지, 코드 파일 레벨로 상세히 정리한 자료입니다. 저장소에는
SAM, DC-AE, Diffusion, GazeSAM 등 다른 모델도 포함되어 있지만, 여기서는
**ICCV 2023 분류/분할 파트만** 다룹니다.

논문: *"EfficientViT: Lightweight Multi-Scale Attention for
High-Resolution Dense Prediction"*, Cai et al., ICCV 2023.

---

## 1. 디렉터리 구조 한눈에 보기

ICCV 2023 분류/분할과 직접 관련된 파일은 다음과 같습니다.

```
efficientvit/
├── cls_model_zoo.py              # 분류 모델 팩토리 + 사전학습 체크포인트 레지스트리
├── seg_model_zoo.py              # 분할 모델 팩토리 + 체크포인트 레지스트리
├── models/
│   ├── efficientvit/
│   │   ├── backbone.py           # EfficientViTBackbone (B시리즈), EfficientViTLargeBackbone (L시리즈)
│   │   ├── cls.py                # ClsHead + EfficientViTCls
│   │   └── seg.py                # SegHead + EfficientViTSeg
│   ├── nn/
│   │   ├── ops.py                # 핵심 빌딩블록: ConvLayer/MBConv/FusedMBConv/LiteMLA/EfficientViTBlock …
│   │   ├── act.py                # 활성함수 레지스트리 (relu/relu6/hswish/silu/gelu)
│   │   ├── norm.py               # 정규화 레지스트리 (bn2d/ln2d/ln/trms2d) + set_norm_eps
│   │   └── drop.py               # DropPath 등
│   └── utils/                    # network/list/random 유틸
└── clscore/                      # 분류 학습 파이프라인
    ├── data_provider/imagenet.py # ImageNet 데이터 프로바이더
    └── trainer/
        ├── cls_trainer.py        # 학습 루프 (mixup/cutmix/label smooth, MESA …)
        └── cls_run_config.py     # 러너 하이퍼파라미터 래퍼

applications/
├── efficientvit_cls/
│   ├── train_efficientvit_cls_model.py
│   ├── eval_efficientvit_cls_model.py
│   ├── configs/imagenet/*.yaml   # 모델별 학습 설정 (b1~b3, l1~l3, default)
│   └── README.md
└── efficientvit_seg/
    ├── eval_efficientvit_seg_model.py
    ├── demo_efficientvit_seg_model.py
    └── README.md                 # ※ 공식적으로 분할 학습 코드는 제공되지 않음
```

> **요지**: 분류는 *모델 정의 + 학습·평가 파이프라인*이 모두 포함되어 있지만,
> 분할은 *모델 정의 + 평가/데모*만 포함되어 있고 **학습 스크립트는 공개되지
> 않습니다** (`applications/efficientvit_seg/`에 `train_*.py`가 없음).

---

## 2. 논문의 핵심 아이디어가 코드에서 어떻게 드러나는지

EfficientViT 의 핵심은 **ReLU 기반 선형 어텐션**으로 O(N²) → O(N) 복잡도를
얻으면서, **멀티스케일 집계(multi-scale aggregation)** 로 하드웨어 친화적인
고해상도 토큰 상호작용을 구현하는 것입니다. 이것이 [LiteMLA](efficientvit/models/nn/ops.py#L518-L668) 에
그대로 담겨 있습니다.

### 2.1 LiteMLA — Lightweight Multi-Scale Linear Attention

[efficientvit/models/nn/ops.py:518-668](efficientvit/models/nn/ops.py#L518-L668)

```python
class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels, out_channels,
        heads=None, heads_ratio=1.0, dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales=(5,),          # 멀티스케일 집계 커널 크기
        eps=1.0e-15,
    ):
```

구성 요소:

1. **QKV 생성** ([ops.py:546-553](efficientvit/models/nn/ops.py#L546-L553))
   - `1x1 Conv` 로 `in_channels → 3 * heads * dim` 으로 프로젝션.
2. **멀티스케일 집계** ([ops.py:554-569](efficientvit/models/nn/ops.py#L554-L569))
   - `scales` 각 원소 `s` 에 대해 `(s×s depthwise conv) → (1×1 grouped conv)`
     로 Q/K/V 를 여러 수용 범위로 확장. 기본 `scales=(5,)`.
3. **ReLU 커널 선형 어텐션** ([ops.py:582-618](efficientvit/models/nn/ops.py#L582-L618))
   - 일반 softmax 대신 `ReLU` 를 QK 커널로 사용.
   - 분배법칙으로 `Softmax(QKᵀ)V` 를 `(V·Kᵀ)·Q` 로 재배열 → O(N·D²) 로 감소.
   - 정규화는 V 끝에 1-텐서를 패딩해서 `합` 을 같이 계산하는 트릭으로 수행:
     ```python
     v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
     vk = torch.matmul(v, trans_k)
     out = torch.matmul(vk, q)
     out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
     ```
4. **동적 전환** ([ops.py:662-665](efficientvit/models/nn/ops.py#L662-L665))
   - `H*W > dim` 이면 선형(`relu_linear_att`), 아니면 저해상도 케이스에서
     수치 안정을 위해 이차식(`relu_quadratic_att`) 으로 전환.
5. **fp16/bf16 안전장치**: `@torch.autocast(..., enabled=False)` 와
   `qkv.float()` 변환으로 AMP 하에서도 수치적으로 안전.

### 2.2 EfficientViTBlock — 컨텍스트 + 로컬의 이중 모듈

[efficientvit/models/nn/ops.py:671-729](efficientvit/models/nn/ops.py#L671-L729)

```
x ─► Residual( LiteMLA )  ─►  Residual( MBConv 또는 GLUMBConv )  ─► out
       (context module)         (local module)
```

- `context_module` : `LiteMLA` 를 `ResidualBlock(IdentityLayer())` 로 감쌈 → 전역 정보.
- `local_module` : `MBConv` (B시리즈) 또는 `GLUMBConv` (GLU 변형) → 국소 정보.
- 두 블록을 직렬 연결하여 장단거리 의존성을 분리해서 처리.

---

## 3. 빌딩블록 요약 (efficientvit/models/nn/ops.py)

| 클래스 | 위치 | 역할 |
|---|---|---|
| `ConvLayer` | [ops.py:37-78](efficientvit/models/nn/ops.py#L37-L78) | `Conv2d + Norm + Dropout + Act` 래퍼, same-padding 자동 |
| `DSConv` | [ops.py:270-309](efficientvit/models/nn/ops.py#L270-L309) | Depthwise-separable conv |
| `MBConv` | [ops.py:312-364](efficientvit/models/nn/ops.py#L312-L364) | MobileNetV2 식 inverted bottleneck (expand-depthwise-project) |
| `FusedMBConv` | [ops.py:367-410](efficientvit/models/nn/ops.py#L367-L410) | 확장+공간컨볼루션을 하나로 fused, L시리즈 저지연용 |
| `GLUMBConv` | [ops.py:413-470](efficientvit/models/nn/ops.py#L413-L470) | Gated Linear Unit 적용 MBConv |
| `LiteMLA` | [ops.py:518-668](efficientvit/models/nn/ops.py#L518-L668) | 위에서 설명한 경량 선형 어텐션 |
| `EfficientViTBlock` | [ops.py:671-729](efficientvit/models/nn/ops.py#L671-L729) | LiteMLA + (GLU)MBConv 조합 |
| `ResidualBlock` | [ops.py:737-767](efficientvit/models/nn/ops.py#L737-L767) | `main(x) + shortcut(x)` |
| `DAGBlock` | [ops.py:770-804](efficientvit/models/nn/ops.py#L770-L804) | 여러 입력 → merge(add/cat) → middle → 여러 출력; SegHead가 이것을 상속 |
| `OpSequential` | [ops.py:807-819](efficientvit/models/nn/ops.py#L807-L819) | `None` 을 스킵하는 `nn.Sequential` (옵션 모듈 처리 용이) |

### 활성함수와 정규화

- **`act.py`** : `relu`, `relu6`, `hswish`, `silu`, `gelu(tanh)` 등록.
  B시리즈는 `hswish`, L시리즈와 일부 Seg 헤드는 `gelu` 사용.
- **`norm.py`** : `bn2d`, `ln2d`, `ln`, `trms2d(Triton RMSNorm)` 등록.
  모델마다 `set_norm_eps(model, eps)` 로 BN ε 값을 바꿔 끼움
  (B시리즈: `1e-5`, L시리즈: `1e-7`).

---

## 4. 백본 아키텍처

### 4.1 EfficientViTBackbone (B시리즈)

[efficientvit/models/efficientvit/backbone.py:33-157](efficientvit/models/efficientvit/backbone.py#L33-L157)

구조:

```
input_stem:
  ConvLayer(3→w0, stride=2)
  + depth_list[0]번 × ResidualBlock(DSConv(w0→w0, expand=1))

stage1 (w1): depth_list[1]개 MBConv (첫번째만 stride=2)
stage2 (w2): depth_list[2]개 MBConv
stage3 (w3): MBConv(stride=2, fewer_norm=True)
           + depth_list[3]개 × EfficientViTBlock
stage4 (w4): MBConv(stride=2, fewer_norm=True)
           + depth_list[4]개 × EfficientViTBlock
```

- `build_local_block` ([backbone.py:119-148](efficientvit/models/efficientvit/backbone.py#L119-L148))
  가 `expand_ratio==1` 일 땐 `DSConv`, 아니면 `MBConv` 를 돌려주는 점이 포인트.
- `fewer_norm=True` 는 stage3/4 의 다운샘플링 블록에서 일부 BN을 제거하는
  최적화 (ViT 블록에 들어가기 직전이라 BN 분포가 불안정해지는 걸 방지).
- `forward` 는 단일 텐서가 아닌 **`dict`** 를 반환해서 Seg 헤드가 다중 스테이지를
  받을 수 있게 함 ([backbone.py:150-156](efficientvit/models/efficientvit/backbone.py#L150-L156)):
  ```
  {"input", "stage0", "stage1", "stage2", "stage3", "stage4", "stage_final"}
  ```

**B 시리즈 설정** ([backbone.py:159-196](efficientvit/models/efficientvit/backbone.py#L159-L196)):

| 모델 | width_list | depth_list | dim |
|---|---|---|---|
| B0 | [8, 16, 32, 64, 128] | [1, 2, 2, 2, 2] | 16 |
| B1 | [16, 32, 64, 128, 256] | [1, 2, 3, 3, 4] | 16 |
| B2 | [24, 48, 96, 192, 384] | [1, 3, 4, 4, 6] | 32 |
| B3 | [32, 64, 128, 256, 512] | [1, 4, 6, 6, 9] | 32 |

### 4.2 EfficientViTLargeBackbone (L시리즈)

[efficientvit/models/efficientvit/backbone.py:199-338](efficientvit/models/efficientvit/backbone.py#L199-L338)

B시리즈보다 유연성을 높이기 위해 **스테이지별로 블록 타입을 선택**할 수
있게 되어 있습니다. `block_list=["res","fmb","fmb","mb","att"]` 기본값이며
각 토큰의 의미는:

- `"res"` → `ResBlock`
- `"fmb"` → `FusedMBConv`
- `"mb"` → `MBConv`
- `"att"` → `EfficientViTBlock` (LiteMLA 포함)

`expand_list`, `fewer_norm_list`, `act_func="gelu"` 로 채널 확장률·BN 위치·
활성함수도 스테이지별로 다르게 세팅합니다.

**L 시리즈 설정** ([backbone.py:341-374](efficientvit/models/efficientvit/backbone.py#L341-L374)):

| 모델 | width_list | depth_list |
|---|---|---|
| L0 | [32, 64, 128, 256, 512] | [1, 1, 1, 4, 4] |
| L1 | [32, 64, 128, 256, 512] | [1, 1, 1, 6, 6] |
| L2 | [32, 64, 128, 256, 512] | [1, 2, 2, 8, 8] |
| L3 | [64, 128, 256, 512, 1024] | [1, 2, 2, 8, 8] |

---

## 5. 분류 (Classification)

### 5.1 모델 정의

[efficientvit/models/efficientvit/cls.py](efficientvit/models/efficientvit/cls.py)

```python
class ClsHead(OpSequential):
    # 1) ConvLayer(in_ch → w0, 1x1)          # 채널 확장
    # 2) AdaptiveAvgPool2d(1)                # 공간 풀링
    # 3) LinearLayer(w0 → w1, norm="ln")     # MLP-1
    # 4) LinearLayer(w1 → n_classes, dropout) # 분류기
```

`forward` 는 백본 `feed_dict["stage_final"]` 을 입력으로 받음
([cls.py:43-45](efficientvit/models/efficientvit/cls.py#L43-L45)).

**변형별 Head 구성:**

| 모델 | in_channels | head width_list | act |
|---|---|---|---|
| B0 | 128 | [1024, 1280] | hswish |
| B1 | 256 | [1536, 1600] | hswish |
| B2 | 384 | [2304, 2560] | hswish |
| B3 | 512 | [2304, 2560] | hswish |
| L1/L2 | 512 | [3072, 3200] | gelu |
| L3 | 1024 | [6144, 6400] | gelu |

### 5.2 모델 Zoo — 사전학습 체크포인트 로딩

[efficientvit/cls_model_zoo.py](efficientvit/cls_model_zoo.py)

```python
REGISTERED_EFFICIENTVIT_CLS_MODEL: dict[str, (Callable, float, Optional[str])] = {
    "efficientvit-b0":     (efficientvit_cls_b0, 1e-5, None),
    "efficientvit-b0-r224":(efficientvit_cls_b0, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b0_r224.pt"),
    ...
    "efficientvit-l3-r384":(efficientvit_cls_l3, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l3_r384.pt"),
}
```

사용법:

```python
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
model = create_efficientvit_cls_model("efficientvit-l2-r384", pretrained=True)
```

- 이름 끝에 `-r{224,256,288,320,384}` 가 붙은 것은 **해당 해상도로 학습된**
  체크포인트를 의미 (멀티해상도 학습 결과).
- 가중치 경로: `assets/checkpoints/efficientvit_cls/efficientvit_{variant}_{res}.pt`
- 공식 체크포인트는 HuggingFace `han-cai/efficientvit-cls` 에서 내려받아
  해당 폴더에 두어야 함 (README 참조).

### 5.3 학습 파이프라인

#### 5.3.1 엔트리 포인트

[applications/efficientvit_cls/train_efficientvit_cls_model.py](applications/efficientvit_cls/train_efficientvit_cls_model.py)

핵심 흐름:

1. CLI 파서 — `config` YAML 필수, `--amp {fp32,fp16,bf16}`, `--rand_init`,
   `--auto_restart_thresh` 등.
2. `setup.setup_dist_env(args.gpu)` → 분산/단일 GPU 환경 초기화.
3. `setup.setup_exp_config(args.config, recursive=True, opt_args=opt)` 로
   YAML 설정을 읽고, `--data_provider.xxx` 같은 CLI 오버라이드를 병합.
4. `ImageNetDataProvider` 인스턴스 생성.
5. `create_efficientvit_cls_model(name, pretrained=False, dropout=...)` 로
   모델 빌드 후, `apply_drop_func(model.backbone.stages, config["backbone_drop"])`
   으로 DropPath 를 주입.
6. `ClsTrainer(path, model, data_provider, auto_restart_thresh)` 생성 →
   `prep_for_training(run_config, ema_decay, amp_dtype)` → `trainer.train()`.

학습 명령 예시 (README):

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
    applications/efficientvit_cls/train_efficientvit_cls_model.py \
    applications/efficientvit_cls/configs/imagenet/efficientvit_l2.yaml \
    --amp bf16 \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_l2_r224/
```

#### 5.3.2 데이터 프로바이더

[efficientvit/clscore/data_provider/imagenet.py](efficientvit/clscore/data_provider/imagenet.py)

`ImageNetDataProvider` 주요 필드:

- `data_dir = "/dataset/imagenet"` (기본)
- `n_classes = 1000`
- `image_size`: 정수 또는 리스트 — **멀티해상도 학습** 지원 (`[128,160,192,224,256,288]`)
- `base_batch_size = 128` (GPU 당)
- `data_aug` 리스트: `randaug(n=1~2, m=3~5)`, `erase(p=0.2)` 등
- 검증 보간: `"bicubic"`, `test_crop_ratio=1.0` (default.yaml 기준)

#### 5.3.3 Trainer — mixup/cutmix/label smooth/MESA

[efficientvit/clscore/trainer/cls_trainer.py](efficientvit/clscore/trainer/cls_trainer.py)

- **`_validate`** ([cls_trainer.py:35-75](efficientvit/clscore/trainer/cls_trainer.py#L35-L75)):
  `CrossEntropyLoss` + top-1 / top-5 accuracy.
- **`before_step`** ([cls_trainer.py:77-105](efficientvit/clscore/trainer/cls_trainer.py#L77-L105)):
  라벨 스무딩 → `mixup_config["op"]` 에서 가중치 기반 랜덤 선택 → Beta(α,α) 샘플링으로
  mixup 또는 cutmix 적용.
- **`run_step`** ([cls_trainer.py:107-139](efficientvit/clscore/trainer/cls_trainer.py#L107-L139)):
  AMP 컨텍스트에서 forward→loss. **MESA** (`self.run_config.mesa`) 가 설정되어
  있으면 학습 후반부(`run_config.progress >= thresh`)에 EMA 모델의 시그모이드
  출력을 소프트 타겟으로 하는 **self-distillation** 손실을 추가 (`loss += ratio * mesa_loss`).
- **`train`** ([cls_trainer.py:188-230](efficientvit/clscore/trainer/cls_trainer.py#L188-L230)):
  `run_config.bce` 가 True면 `BCEWithLogitsLoss`, 아니면 `CrossEntropyLoss`.
  `auto_restart_thresh` 초과로 정확도가 급락하면 best 체크포인트에서 재학습.

#### 5.3.4 RunConfig

[efficientvit/clscore/trainer/cls_run_config.py](efficientvit/clscore/trainer/cls_run_config.py)

필드: `base_lr`, `label_smooth`, `mixup_config`, `bce`, `mesa` 등.

#### 5.3.5 YAML 설정 샘플

**default.yaml** ([configs/imagenet/default.yaml](applications/efficientvit_cls/configs/imagenet/default.yaml)):

```yaml
data_provider:
  dataset: imagenet
  data_dir: /dataset/imagenet
  image_size: [128, 160, 192, 224]   # 멀티해상도
  base_batch_size: 128
  data_aug: {name: randaug, n: 1, m: 3}

run_config:
  n_epochs: 300
  warmup_epochs: 20
  base_lr: 0.00025
  lr_schedule_name: cosine
  optimizer_name: adamw
  optimizer_params: {eps: 1.0e-08, betas: [0.9, 0.999]}
  weight_decay: 0.1
  no_wd_keys: [norm, bias]        # BN·bias 는 weight decay 제외
  grad_clip: 2.0
  reset_bn: true                   # 학습 후 BN 통계 재계산
  reset_bn_size: 16000
  eval_image_size: [224]
  label_smooth: 0.1
  mixup_config:
    op:
      - [mixup,  0.1, 1.0]
      - [cutmix, 0.1, 1.0]
  bce: true                        # BCEWithLogitsLoss 사용

backbone_drop:
  name: droppath
  drop_prob: 0.05
  linear_decay: true               # stage 별 선형 증가

ema_decay: 0.9998
```

**efficientvit_l2.yaml** 오버라이드:

```yaml
run_config:
  base_lr: 0.00015
  mixup_config:
    op:
      - [mixup,  0.4, 1.0]
      - [cutmix, 0.4, 1.0]
  mesa:
    thresh: 0.25      # 학습 진행률 25% 이후 MESA 활성화
    ratio: 2.75
backbone_drop:
  drop_prob: 0.1
  skip: 3             # 앞 3블록은 drop 안 함
net_config:
  name: efficientvit-l2
  dropout: 0
```

YAML 은 `default.yaml` → `efficientvit_{variant}.yaml` 순서로 재귀적으로
머지됩니다 (`setup_exp_config(..., recursive=True)`).

### 5.4 평가

`applications/efficientvit_cls/eval_efficientvit_cls_model.py` 는
`create_efficientvit_cls_model(name, pretrained=True/weight_url=...)` 로 모델을
불러 ImageNet val 셋에서 top-1/top-5 를 측정합니다.

---

## 6. 분할 (Segmentation)

### 6.1 SegHead — FPN 스타일 다중 스케일 퓨전

[efficientvit/models/efficientvit/seg.py:30-104](efficientvit/models/efficientvit/seg.py#L30-L104)

`SegHead` 는 `DAGBlock` 을 상속한 DAG 구조:

```
[stage4 @stride32]  ─1x1─► upsample×4  ┐
[stage3 @stride16]  ─1x1─► upsample×2  ├─(add)─► middle (MBConv/FMBConv × depth) ─► (final_expand ConvLayer) ─► 1x1 Conv(n_classes) ─► segout
[stage2 @stride8 ]  ─1x1─► identity    ┘
```

- **inputs** ([seg.py:47-58](efficientvit/models/efficientvit/seg.py#L47-L58)):
  각 백본 스테이지를 1×1 Conv 로 `head_width` 로 프로젝션하고, `stride/head_stride`
  가 1보다 크면 `UpSampleLayer(factor)` 로 공간 해상도를 맞춤. 모든 피처는
  **덧셈(merge="add")** 으로 합침 — 즉 **경량 FPN** 역할.
- **middle** ([seg.py:60-81](efficientvit/models/efficientvit/seg.py#L60-L81)):
  `middle_op` 가 `"mbconv"` 면 MBConv, `"fmbconv"` 면 FusedMBConv 를
  `head_depth` 번 쌓되, 각 블록을 `ResidualBlock(..., IdentityLayer())` 로 감쌈.
- **outputs** ([seg.py:83-102](efficientvit/models/efficientvit/seg.py#L83-L102)):
  `final_expand` 가 주어지면 채널 확장 `ConvLayer` 를 거친 뒤 `1×1 Conv →
  n_classes` 를 `"segout"` 로 내보냄.

### 6.2 EfficientViTSeg

[efficientvit/models/efficientvit/seg.py:107-117](efficientvit/models/efficientvit/seg.py#L107-L117)

```python
class EfficientViTSeg(nn.Module):
    def forward(self, x):
        feed_dict = self.backbone(x)    # dict of stage features
        feed_dict = self.head(feed_dict)
        return feed_dict["segout"]      # (B, n_classes, H/8, W/8)
```

출력 해상도는 `head_stride=8` 이므로 입력 대비 1/8. 실제 평가시엔 GT 마스크
크기로 `resize(..., size=mask.shape[-2:])` 해서 비교함 (평가 스크립트 참고).

### 6.3 변형별 SegHead 설정

[efficientvit/models/efficientvit/seg.py:120-341](efficientvit/models/efficientvit/seg.py#L120-L341)

기본 공통 파라미터:
- `fid_list = ["stage4", "stage3", "stage2"]`
- `stride_list = [32, 16, 8]`, `head_stride = 8`

| 모델 | 데이터셋 | in_channel_list | head_width | head_depth | expand_ratio | middle_op | final_expand | n_classes | act |
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

- B시리즈는 `MBConv + hswish`, L시리즈는 `FusedMBConv + gelu` 로 일관됨.
- Cityscapes Head 는 `final_expand=4` (B) 또는 `None` (L) — 해상도가 커서
  채널을 미리 늘리는 대신 내부 channel 을 크게 둠.
- ADE20K Head 는 반대로 마지막에 `final_expand=8` 로 크게 부풀려 분류.

### 6.4 모델 Zoo

[efficientvit/seg_model_zoo.py](efficientvit/seg_model_zoo.py)

`functools.partial` 로 `dataset` 인자를 바인딩한 뒤 등록:

```python
"efficientvit-seg-l2-cityscapes":
    (partial(efficientvit_seg_l2, dataset="cityscapes"),
     1e-7,
     "assets/checkpoints/efficientvit_seg/efficientvit_seg_l2_cityscapes.pt"),
```

등록된 모델 (총 11개):

- Cityscapes (6): `efficientvit-seg-{b0,b1,b2,b3,l1,l2}-cityscapes`
- ADE20K (5): `efficientvit-seg-{b1,b2,b3,l1,l2}-ade20k`
  - ADE20K는 B0 미제공 (B0는 Cityscapes 전용).

### 6.5 평가 스크립트

[applications/efficientvit_seg/eval_efficientvit_seg_model.py](applications/efficientvit_seg/eval_efficientvit_seg_model.py)

주요 요소:

- **`SegIOU`** ([eval_*.py:73-106](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L73-L106)):
  `torch.histc` 로 클래스별 intersection/union 누적 → 최종 mIoU.
- **`CityscapesDataset`** ([eval_*.py:109-237](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L109-L237)):
  `leftImg8bit/*/*.png` 와 `gtFine/*/*_labelIds.png` 쌍을 수집.
  `label_map` ([eval_*.py:152-189](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L152-L189))
  으로 원본 34개 Cityscapes 라벨을 학습용 19 클래스로 매핑 (`-1` 은 void).
  기본 `crop_size = (1024, 2048)` (H, W)로 리사이즈.
- **`ADE20KDataset`** ([eval_*.py:240-602](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L240-L602)):
  150 클래스 이름/컬러 팔레트 내장. 마스크에서 `-1` 은 ignore.
  짧은 변을 `crop_size=512` 로 맞추고 긴 변은 `/32` 로 라운딩
  ([eval_*.py:578-590](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L578-L590)).
- **전처리**: ImageNet 정규화 `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`.
- **추론**: `torch.inference_mode()` + `torch.argmax(output, dim=1)`, 출력
  크기가 마스크와 다르면 `efficientvit.models.utils.resize` 로 맞춤
  ([eval_*.py:676-680](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L676-L680)).
- `--save_path` 가 주어지면 `get_canvas` ([eval_*.py:605-620](applications/efficientvit_seg/eval_efficientvit_seg_model.py#L605-L620))
  로 원본과 예측 마스크를 반투명 오버레이해서 PNG 저장.

**실행 예시** (README):

```bash
python applications/efficientvit_seg/eval_efficientvit_seg_model.py \
    --model efficientvit-seg-l2-cityscapes \
    --path ~/dataset/cityscapes/leftImg8bit/val \
    --dataset cityscapes --crop_size 1024
```

### 6.6 데모

[applications/efficientvit_seg/demo_efficientvit_seg_model.py](applications/efficientvit_seg/demo_efficientvit_seg_model.py)

한 장의 이미지를 받아 팔레트로 시각화. Cityscapes/ADE20K 둘 다 지원.

### 6.7 학습 코드는 없음

공식 저장소 방침상 **Segmentation 학습 스크립트는 공개되어 있지 않습니다.**
사용자는 제공된 **사전학습 체크포인트** 를 로드해 평가·추론만 가능합니다.
(학습을 재현하려면 mmSegmentation 등 외부 프레임워크로 구성해야 함.)

---

## 7. 분류 vs 분할 통합 비교

| 항목 | Classification | Segmentation |
|---|---|---|
| 모델 클래스 | `EfficientViTCls` | `EfficientViTSeg` |
| Head | `ClsHead` (Conv→Pool→MLP→MLP) | `SegHead` (DAG: multi-stage add → middle → 1×1) |
| 백본 출력 사용 | `feed_dict["stage_final"]` | `stage4`, `stage3`, `stage2` 3개 |
| 데이터셋 | ImageNet-1K | Cityscapes (19), ADE20K (150) |
| 학습 스크립트 | 제공됨 (`train_efficientvit_cls_model.py`) | **미제공** |
| 평가 스크립트 | 제공됨 | 제공됨 |
| Norm eps | B: 1e-5, L: 1e-7 | B: 1e-5, L: 1e-7 (동일) |
| B시리즈 act | hswish | hswish |
| L시리즈 act | gelu | gelu |

---

## 8. 빠른 시작 예제

### 분류 추론

```python
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
model = create_efficientvit_cls_model("efficientvit-l2-r384", pretrained=True)
model.eval().cuda()

# 입력: (B, 3, 384, 384), ImageNet 정규화 적용 후
logits = model(images)  # (B, 1000)
```

### 분할 추론

```python
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
model = create_efficientvit_seg_model("efficientvit-seg-l2-cityscapes", pretrained=True)
model.eval().cuda()

# 입력: (B, 3, 1024, 2048), ImageNet 정규화 적용 후
logits = model(images)              # (B, 19, 128, 256)  — stride 8
pred   = logits.argmax(dim=1)       # (B, 128, 256)
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

저장소의 `cls_model_zoo.py` / `seg_model_zoo.py` 가 기대하는 경로:

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
다운로드하여 위 경로에 그대로 두면 `pretrained=True` 로 바로 로드됩니다.

---

## 10. 핵심 설계 포인트 요약

1. **LiteMLA의 O(N) 선형 어텐션** — 고해상도(분할) 시나리오에서
   softmax 어텐션 대비 연산·메모리 이점이 결정적. 멀티스케일 depthwise
   집계와 결합해 수용 영역도 함께 확보.
2. **EfficientViTBlock = LiteMLA + (GLU)MBConv** — 전역/국소 모듈을
   잔차로 연결하는 Metaformer 스타일.
3. **백본이 dict 반환** — 분류 헤드는 `stage_final` 만, 분할 헤드는
   `stage2/3/4` 모두를 활용. 같은 백본으로 두 과제를 공유.
4. **B 시리즈(모바일)** vs **L 시리즈(고성능)** 분리
   - B: MBConv + hswish + BN(ε=1e-5)
   - L: FusedMBConv/MBConv 혼합 + gelu + BN(ε=1e-7), 블록 타입을 스테이지별
     지정 가능.
5. **멀티해상도 학습 + 해상도별 체크포인트** — 한 모델을 224–384 범위에서
   평가할 수 있게 해상도별 `-r{res}` 바리언트를 사전학습으로 제공.
6. **학습 트릭**: label smoothing, mixup/cutmix (가중치 기반 선택),
   BCE 로스, DropPath 선형 감쇠, EMA, AMP bf16, MESA self-distillation,
   BN 재계산(`reset_bn`), 정확도 급락 시 자동 재시작.
7. **분할 헤드**: 경량 FPN (1×1 project + upsample + add) → MBConv/FMBConv
   residual stack → 1×1 classifier. 데이터셋별 `final_expand` 로 용량 튜닝.

---

## 참고

- 논문: Cai et al., *"EfficientViT: Lightweight Multi-Scale Attention for
  High-Resolution Dense Prediction"*, ICCV 2023, pp. 17302–17313.
- 저장소 README: [README.md](README.md)
- Classification README: [applications/efficientvit_cls/README.md](applications/efficientvit_cls/README.md)
- Segmentation README: [applications/efficientvit_seg/README.md](applications/efficientvit_seg/README.md)
