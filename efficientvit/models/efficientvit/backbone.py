"""
EfficientViT 백본 정의 모듈.

ICCV 2023 "EfficientViT: Lightweight Multi-Scale Attention for
High-Resolution Dense Prediction" 논문의 백본 구현.

이 파일은 두 계열의 백본을 모두 담고 있다.
  1) EfficientViTBackbone      (B 시리즈: B0/B1/B2/B3)
  2) EfficientViTLargeBackbone (L 시리즈: L0/L1/L2/L3)

두 백본 모두 분류(cls.py) 와 분할(seg.py) 헤드에서 공용으로 사용된다.

forward() 가 단일 텐서가 아닌 dict 를 반환하는 이유는
분할(SegHead) 처럼 멀티 스테이지 피처(stage2/3/4) 를 동시에 활용하는
dense prediction 헤드에서 여러 스케일의 feature 를 한 번에 얻기 위함이다.
반환 dict 키는 {"input", "stage0", "stage1", ..., "stage_final"}.
"""

from typing import Optional

import torch
import torch.nn as nn

from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]


class EfficientViTBackbone(nn.Module):
    """EfficientViT B 시리즈 백본 (B0/B1/B2/B3).

    전체 구조:
      input_stem (stride 2 Conv + DSConv residual 블록 × depth_list[0])
        → stage1: MBConv × depth_list[1]   (첫 블록만 stride 2)
        → stage2: MBConv × depth_list[2]   (첫 블록만 stride 2)
        → stage3: MBConv(stride 2, fewer_norm=True) + EfficientViTBlock × depth_list[3]
        → stage4: MBConv(stride 2, fewer_norm=True) + EfficientViTBlock × depth_list[4]

    stage3/4 에서 다운샘플 MBConv 에 fewer_norm=True 를 쓰는 이유:
      바로 뒤에 ViT 블록(LiteMLA) 이 오는데, 직전에 BN 분포가 불안정하면
      어텐션 학습이 흔들릴 수 있어 pointwise conv 쪽 BN 을 줄여 안정화한다.

    Args:
        width_list: 5개 스테이지(stage0~4) 의 채널 수.
        depth_list: 5개 스테이지의 블록 반복 수.
        in_channels: 입력 채널 (RGB=3 기본).
        dim: LiteMLA 의 attention head 차원 (B0/B1=16, B2/B3=32).
        expand_ratio: MBConv inverted bottleneck 확장률 (기본 4).
        norm: 정규화 종류 (기본 BatchNorm2d).
        act_func: 활성 함수 (B 시리즈는 hswish 가 기본).
    """

    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        # 각 stage 출력 채널 수를 기록해두어 헤드가 참조할 수 있도록 한다.
        self.width_list = []
        # ===== input stem =====
        # 첫 Conv 는 stride 2 로 해상도를 절반(1/2) 으로 축소.
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        # 이어서 DSConv(= expand_ratio==1 경로) residual 블록을 depth_list[0] 번 반복.
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # ===== stages =====
        self.stages = []
        # stage1, stage2: 전형적인 MBConv 스테이지. 첫 블록만 stride 2 다운샘플.
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                # stride 1 이면 skip connection (IdentityLayer) 을 붙인다.
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        # stage3, stage4: "MBConv 다운샘플 + EfficientViTBlock × d" 구성.
        # EfficientViTBlock 내부에 LiteMLA(multi-scale linear attention) 이 들어간다.
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            # 다운샘플용 MBConv. 뒤에 ViT 블록이 올 것이므로 fewer_norm=True 로 BN 개수를 줄인다.
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """로컬(= 어텐션이 아닌) 블록을 확장률에 따라 선택해서 반환.

        규칙:
          - expand_ratio == 1  → DSConv (depthwise-separable conv)
          - expand_ratio != 1  → MBConv (inverted residual with expansion)

        fewer_norm=True 이면 pointwise conv 경로의 BN 을 제거해
        ViT 블록 직전에 BN 분포 교란을 줄인다.
        """
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """각 스테이지 출력을 dict 로 묶어서 반환.

        반환 dict 키:
          - "input"      : 원본 입력
          - "stage0"     : input_stem 의 출력 (해상도 1/2)
          - "stage1"     : stage1 출력 (1/4)
          - "stage2"     : stage2 출력 (1/8)   ← SegHead 에서 씀
          - "stage3"     : stage3 출력 (1/16)  ← SegHead 에서 씀
          - "stage4"     : stage4 출력 (1/32)  ← SegHead/ClsHead 에서 씀
          - "stage_final": 마지막 스테이지 출력 (= stage4)
        """
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    """B0: 가장 작은 변형. 채널 [8,16,32,64,128], depth [1,2,2,2,2], head_dim=16."""
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    """B1: 채널 [16,32,64,128,256], depth [1,2,3,3,4], head_dim=16."""
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    """B2: 채널 [24,48,96,192,384], depth [1,3,4,4,6], head_dim=32."""
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    """B3: B 시리즈 중 가장 큼. 채널 [32,64,128,256,512], depth [1,4,6,6,9], head_dim=32."""
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    """EfficientViT L 시리즈 백본 (L0/L1/L2/L3).

    B 시리즈와 달리 각 스테이지의 블록 타입(res/fmb/mb/att) 을
    block_list 로 직접 고를 수 있게 한 일반화된 설계.

    block_list 토큰 의미:
      - "res" : ResBlock         (ResNet-style basic block)
      - "fmb" : FusedMBConv      (Fused inverted residual, 3x3 conv + 1x1)
      - "mb"  : MBConv           (inverted residual with depthwise)
      - "att" : EfficientViTBlock (LiteMLA + MBConv 기반 ViT 블록)
                "att@3" / "att@5" 처럼 스케일을 지정하면 LiteMLA 의
                multi-scale aggregation 커널 크기가 바뀐다.

    기본값은 block_list=["res","fmb","fmb","mb","att"] 로 stage0~4 순서:
      - stage0(res): 입력 stem 용 가벼운 ResBlock
      - stage1/2(fmb): fused MBConv 로 초반 연산 효율 확보
      - stage3(mb): 일반 MBConv (어텐션 직전 단계)
      - stage4(att): EfficientViTBlock 으로 글로벌 정보 결합

    Args:
        width_list: 스테이지 채널 수.
        depth_list: 스테이지별 블록 반복 수.
        block_list: 스테이지별 블록 타입.
        expand_list: 스테이지별 확장률. stage1~4 의 다운샘플 블록에선
                     expand_list[stage_id] * 4 가 쓰인다.
        fewer_norm_list: 스테이지별 fewer_norm 적용 여부.
        in_channels: 입력 채널 (기본 3).
        qkv_dim: LiteMLA 의 head 차원 (기본 32).
        norm: 정규화 종류.
        act_func: 활성 함수 (L 시리즈 기본은 gelu).
    """

    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: Optional[list[str]] = None,
        expand_list: Optional[list[float]] = None,
        fewer_norm_list: Optional[list[bool]] = None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()
        # 스테이지별 블록 설정의 기본값.
        block_list = ["res", "fmb", "fmb", "mb", "att"] if block_list is None else block_list
        expand_list = [1, 4, 4, 4, 6] if expand_list is None else expand_list
        # stage3/4 (mb, att) 는 fewer_norm=True 로 ViT 직전 BN 분포 안정화.
        fewer_norm_list = [False, False, False, True, True] if fewer_norm_list is None else fewer_norm_list

        self.width_list = []
        self.stages = []
        # ===== stage 0 =====
        # 입력 stem: stride 2 Conv 로 1/2 다운샘플 후 block_list[0] (보통 "res") 반복.
        stage0 = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        # ===== stage 1 ~ 4 =====
        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            # 다운샘플 블록: "att" 스테이지에서도 실제 다운샘플은 MBConv 로 수행해야 하므로
            # block_list 가 mb/fmb 가 아니면(= "att" 등) 강제로 "mb" 를 쓴다.
            # 확장률은 expand_list[stage_id] * 4 로, 이어지는 블록보다 더 크게 잡아
            # 다운샘플 지점에서 표현력을 충분히 확보한다.
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    # ViT 스테이지: EfficientViTBlock.
                    # "att@3"/"att@5" 구분으로 LiteMLA 의 multi-scale 커널을 선택.
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    # ConvNet 스테이지: res/fmb/mb 중 해당 타입의 residual 블록.
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """block 토큰에 따라 ResBlock / FusedMBConv / MBConv 를 생성해 반환.

        fewer_norm=True 이면 pointwise 경로의 BN 을 제거하고 bias 를 살려
        다음 단계(특히 어텐션) 입력 분포가 덜 왜곡되게 한다.
        """
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """B 시리즈와 달리 stage0 가 이미 입력 stem 역할이므로
        enumerate 에 start=0 을 그대로 사용한다. 반환 dict 키는
        {"input", "stage0", "stage1", ..., "stage_final"}.
        """
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    """L0: 채널 [32,64,128,256,512], depth [1,1,1,4,4]."""
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    """L1: L0 와 채널은 같고 마지막 두 스테이지 depth 만 4→6 으로 증가."""
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    """L2: 채널은 같고 depth 를 [1,2,2,8,8] 로 더 늘린 변형."""
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    """L3: L 시리즈 중 최대. 채널 폭이 전부 2배 [64,128,256,512,1024]."""
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone
