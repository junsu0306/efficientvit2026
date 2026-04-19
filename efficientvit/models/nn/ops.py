"""
EfficientViT 의 기본 빌딩 블록 모듈.

이 모듈은 논문 "EfficientViT: Lightweight Multi-Scale Attention for
High-Resolution Dense Prediction" (ICCV 2023) 의 구현에 사용되는
핵심 신경망 연산자들을 정의한다.

주요 구성:
- ConvLayer / LinearLayer: Conv2d / Linear + Norm + Dropout + Act 를 묶은 기본 래퍼.
- UpSampleLayer / ConvPixelUnshuffleDownSampleLayer / PixelUnshuffleChannelAveragingDownSampleLayer
  / ConvPixelShuffleUpSampleLayer / InterpolateConvUpSampleLayer
  / ChannelDuplicatingPixelUnshuffleUpSampleLayer: 다양한 해상도 변환(업/다운 샘플) 레이어.
- DSConv / MBConv / FusedMBConv / GLUMBConv / ResBlock: 경량 합성곱 블록들.
- LiteMLA: ReLU 커널 기반 선형 어텐션. (V·Kᵀ)·Q 재배열로 O(N·D²) 복잡도 달성이 핵심 기여.
- EfficientViTBlock: LiteMLA(컨텍스트) + (GLU)MBConv(로컬) 조합의 2단 잔차 블록.
- ResidualBlock / DAGBlock / OpSequential: 구조 구성용 유틸 블록.

본 파일의 모든 주석은 가독성을 위해 한국어로 작성되었으나, 원본 Python 코드는
그대로 보존되었다.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm
from efficientvit.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple

__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "ConvPixelUnshuffleDownSampleLayer",
    "PixelUnshuffleChannelAveragingDownSampleLayer",
    "ConvPixelShuffleUpSampleLayer",
    "ChannelDuplicatingPixelUnshuffleUpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    """Conv2d + Norm + Activation 을 묶어 한 블록으로 제공하는 래퍼.

    - `get_same_padding(kernel_size)` 로 자동 same-padding 을 계산하고,
      dilation 을 곱해 실제 padding 크기를 보정한다.
    - dropout 이 0 보다 크면 입력단에서 `nn.Dropout2d` 가 적용되고,
      이어서 `conv -> norm -> act` 순으로 호출된다.
    - norm / act_func 이 None 이거나 빈 문자열이면 해당 단계가 비활성화된다.

    Args:
        in_channels:  입력 채널 수.
        out_channels: 출력 채널 수.
        kernel_size:  커널 크기 (기본 3).
        stride:       스트라이드 (기본 1).
        dilation:     팽창(dilation) 계수 (기본 1).
        groups:       그룹 수. depthwise 컨볼루션을 만들려면 in_channels 로 지정.
        use_bias:     Conv2d 의 bias 사용 여부.
        dropout:      dropout 확률 (0 이면 비활성).
        norm:         사용할 normalization 이름 (예: "bn2d", "ln2d", None).
        act_func:     사용할 activation 이름 (예: "relu", "hswish", None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        # same-padding 계산 후 dilation 을 고려해 패딩을 확장한다.
        padding = get_same_padding(kernel_size)
        padding *= dilation

        # dropout > 0 인 경우에만 dropout 모듈을 두고, 아니면 None 으로 두어 forward 에서 건너뛴다.
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        # norm / act 는 build_* 가 None 을 반환하면 forward 에서 스킵된다.
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 순서: dropout -> conv -> norm -> act (각 단계는 선택적)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    """보간(interpolation) 기반 업샘플링 레이어.

    `size` 가 주어지면 정확히 그 해상도로, 그렇지 않으면 `factor` 만큼 확대한다.
    AMP 환경에서 bicubic 같은 모드가 fp16/bf16 을 제대로 지원하지 않는 이슈가
    있어 forward 진입 시 autocast 를 끄고 필요 시 fp32 로 승격시킨 뒤 보간한다.
    """

    def __init__(
        self,
        mode="bicubic",
        size: Optional[int | tuple[int, int] | list[int]] = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        # size 가 int 나 tuple 로 들어오면 모두 길이 2 리스트로 통일.
        self.size = val2list(size, 2) if size is not None else None
        # size 가 지정되면 factor 는 무시한다.
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 이미 원하는 크기거나 factor 가 1 이면 그대로 반환(불필요 보간 회피).
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        # AMP 로 들어온 fp16/bf16 텐서는 수치 안정을 위해 fp32 로 승격.
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    """Conv -> PixelUnshuffle 로 구성된 다운샘플 레이어.

    먼저 Conv 로 채널을 `out_channels / factor**2` 로 축소한 뒤,
    `pixel_unshuffle` 으로 공간 해상도를 factor 배 줄이고 채널을 factor**2 배 늘린다.
    최종 텐서 채널 수는 out_channels 와 같다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        # pixel_unshuffle 이 채널을 factor**2 배 증가시키므로, 미리 그만큼 나눠놓아야 한다.
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # (B, C', H, W) -> (B, C' * factor**2, H/factor, W/factor)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    """파라미터 없이 pixel_unshuffle + 채널 평균으로 다운샘플하는 레이어.

    1) `pixel_unshuffle` 로 (B, C, H, W) -> (B, C*factor**2, H/factor, W/factor).
    2) 채널 축을 `(out_channels, group_size)` 로 뷰 해 두고 group 내부 평균을 취해
       out_channels 로 축약한다.

    학습 파라미터가 없어 매우 가벼우며 shortcut/skip 용도로 자주 쓰인다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        # pixel_unshuffle 후 채널 수 (in_channels * factor**2) 가 out_channels 의 배수여야 평균 가능.
        assert in_channels * factor**2 % out_channels == 0
        # out_channels 당 평균낼 원본 채널 개수.
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        # (B, out_channels, group_size, H, W) 로 쪼개 group 축(2) 을 평균.
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    """Conv -> PixelShuffle 로 구성된 업샘플 레이어.

    먼저 Conv 로 채널 수를 `out_channels * factor**2` 로 늘린 뒤,
    `pixel_shuffle` 으로 해상도는 factor 배, 채널은 1/factor**2 배로 재배열한다.
    서브픽셀(sub-pixel) 컨볼루션 업샘플과 동일한 패턴이다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        # pixel_shuffle 이 채널을 1/factor**2 로 축소하므로 미리 factor**2 배 부풀려 둔다.
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # (B, C*factor**2, H, W) -> (B, C, H*factor, W*factor)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    """Interpolate(보간) -> Conv 순서로 업샘플하는 레이어.

    먼저 `F.interpolate` 로 해상도를 factor 배 키우고, 이어서 Conv 로 채널 변환과
    디테일 보정을 수행한다. 기본 mode 는 `nearest` 로 체커보드 아티팩트가 거의 없다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 해상도 factor 배 확대 후 Conv 적용.
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    """파라미터 없는 업샘플 레이어. 채널을 repeat_interleave 로 복제한 뒤 pixel_shuffle.

    1) 채널 축을 `repeats` 만큼 복제해 채널을 `out_channels * factor**2` 로 맞춘다.
    2) `pixel_shuffle` 로 해상도는 factor 배, 채널은 out_channels 로 줄인다.

    학습 파라미터가 없어 가볍고, shortcut/skip 경로에 흔히 쓰인다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        # in_channels 가 out_channels*factor**2 를 정확히 나눠야(정수 repeats) 한다.
        assert out_channels * factor**2 % in_channels == 0
        # 각 입력 채널을 몇 번 복제할지.
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 채널 축으로 각 원소를 인접하게 복제: [c0,c1] -> [c0,c0,...,c1,c1,...]
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


class LinearLayer(nn.Module):
    """Linear + Norm + Activation 을 묶은 래퍼.

    - 입력이 2 차원보다 크면(예: (B, C, H, W)) 자동으로 `flatten(start_dim=1)` 을 수행해
      전결합 층에 넣을 수 있도록 squeeze 한다.
    - dropout -> linear -> norm -> act 순서로 적용된다.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) 같은 다차원 입력을 (B, C*H*W) 로 평탄화.
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    """입력을 그대로 반환하는 항등 레이어.

    `ResidualBlock` 의 shortcut 으로 자주 사용되며, shortcut=None 과는 의미가 다르다:
    - `IdentityLayer`: skip connection 이 존재 (res = main(x) + x).
    - `None`: skip connection 없음. main(x) 결과만 반환.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet v1 스타일).

    구조: depthwise(3x3, groups=in_channels) -> pointwise(1x1).
    일반 Conv 대비 연산량을 (1/out_channels + 1/k^2) 수준으로 줄이는 경량 블록.

    Args:
        in_channels / out_channels: 입력/출력 채널 수.
        kernel_size: depthwise 의 커널 크기.
        stride: depthwise 의 스트라이드.
        use_bias / norm / act_func:
            (depthwise, pointwise) 각 단계에 적용할 2-튜플.
            스칼라를 주면 `val2tuple` 이 두 단계에 동일하게 적용한다.
            기본값은 depthwise 뒤에 ReLU6, pointwise 뒤에는 활성화 없음.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        # 각 옵션을 (depthwise, pointwise) 2-튜플로 정규화.
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        # depthwise: groups=in_channels 이라 채널별 독립 spatial 컨볼루션.
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        # pointwise: 1x1 컨볼루션으로 채널 믹싱.
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    """Inverted Residual Bottleneck (MobileNet v2 의 MBConv).

    3 단 구조:
        1) inverted_conv   : 1x1 pointwise, 채널을 expand_ratio 배로 확장.
        2) depth_conv      : kxk depthwise, 공간 정보 추출.
        3) point_conv      : 1x1 pointwise, 채널을 out_channels 로 축소(선형 projection).

    채널을 한 번 늘렸다가 줄이는 "inverted" 구조로 파라미터 대비 표현력을 확보한다.
    기본 `expand_ratio=6` 은 MobileNet v2 / EfficientNet 계열에서의 관행적 기본값이다.

    `FusedMBConv` 와의 차이: FusedMBConv 는 inverted(1x1) + depthwise(kxk) 를 하나의
    (kxk, group=1) 컨볼루션으로 합친 2 단 구조로, 얕은 스테이지의 HW 친화 실효율을
    높인다. 반면 MBConv 는 3 단이라 표현력이 더 높고 파라미터 효율이 좋다.

    Args:
        mid_channels: 병목 폭. 지정하지 않으면 `round(in_channels * expand_ratio)`.
        norm/act_func/use_bias: (inverted, depth, point) 세 단계에 대한 3-튜플.
                                 스칼라 입력은 세 단계에 동일 적용.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        # (inverted, depth, point) 3-튜플로 정규화.
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        # 병목 폭 기본값: 입력 채널의 expand_ratio 배.
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        # 1) 1x1 pointwise: 채널 확장.
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        # 2) kxk depthwise: 공간 정보 추출. stride 는 여기에만 적용.
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        # 3) 1x1 pointwise: 선형 projection (기본 act_func[2]=None → 비선형 없음).
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    """Fused Inverted Residual Bottleneck (EfficientNetV2 의 Fused-MBConv).

    MBConv 의 `inverted_conv(1x1) + depth_conv(kxk depthwise)` 두 단계를
    하나의 `spatial_conv(kxk, groups=groups)` 로 합친 2 단 구조:
        1) spatial_conv  : kxk 컨볼루션으로 공간 정보 추출 + 채널 확장.
        2) point_conv    : 1x1 pointwise 로 채널 축소(선형 projection).

    얕은 스테이지(해상도가 큰 초기 단) 에서는 depthwise 가 메모리/연산 비효율의
    원인이 되므로 일반 convolution 으로 합치는 편이 GPU/Mobile 가속기에서
    훨씬 빠른 경우가 많다.

    Args:
        groups: spatial_conv 의 그룹 수. 기본 1 은 일반 컨볼루션.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        # (spatial, point) 2-튜플 정규화.
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        # 1) kxk 합성곱: 공간 정보 + 채널 확장 (MBConv 의 1x1 + depthwise 를 합침).
        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        # 2) 1x1 pointwise: 선형 projection.
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class GLUMBConv(nn.Module):
    """Gated Linear Unit 스타일의 MBConv 변형.

    중간 표현을 `데이터 채널` 과 `게이트 채널` 두 부분으로 분리해 element-wise
    곱으로 정보 흐름을 조절하는 GLU 메커니즘을 쓴다:

        y = (data) * act(gate)

    구현상 핵심은 `inverted_conv` 에서 한번에 `mid_channels * 2` 채널을 뽑고,
    이어 `depth_conv` 도 2배 채널을 그대로 유지한 다음, `torch.chunk(x, 2, dim=1)`
    로 절반은 `x`(데이터), 절반은 `gate` 로 쪼개는 것이다. gate 에만 activation
    (기본 SiLU) 을 적용한 뒤 element-wise 곱을 취해 출력한다.

    이 구조는 EfficientViT-SAM 등의 고성능 백본에서 비선형 표현력을 높이는 데
    효과적임이 보고되었다.

    Args:
        expand_ratio: mid_channels 배율. 실제 내부 채널은 `mid_channels * 2` 가 된다.
        norm/act_func/use_bias: (inverted, depth+gate_act, point) 3-튜플.
            - act_func[1] 은 depth_conv 뒤가 아니라 gate 에 적용하는 activation 이다.
            - depth_conv 자체의 act 는 None 으로 비활성화된다 (GLU 곱셈 전 활성화 금지).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        # (inverted, depth/gate, point) 3-튜플 정규화.
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        # GLU 의 gate 에 쓰일 비선형 함수(기본 SiLU). inplace=False 로 두어 chunk 후 안전하게 사용.
        self.glu_act = build_act(act_func[1], inplace=False)
        # 1) 1x1 pointwise: 데이터/게이트 두 트랙을 동시에 뽑기 위해 mid_channels*2 로 확장.
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        # 2) depthwise. 채널 수는 mid_channels*2 로 유지되며 이후 chunk 로 절반씩 나눈다.
        #    NOTE: 여기서는 activation 을 적용하지 않는다. (GLU 곱셈 전에는 raw 상태)
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        # 3) 1x1 pointwise: 데이터 트랙(mid_channels) 을 out_channels 로 projection.
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 확장 + depthwise 공간 정보 추출 (채널 2배 유지).
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        # 2) 채널 축을 반으로 쪼개 데이터(x) / 게이트(gate) 로 분리.
        x, gate = torch.chunk(x, 2, dim=1)
        # 3) 게이트에만 activation 을 걸고 element-wise 곱으로 정보 흐름 제어.
        gate = self.glu_act(gate)
        x = x * gate

        # 4) 1x1 projection 으로 out_channels 로 축소.
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    """ResNet 계열에서 쓰이는 전통적인 2 단 Conv 블록 (3x3 -> 3x3).

    - conv1: kxk 컨볼루션 (stride 적용).
    - conv2: kxk 컨볼루션 (stride=1, 선형 projection).

    기본 `expand_ratio=1` 이라 병목 확장이 없는 basic block 에 해당한다.
    `ResidualBlock` 으로 감싸면 전형적인 residual 구조가 된다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        # (conv1, conv2) 2-튜플 정규화.
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        # 1) 첫 번째 합성곱: 해상도 축소 및 채널 변환(expand).
        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        # 2) 두 번째 합성곱: 선형 projection (기본 act 없음).
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""경량 멀티스케일 선형 어텐션 (Lightweight Multi-Scale Linear Attention).

    EfficientViT 의 핵심 기여 모듈. 일반 softmax 어텐션은 Q·Kᵀ 가 (N×N) 행렬이라
    연산/메모리가 O(N²) 이지만, LiteMLA 는 softmax 를 ReLU 커널로 대체하여
    행렬곱 결합법칙으로 O(N·d²) 로 줄인다.

        Attention(Q, K, V)_i
            = Σ_j  φ(Q_i)·φ(K_j)ᵀ · V_j  /  Σ_j  φ(Q_i)·φ(K_j)ᵀ
            = φ(Q_i) · [ Σ_j V_j · φ(K_j)ᵀ ]  /  φ(Q_i) · [ Σ_j φ(K_j)ᵀ ]

    여기서 φ = ReLU. 분자/분모 모두 `(V · Kᵀ)` 혹은 `(1 · Kᵀ)` 같은 (d×d) 중간
    행렬을 먼저 계산하면 공간 길이 N 에 선형인 복잡도로 떨어진다.

    추가로 "멀티스케일" 파트는 qkv 를 여러 커널 크기의 depthwise 로 한 번씩 더
    집계해 concat 하여 수용장(receptive field) 을 확장한다(기본 `scales=(5,)`).

    Args:
        in_channels:  입력 채널 수.
        out_channels: 출력 채널 수.
        heads:        어텐션 헤드 수. None 이면 `in_channels // dim * heads_ratio` 로 자동 계산.
        heads_ratio:  heads 자동 계산 시 배수.
        dim:          헤드당 차원. 기본 8. 작게 유지해 D^2 중간 행렬이 작도록 한다.
        use_bias:     (qkv, proj) 각각에 대한 bias 사용 여부 (2-튜플).
        norm:         (qkv, proj) 각각의 normalization (2-튜플). 기본 (None, "bn2d").
        act_func:     (qkv, proj) 각각의 activation (2-튜플). 기본 (None, None).
        kernel_func:  선형 어텐션의 커널 함수. 기본 "relu" → φ(x) = ReLU(x).
        scales:       멀티스케일 집계 커널 크기 튜플. 기본 `(5,)` 한 개.
                      각 scale 마다 depthwise(kxk) + pointwise(1x1) 를 거쳐 qkv 에
                      concat 되므로, `len(scales)` 가 늘수록 수용장은 넓어지지만
                      채널 수도 `1 + len(scales)` 배로 증가한다.
        eps:          선형 어텐션 정규화 분모에 더하는 안정화 상수.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        # 헤드 수 자동 계산: heads = in_channels // dim * heads_ratio
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        # 전체 어텐션 차원 = heads × head_dim
        total_dim = heads * dim

        # (qkv, proj) 2-튜플 정규화.
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        # Q, K, V 를 한 번에 뽑는 1x1 pointwise. 출력 채널은 3*total_dim (Q+K+V).
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        # 멀티스케일 집계: scales 각각에 대해
        #   depthwise(3*total_dim, scale x scale, groups=3*total_dim)
        #   -> pointwise(1x1, groups=3*heads) 로 헤드 단위의 믹싱.
        # 이를 통해 동일한 qkv 에 scale 크기만큼의 수용장을 가진 변형본을 추가로 만든다.
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        # 선형 어텐션의 커널 함수 φ. inplace=False 이어야 q,k 각각에 독립 적용 가능.
        self.kernel_func = build_act(kernel_func, inplace=False)

        # 출력 projection: 멀티스케일 concat 결과(total_dim * (1 + len(scales))) 를
        # out_channels 로 축약. 기본 norm 은 bn2d.
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        """O(N·d²) 선형 어텐션 경로.

        (V·Kᵀ) 를 먼저 계산해 (d+1, d) 크기의 중간 행렬로 축약한 뒤 Q 와 곱한다.
        정규화 분모(Σ K_j) 는 V 에 1 로 채워진 추가 행을 붙여 함께 계산한다.
        """
        B, _, H, W = list(qkv.size())

        # AMP 에서 fp16 로 들어오면 (V·Kᵀ)·Q 축적합에서 오버/언더플로가 쉽게 발생해 fp32 로 승격.
        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        # (B, 3*total_dim, H, W) -> (B, heads, 3*dim, H*W)
        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        # 헤드당 dim 채널씩 Q, K, V 로 분리.
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # 선형 어텐션의 커널 φ(·) 적용. 기본 ReLU.
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # K 를 전치해서 (d, N) 형태로.
        trans_k = k.transpose(-1, -2)

        # --- 정규화 분모까지 한 번에 계산하는 1-패딩 트릭 ---
        # 분자: out  = Σ_j V_j · K_jᵀ · Q_i
        # 분모: norm = Σ_j  1  · K_jᵀ · Q_i
        # V 끝에 1 로 채워진 한 행을 붙이면(= 채널 축 마지막에 pad) 한 번의
        # matmul 로 [out; norm] 이 동시에 계산된다.
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)

        # 선형 어텐션 핵심: (V·Kᵀ)·Q 순서로 곱해 중간 행렬 크기를 (d+1)×d 로 유지 → O(N·d²).
        vk = torch.matmul(v, trans_k)  # (B, heads, d+1, d)
        out = torch.matmul(vk, q)      # (B, heads, d+1, N)
        if out.dtype == torch.bfloat16:
            out = out.float()
        # 마지막 행(= 분모) 으로 나머지 행(= 분자) 을 나눠 정규화. eps 는 0-division 방지.
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        # (B, heads*d, H, W) 로 복원.
        out = torch.reshape(out, (B, -1, H, W))
        return out

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        """O(N²) 2차 어텐션 경로.

        N (= H*W) 이 너무 작을 때는 (N×N) 어텐션 맵이 오히려 더 작기 때문에
        선형 경로보다 빠르고 정확하다. 동일하게 ReLU 커널을 쓰지만, softmax 대신
        column-sum 정규화(분모를 명시적으로 계산) 를 적용한다.
        """
        B, _, H, W = list(qkv.size())

        # (B, 3*total_dim, H, W) -> (B, heads, 3*dim, H*W)
        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # (N x N) 어텐션 맵: Kᵀ · Q.
        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        # 열 합계로 정규화할 때의 fp16/bf16 수치 안정 이슈를 피하기 위해 fp32 승격 후 다시 캐스트.
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        # 정규화된 맵을 V 와 곱해 최종 출력.
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 기본 qkv 생성 (1x1 pointwise).
        qkv = self.qkv(x)
        # 2) 멀티스케일 qkv 변형본을 추가해 채널 방향으로 concat.
        #    최종 채널 수는 3*total_dim * (1 + len(scales)).
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        # 3) 토큰 수 N = H*W 와 head 차원 d 를 비교해 분기.
        #    - N > d 이면 (d×d) 중간 행렬을 거치는 선형 경로가 더 싸다.
        #    - N <= d 이면 (N×N) 가 더 작아 2차 경로가 유리 (그리고 근사 오차도 작음).
        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            # 선형 경로는 내부에서 fp32 로 올렸다가 내릴 수 있어 원래 dtype 로 캐스트.
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        # 4) 멀티스케일 결과를 out_channels 로 projection.
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    """EfficientViT 백본의 기본 트랜스포머 블록.

    두 개의 잔차(residual) 서브블록이 직렬로 연결된 구조:
        1) context_module : 전역 컨텍스트를 섞는 모듈. 기본은 LiteMLA.
        2) local_module   : 국소(local) 공간 정보를 정제하는 모듈. 기본 MBConv.

    전통적인 Transformer 블록이 "Attention + FFN" 으로 전역→특징변환을 수행하는 것처럼,
    EfficientViTBlock 은 "LiteMLA(선형 어텐션) + (GLU)MBConv(경량 convolution FFN)" 을
    쌓는다. 각 서브블록은 `ResidualBlock(main, IdentityLayer())` 로 감싸 skip connection 을 갖는다.

    Args:
        in_channels:   입력/출력 채널 수 (블록 내부에서 채널 수를 유지).
        heads_ratio:   LiteMLA 헤드 수 자동 계산용 배수.
        dim:           LiteMLA 헤드당 차원. 기본 32.
        expand_ratio:  (GLU)MBConv 의 채널 확장 배수. 기본 4.
        scales:        LiteMLA 멀티스케일 커널 크기 튜플. 기본 `(5,)`.
        norm:          블록 내부 normalization 이름.
        act_func:      (GLU)MBConv 의 activation 이름. 기본 "hswish".
        context_module: "LiteMLA" 만 지원.
        local_module:  "MBConv" 또는 "GLUMBConv".
    """

    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super(EfficientViTBlock, self).__init__()
        # 1) 컨텍스트(전역) 모듈: 현재는 LiteMLA 만 지원.
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                ),
                IdentityLayer(),
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")
        # 2) 로컬(국소) 모듈: MBConv 또는 GLUMBConv. Transformer 의 FFN 역할.
        if local_module == "MBConv":
            self.local_module = ResidualBlock(
                MBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        elif local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        else:
            raise NotImplementedError(f"local_module {local_module} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 컨텍스트 -> 로컬 순서로 통과 (각각 residual 구조).
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    """범용 잔차(residual) 래퍼: `res = main(x) + shortcut(x)`.

    shortcut 의 종류별 동작:
      - `IdentityLayer()`       : 표준 residual (res = main(x) + x).
      - `ConvLayer` 등 변환 모듈 : projection shortcut (채널/해상도 변경 시 사용).
      - `None`                  : shortcut 없음. forward 는 main(x) 만 반환
                                   (post_act 도 적용되지 않음).

    `main` 이 None 이면 입력을 그대로 반환(no-op).

    옵션:
        pre_norm : main 에 들어가기 전에 적용할 normalization (예: pre-LN).
        post_act : 잔차 합 후 적용할 activation (skip 자체에는 적용되지 않음).
    """

    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        # pre_norm 이 있으면 normalize 후 main 실행 (pre-LN 패턴).
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            # main 자체가 없으면 입력을 그대로 반환.
            res = x
        elif self.shortcut is None:
            # shortcut 이 명시적으로 None 이면 잔차 합 없이 main(x) 만 반환.
            res = self.forward_main(x)
        else:
            # 표준 residual 경로.
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    """다중 입력 → 병합 → 변환 → 다중 출력의 DAG(Directed Acyclic Graph) 블록.

    forward 파이프라인:
        feat_k = input_ops[k](feature_dict[input_keys[k]])       # 각 입력을 개별 변환
        feat   = merge(feat_1, ..., feat_K)                      # "add" 또는 "cat"
        feat   = post_input(feat)   (선택)                        # 병합 후 공통 전처리
        feat   = middle(feat)                                     # 공통 변환 본체
        feature_dict[output_keys[j]] = output_ops[j](feat)       # 각 출력 head

    `feature_dict` 를 in-place 로 갱신해 반환하므로 원본에 추가 key 를 덧붙이는 용도에도
    적합하다. EfficientViT 의 SegHead 등이 이 DAGBlock 을 상속해 멀티스케일 feature 를
    받아 출력을 생성한다.

    Args:
        inputs:     입력 key → 해당 입력을 전처리할 모듈 매핑.
        merge:      입력들의 병합 방식. "add" 는 element-wise 합, "cat" 은 채널 concat.
        post_input: 병합 직후 적용할 모듈 (None 이면 생략).
        middle:     병합 결과 전체에 공통 적용할 본 변환 모듈.
        outputs:    출력 key → 출력 직전에 적용할 head 모듈 매핑.
    """

    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        # 딕셔너리를 nn.ModuleList + key list 로 분리 저장 (nn.ModuleDict 대신 순서 유지 용이).
        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 1) 각 입력 key 별로 해당 모듈을 적용해 특징 리스트 생성.
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        # 2) 병합. add 는 공간/채널 shape 이 모두 같아야 하고, cat 은 채널만 다르면 됨.
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        # 3) 병합 후 공통 전처리.
        if self.post_input is not None:
            feat = self.post_input(feat)
        # 4) 공통 변환 본체.
        feat = self.middle(feat)
        # 5) 각 출력 head 를 적용해 feature_dict 에 추가 (in-place 갱신).
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    """`nn.Sequential` 의 변형. None 모듈을 자동으로 건너뛴다.

    - `nn.Sequential` 은 None 을 넣으면 에러를 내지만, `OpSequential` 은
      None 항목을 생성 시점에 제거하고 남은 모듈만 순차 실행한다.
    - 옵션으로 활성화/비활성화되는 레이어(예: dropout, auxiliary head) 를 편하게
      구성할 때 유용하다.
    """

    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        # None 항목 제거.
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 각 모듈을 순서대로 적용.
        for op in self.op_list:
            x = op(x)
        return x
