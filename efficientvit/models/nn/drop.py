"""DropPath(stochastic depth) 적용 유틸리티.

DropPath 는 학습 중 residual 블록의 "본체(main) 경로" 를 확률적으로 0 으로
만들어 버리는 정규화 기법이다 (shortcut 만 통과). 깊은 네트워크(DeiT/Swin
등) 학습의 표준 레시피이며, EfficientViT 도 동일한 방식을 채택한다.

주요 함수:
  * :func:`apply_drop_func` — 학습 config 의 ``drop`` 섹션을 해석해 적절한
    drop 루틴(현재는 droppath 만 지원) 을 호출한다.
  * :func:`apply_droppath` — 네트워크를 순회하여 residual + identity shortcut
    조합의 블록(``EfficientViTBlock`` 등) 을 :class:`DropPathResidualBlock`
    으로 치환한다.

옵션 요약:
  * ``drop_prob`` — 최대 drop 확률.
  * ``linear_decay=True`` — 얕은 블록 → 깊은 블록으로 갈수록 drop_prob 를
    선형 증가시키는 DeiT/Swin 표준 레시피.
  * ``scheduled=True`` — 학습 진행도(``Scheduler.PROGRESS``, 0~1) 에 따라
    drop_prob 를 0 → 최대값으로 서서히 키움 (웜업 느낌).
  * ``skip=N`` — 앞쪽 ``N`` 개 valid 블록은 DropPath 적용에서 제외.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from efficientvit.apps.trainer.run_config import Scheduler
from efficientvit.models.nn.ops import IdentityLayer, ResidualBlock
from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["apply_drop_func"]


def apply_drop_func(network: nn.Module, drop_config: Optional[dict[str, Any]]) -> None:
    """학습 config 의 ``drop`` 블록을 읽고 해당 drop 기법을 모델에 적용한다.

    ``drop_config`` 예시::

        {"name": "droppath", "drop_prob": 0.1, "linear_decay": true, "skip": 0}

    ``drop_config`` 가 ``None`` 이면 아무 것도 하지 않는다.
    """
    if drop_config is None:
        return

    # 현재는 "droppath" 만 지원. 새 drop 기법을 추가하려면 여기에 등록.
    drop_lookup_table = {
        "droppath": apply_droppath,
    }

    drop_func = drop_lookup_table[drop_config["name"]]
    # 선택된 함수가 받는 인자만 config 에서 추려낸다.
    drop_kwargs = build_kwargs_from_config(drop_config, drop_func)

    drop_func(network, **drop_kwargs)


def apply_droppath(
    network: nn.Module,
    drop_prob: float,
    linear_decay=True,
    scheduled=True,
    skip=0,
) -> None:
    """네트워크의 identity-shortcut residual 블록에 DropPath 를 주입한다.

    Args:
        network: 대상 모델.
        drop_prob: 최대 drop 확률. ``linear_decay`` 이면 마지막 블록의 확률.
        linear_decay: True 면 블록 인덱스 ``i`` 에 대해
            ``prob = drop_prob * (i+1)/N`` 로 선형 증가 (DeiT/Swin 스타일).
        scheduled: True 면 학습 진행도에 따라 확률을 스케일 (Scheduler.PROGRESS).
        skip: 앞쪽에서 제외할 블록 수. 얕은 층은 drop 을 적용하지 않고 학습을
            안정화하는 데 쓰인다.
    """
    # DropPath 적용 대상: shortcut 이 IdentityLayer 인 ResidualBlock 만.
    # (다운샘플 블록처럼 shortcut 이 conv 인 경우에는 적용하지 않는다.)
    all_valid_blocks = []
    for m in network.modules():
        for name, sub_module in m.named_children():
            if isinstance(sub_module, ResidualBlock) and isinstance(sub_module.shortcut, IdentityLayer):
                all_valid_blocks.append((m, name, sub_module))
    # 앞쪽 skip 개는 건너뛴다.
    all_valid_blocks = all_valid_blocks[skip:]
    for i, (m, name, sub_module) in enumerate(all_valid_blocks):
        # 선형 decay 스케줄: 깊을수록 drop 확률 증가.
        prob = drop_prob * (i + 1) / len(all_valid_blocks) if linear_decay else drop_prob
        new_module = DropPathResidualBlock(
            sub_module.main,
            sub_module.shortcut,
            sub_module.post_act,
            sub_module.pre_norm,
            prob,
            scheduled,
        )
        # 부모 모듈의 해당 이름 속성을 교체 (setattr 와 동등).
        m._modules[name] = new_module


class DropPathResidualBlock(ResidualBlock):
    """DropPath 를 포함한 :class:`ResidualBlock` 변형.

    학습 중 매 배치마다 배치 내 각 샘플에 대해 독립적으로 main 경로를
    확률 ``drop_prob`` 로 0 으로 만든다 (shortcut 은 그대로 유지). 이는
    stochastic depth (Huang et al., 2016) 와 동일한 아이디어다.
    """

    def __init__(
        self,
        main: nn.Module,
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
        ######################################
        drop_prob: float = 0,
        scheduled=True,
    ):
        super().__init__(main, shortcut, post_act, pre_norm)

        # 해당 블록에서의 최종 drop 확률 (linear_decay 반영 후의 값).
        self.drop_prob = drop_prob
        # True 면 학습 진행도에 비례해 drop 확률을 스케일.
        self.scheduled = scheduled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 평가 모드이거나, drop_prob=0, 혹은 shortcut 이 identity 가 아니면
        # 일반 ResidualBlock 동작으로 대체.
        if not self.training or self.drop_prob == 0 or not isinstance(self.shortcut, IdentityLayer):
            return ResidualBlock.forward(self, x)
        else:
            drop_prob = self.drop_prob
            if self.scheduled:
                # Scheduler.PROGRESS 는 0(시작) → 1(끝) 로 증가하는 전역 진행도.
                drop_prob *= np.clip(Scheduler.PROGRESS, 0, 1)
            keep_prob = 1 - drop_prob

            # 배치 차원만 독립적으로 0/1 마스크 생성 (나머지 차원은 1 로 브로드캐스트).
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # 0 또는 1 로 이진화 → Bernoulli(keep_prob)

            # main 결과를 keep_prob 로 나눠 기대값 보정(inverted dropout) 후 마스크 곱.
            res = self.forward_main(x) / keep_prob * random_tensor + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
            return res
