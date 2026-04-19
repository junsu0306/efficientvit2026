"""활성함수(activation) 빌더 및 레지스트리 모듈.

EfficientViT 전역에서 사용하는 활성함수들을 문자열 이름으로 조회/생성한다.
- B 시리즈(모바일 친화 백본)는 ``hswish`` 를 기본으로 사용
- L 시리즈(대형 백본)는 ``gelu`` (tanh 근사) 를 사용
- GLU 계열(예: ``GLUMBConv``) 의 게이트 경로에는 ``silu`` 를 자주 사용

이 모듈의 핵심 API 는 :func:`build_act` 이며, ``ConvLayer``/``MBConv`` 등
거의 모든 기본 블록에서 ``act_func`` 문자열 인자를 받아 이 함수를 호출한다.
"""

from functools import partial
from typing import Optional

import torch.nn as nn

from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["build_act"]


# 활성함수 레지스트리: 문자열 이름 → nn.Module 클래스(또는 partial) 매핑.
# 새 활성함수를 추가하려면 여기에 키/값을 한 줄 추가하면 된다.
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,        # 표준 ReLU. 일반 CNN 스타일 블록
    "relu6": nn.ReLU6,      # [0,6] 클리핑 ReLU. MobileNet 계열 양자화 친화
    "hswish": nn.Hardswish,  # Hard-Swish (MobileNetV3). B 시리즈 기본 활성
    "silu": nn.SiLU,        # Swish/SiLU. GLU 변형의 게이트 경로에서 사용
    "gelu": partial(nn.GELU, approximate="tanh"),  # tanh 근사 GELU. L 시리즈 기본
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    """이름(문자열) 으로 활성함수 인스턴스를 생성한다.

    Args:
        name: ``REGISTERED_ACT_DICT`` 의 키. 예: ``"hswish"``, ``"gelu"``.
        **kwargs: 해당 활성함수 생성자에 전달할 추가 인자. ``build_kwargs_from_config``
            로 해당 클래스가 실제로 받는 인자만 필터링하여 넘긴다.

    Returns:
        생성된 ``nn.Module`` 인스턴스. 등록되지 않은 이름이면 ``None`` 을 반환하여
        "활성함수 없음" 상태를 명시적으로 표현할 수 있도록 한다.
    """
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        # 활성함수마다 허용 인자가 다르므로 config dict 에서 필요한 키만 추려낸다.
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        # 알 수 없는 이름이면 None 을 돌려준다. 호출부에서 "활성 없음" 으로 처리.
        return None
