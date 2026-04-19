"""네트워크/텐서 관련 공통 유틸리티.

이 모듈은 EfficientViT 전반에서 쓰이는 얇은 헬퍼들을 모아둔다.

주요 함수:
  * :func:`is_parallel` — 모델이 DP/DDP 래퍼로 감싸져 있는지 판별.
  * :func:`get_device` / :func:`get_dtype` — 모델 파라미터의 디바이스/dtype 조회.
  * :func:`get_same_padding` — 홀수 커널에 대해 "same" 패딩을 계산.
  * :func:`resize` — bilinear/bicubic/nearest/area 모드 공통 리사이즈 래퍼.
  * :func:`build_kwargs_from_config` — dict 에서 대상 함수의 시그니처와 교집합인
    키만 추려 반환. 팩토리에서 불필요한 인자 제거용으로 폭넓게 쓰인다.
  * :func:`load_state_dict_from_file` — 체크포인트 파일에서 state_dict 로드
    (``pretrained=True`` 옵션 등에서 사용).
  * :func:`get_submodule_weights` — prefix 기반으로 서브모듈 weight 추출.
  * :func:`get_dtype_from_str` — 문자열("fp32"/"fp16"/"bf16") → ``torch.dtype``.
"""

import collections
import os
from inspect import signature
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "is_parallel",
    "get_device",
    "get_same_padding",
    "resize",
    "build_kwargs_from_config",
    "load_state_dict_from_file",
    "get_submodule_weights",
]


def is_parallel(model: nn.Module) -> bool:
    """모델이 ``DataParallel`` 또는 ``DistributedDataParallel`` 로 감싸졌는지 여부."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_device(model: nn.Module) -> torch.device:
    """모델의 "첫 번째 파라미터" 가 올라가 있는 디바이스를 반환한다."""
    return model.parameters().__next__().device


def get_dtype(model: nn.Module) -> torch.dtype:
    """모델의 "첫 번째 파라미터" dtype 을 반환한다."""
    return model.parameters().__next__().dtype


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    """홀수 커널 크기에 대한 "same" 패딩 값을 계산한다.

    예: ``3 → 1``, ``5 → 2``, ``(3, 5) → (1, 2)``.
    짝수 커널은 대칭 same 패딩이 불가능하므로 assertion 으로 막는다.
    """
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    """공통 리사이즈 래퍼.

    ``bilinear``/``bicubic`` 은 ``align_corners`` 를 받지만, ``nearest``/``area`` 는
    해당 인자를 받지 않으므로 내부에서 분기한다.
    """
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    """``target_func`` 시그니처가 받는 키만 ``config`` 에서 추려 새 dict 로 반환.

    팩토리/빌더 패턴에서 사용자가 넘긴 거대한 config dict 에서 각 하위 모듈이
    실제로 소비 가능한 인자만 선택적으로 전달하고 싶을 때 쓴다.
    (``build_act``, ``build_norm``, ``apply_drop_func`` 등에서 사용)
    """
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    """로컬 경로의 체크포인트 파일에서 state_dict 를 로드한다.

    - ``~`` 홈 확장 및 realpath 정규화 후 ``torch.load(..., map_location="cpu",
      weights_only=True)`` 로 안전하게 로드.
    - 체크포인트가 ``{"state_dict": ..., "optimizer": ...}`` 형태라면
      ``only_state_dict=True`` 일 때 내부 state_dict 만 추출.

    ``pretrained=True`` 로 모델을 생성할 때 내부에서 호출된다.
    """
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    """state_dict 에서 특정 prefix 로 시작하는 서브모듈의 weight 만 추출한다.

    예를 들어 ``"backbone.stage3."`` prefix 로 서브네트워크 가중치만 골라낸 뒤
    prefix 를 잘라낸 새 OrderedDict 를 돌려주므로, 해당 서브모듈에 바로
    ``load_state_dict`` 할 수 있다.
    """
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights


def get_dtype_from_str(dtype: str) -> torch.dtype:
    """문자열 dtype 표기("fp32"/"fp16"/"bf16") 를 ``torch.dtype`` 으로 변환."""
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise NotImplementedError(f"dtype {dtype} is not supported")
