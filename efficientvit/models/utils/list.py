"""리스트/튜플 관련 얇은 유틸리티 모음.

대부분 모델 블록 래퍼(``ConvLayer``, ``MBConv``, ``GLUMBConv`` 등) 에서
"이 인자는 스칼라일 수도, 튜플일 수도 있다" 류의 정규화를 처리하거나,
학습 loop 에서 손실/지표 리스트를 평균·합산·포맷팅하는 데 쓰인다.

핵심 유틸:
  * :func:`list_sum`, :func:`list_mean`, :func:`weighted_list_sum`
    — 리스트 합/평균/가중합 (재귀 구현).
  * :func:`list_join` — sep 문자열로 값들을 포맷·연결 (로그 출력용).
  * :func:`val2list` — 스칼라를 길이 ``repeat_time`` 의 리스트로 확장.
  * :func:`val2tuple` — 스칼라/리스트 혼합 입력을 고정 길이 튜플로 정규화.
    ``ConvLayer`` 등에서 ``(norm, act)`` 같은 이중 인자를 받을 때 널리 사용된다.
  * :func:`squeeze_list` — 길이 1 리스트를 스칼라로 평탄화.
"""

from typing import Any, Optional

__all__ = [
    "list_sum",
    "list_mean",
    "weighted_list_sum",
    "list_join",
    "val2list",
    "val2tuple",
    "squeeze_list",
]


def list_sum(x: list) -> Any:
    """리스트 원소들의 합. 원소가 텐서/커스텀 객체여도 ``+`` 연산만 되면 동작한다."""
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> Any:
    """리스트 원소들의 산술 평균."""
    return list_sum(x) / len(x)


def weighted_list_sum(x: list, weights: list) -> Any:
    """원소별 가중 합 (``sum(x_i * w_i)``). 두 리스트 길이가 같아야 한다."""
    assert len(x) == len(weights)
    return x[0] * weights[0] if len(x) == 1 else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])


def list_join(x: list, sep="\t", format_str="%s") -> str:
    """리스트를 포맷 문자열로 변환 후 ``sep`` 로 이어 붙인다.

    예: ``list_join([1, 2, 3], sep="x", format_str="%d") == "1x2x3"``.
    해상도나 shape 을 로그에 찍을 때 자주 사용된다.
    """
    return sep.join([format_str % val for val in x])


def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    """스칼라는 길이 ``repeat_time`` 리스트로, 리스트/튜플은 그대로 리스트화."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """가변 입력을 길이 ``min_len`` 이상의 튜플로 정규화한다.

    동작:
      1) ``val2list`` 로 리스트화.
      2) 길이가 ``min_len`` 에 못 미치면 ``idx_repeat`` 위치 원소를 반복 삽입해
         모자란 수만큼 채운다. 기본 ``idx_repeat=-1`` 은 "마지막 원소" 를 반복.

    ``ConvLayer``/``MBConv`` 등에서 ``(norm, act)``, ``(norm1, norm2, norm3)`` 같은
    튜플 인자를 편하게 다루기 위해 사용된다. 예를 들어 사용자가 ``norm="bn2d"``
    한 개만 넘겨도 내부에서 ``("bn2d", "bn2d")`` 로 확장해 쓸 수 있다.
    """
    x = val2list(x)

    # 부족한 길이만큼 idx_repeat 위치 원소를 복제해 채운다.
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def squeeze_list(x: Optional[list]) -> list | Any:
    """길이 1 인 리스트는 그 원소만 꺼내고, 그 외에는 그대로 반환.

    다중 출력과 단일 출력을 같은 코드 경로에서 처리할 때 끝단에서 편의상 쓰인다.
    """
    if x is not None and len(x) == 1:
        return x[0]
    else:
        return x
