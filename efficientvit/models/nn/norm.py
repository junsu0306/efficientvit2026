"""정규화(normalization) 레이어 빌더 및 유틸리티.

이 모듈은 EfficientViT 에서 쓰는 정규화 레이어들을 한 곳에 모은다.

주요 구성 요소:
  * :class:`LayerNorm2d` — ``(B, C, H, W)`` 4D 텐서의 채널 축 기준 LayerNorm.
    ``torch.nn.LayerNorm`` 은 기본적으로 "마지막 축" 만 정규화하므로
    커스텀 구현이 필요하다.
  * :class:`TritonRMSNorm2d` — Triton 커널로 가속한 2D RMSNorm (선택적).
  * :func:`build_norm` — 이름 문자열로 정규화 레이어 인스턴스 생성 (팩토리).
  * :func:`set_norm_eps` — 모델 내 모든 BN/LN 의 ``eps`` 를 일괄 변경.
    (``cls_model_zoo`` 에서 B 시리즈는 ``1e-5``, L 시리즈는 ``1e-7`` 로 설정)
  * :func:`reset_bn` — 학습 후 BN 의 running mean/var 를 재계산하여
    모바일 배포 전에 통계를 안정화하는 루틴.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from efficientvit.models.nn.triton_rms_norm import TritonRMSNorm2dFunc
from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "build_norm", "reset_bn", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    """채널 차원(axis=1) 을 기준으로 LayerNorm 을 수행하는 2D 버전.

    ``nn.LayerNorm`` 은 ``normalized_shape`` 으로 지정한 "마지막 축" 을
    정규화하므로, NCHW 텐서에서 채널 차원을 정규화하려면 ``permute`` 가 필요하다.
    이 구현은 NCHW 를 유지한 채로 평균/분산을 채널 축에서 계산하여
    permute 오버헤드 없이 LN 을 적용한다.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 채널 축 평균 빼기 → zero-mean
        out = x - torch.mean(x, dim=1, keepdim=True)
        # 2) 채널 축 RMS 로 나눠 unit variance 로 스케일링. eps 는 수치 안정용.
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        # 3) elementwise affine 이 켜져 있으면 학습 가능한 γ/β 로 재스케일/재이동.
        #    채널 축(axis=1) 에 맞게 1x C x 1 x 1 로 view 해서 브로드캐스트.
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    """Triton 커널로 가속된 2D RMSNorm.

    평균을 빼지 않고 RMS 만으로 스케일링하는 RMSNorm 을 NCHW 에 적용.
    실제 연산은 ``TritonRMSNorm2dFunc`` (autograd function) 이 담당한다.
    ``nn.LayerNorm`` 을 상속해 weight/bias/eps 를 재사용한다.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)


# 정규화 레지스트리: 이름 → 클래스. ``build_norm`` 이 이 딕셔너리를 조회한다.
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,   # 표준 2D BatchNorm (B 시리즈 기본)
    "ln": nn.LayerNorm,       # 표준 LayerNorm (1D/2D/3D 마지막 축 기준)
    "ln2d": LayerNorm2d,      # NCHW 채널 축 기준 LN (L 시리즈 등에서 사용)
    "trms2d": TritonRMSNorm2d,  # Triton 가속 RMSNorm2d (옵션)
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    """이름과 채널/피처 수로 정규화 레이어를 생성하는 팩토리.

    Args:
        name: ``REGISTERED_NORM_DICT`` 의 키. 기본값 ``"bn2d"``.
        num_features: 채널(BN) 혹은 normalized_shape(LN/RMSNorm) 로 쓰일 정수.
        **kwargs: ``eps``, ``momentum``, ``affine`` 등 추가 생성자 인자.

    Returns:
        생성된 정규화 레이어. 이름이 미등록이면 ``None``.
    """
    # LN 계열은 ``normalized_shape`` 인자를, BN 계열은 ``num_features`` 인자를 받는다.
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        # 클래스가 실제로 받는 인자만 필터링 (불필요 키 제거).
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    """학습이 끝난 모델의 BN running mean/var 를 전체 데이터로 재계산한다.

    학습 중 momentum 기반으로 누적된 BN 통계는 최근 배치에 편향되어 있을 수
    있어, 배포 전에 전체 데이터셋(혹은 충분히 큰 calibration set) 을 한 바퀴
    돌며 통계를 다시 계산해 주는 것이 관례다. 특히 모바일 양자화/변환 전에
    수치 안정성을 높이는 데 유용하다.

    Args:
        model: BN 을 포함한 평가 대상 모델 (원본을 건드리지 않도록 내부에서 deepcopy).
        data_loader: ``images`` 텐서를 yield 하는 iterable (레이블은 필요 없음).
        sync: 분산 학습 시 rank 간 평균을 동기화할지 여부.
        progress_bar: tqdm 진행바 표시 여부.
    """
    import copy

    import torch.nn.functional as F
    from tqdm import tqdm

    from efficientvit.apps.utils import AverageMeter, is_master, sync_tensor
    from efficientvit.models.utils import get_device, list_join

    # 각 BN 레이어별 running mean/var 추정치를 누적할 meter.
    bn_mean = {}
    bn_var = {}

    # 원본 모델 보존을 위해 복사본 위에서 forward 를 모두 갈아끼운다.
    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter(is_distributed=False)
            bn_var[name] = AverageMeter(is_distributed=False)

            def new_forward(bn, mean_est, var_est):
                """각 BN 에 주입될 커스텀 forward 를 만드는 클로저."""

                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        # 분산 환경: 현재 rank 의 채널별 평균을 구한 뒤 전 rank 에서 모아 평균.
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_mean = sync_tensor(batch_mean, reduce="cat")
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                        batch_var = sync_tensor(batch_var, reduce="cat")
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        # 단일 프로세스: 현재 배치 내에서만 평균/분산 계산.
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    # (1, C, 1, 1) → (C,) 로 squeeze.
                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    # AverageMeter 에 배치 크기 가중치로 누적.
                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # 계산한 mean/var 로 직접 BN forward 수행 (running stat 미사용).
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,       # training=False: running stat 업데이트 안 함
                        0.0,         # momentum=0
                        bn.eps,      # 원 BN 의 eps 재사용
                    )

                return lambda_forward

            # 실제 주입: 해당 BN 모듈의 forward 를 교체.
            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # BN 이 하나도 없으면 할 일 없음.
    if len(bn_mean) == 0:
        return

    # 통계 수집을 위해 eval 모드 + no_grad 로 데이터 전수 순회.
    tmp_model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="reset bn", disable=not progress_bar or not is_master()) as t:
            for images in data_loader:
                images = images.to(get_device(tmp_model))
                tmp_model(images)
                t.set_postfix(
                    {
                        "bs": images.size(0),
                        "res": list_join(images.shape[-2:], "x"),
                    }
                )
                t.update()

    # 수집한 평균/분산을 원본 모델의 running_mean/running_var 에 복사.
    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    """모델 내 모든 GN/LN/BN 레이어의 ``eps`` 를 일괄 세팅한다.

    EfficientViT 모델 주(zoo) 에서는 시리즈에 따라 기본 eps 값이 다르다.
      * B 시리즈: ``1e-5`` (BN 위주)
      * L 시리즈: ``1e-7`` (더 깊고 넓은 네트워크에서 수치 안정성 조정)
    ``eps=None`` 이면 아무 것도 바꾸지 않는다.
    """
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
