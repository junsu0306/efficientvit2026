"""
Reduced 모델 평가 스크립트
===========================
efficientvit_reducing.py로 생성된 dense 모델(_full.pth)의 ImageNet 정확도를 평가한다.

사용법:
  cd /workspace/etri_iitp/JS/EfficientViT/classification

  # 전체 1K 평가
  python eval_reduced.py \
    --model-path /workspace/.../reduced_m4_30pct_full.pth \
    --data-path /workspace/.../data/imagenet \
    --device cuda:0

  # 10-class 서브셋 평가
  python eval_reduced.py \
    --model-path /workspace/.../reduced_m4_subset10_30pct_full.pth \
    --data-path /workspace/.../data/imagenet \
    --data-set IMNET10 \
    --device cuda:0

  # soft-pruned 체크포인트에서 바로 reducing + 평가 (full.pth 없어도 됨)
  python eval_reduced.py \
    --checkpoint /workspace/.../output/pruning_30pct/checkpoint_best.pth \
    --data-path /workspace/.../data/imagenet \
    --device cuda:0
"""

import argparse
import sys
import os

import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from data.datasets import build_dataset
from engine import evaluate


def get_args():
    p = argparse.ArgumentParser('EfficientViT Reduced Model Eval')

    # 모델 로드 방식: --model-path 또는 --checkpoint 중 하나
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model-path', type=str,
                   help='reducing 완료된 full 모델 경로 (*_full.pth). '
                        'torch.load()로 즉시 로드.')
    g.add_argument('--checkpoint', type=str,
                   help='soft-pruned 학습 체크포인트 경로 (checkpoint_best.pth). '
                        '내부에서 reducing을 자동 적용.')

    p.add_argument('--data-path', required=True, type=str)
    p.add_argument('--data-set', default='IMNET',
                   choices=['IMNET', 'IMNET10', 'CIFAR'],
                   help='IMNET10: 10개 클래스 서브셋')
    p.add_argument('--subset-classes', default='', type=str,
                   help='IMNET10 전용: 쉼표 구분 클래스 폴더명. 비워두면 첫 10개.')
    p.add_argument('--num-classes', default=1000, type=int,
                   help='분류 클래스 수. IMNET10이면 10 (자동 설정됨)')
    p.add_argument('--model', default='EfficientViT_M4', type=str,
                   help='체크포인트가 state_dict일 때 사용할 모델명 (default: EfficientViT_M4)')
    p.add_argument('--input-size', default=224, type=int)
    p.add_argument('--batch-size', default=256, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--device', default='cuda:0', type=str)

    # build_dataset이 요구하는 나머지 인자 (기본값 유지)
    p.add_argument('--color-jitter', default=0.4, type=float)
    p.add_argument('--aa', default='rand-m9-mstd0.5-inc1', type=str)
    p.add_argument('--train-interpolation', default='bicubic', type=str)
    p.add_argument('--reprob', default=0.25, type=float)
    p.add_argument('--remode', default='pixel', type=str)
    p.add_argument('--recount', default=1, type=int)
    p.add_argument('--inat-category', default='name', type=str)
    p.add_argument('--finetune', default='', type=str)  # build_transform 분기용
    return p.parse_args()


def main():
    args = get_args()

    # IMNET10이면 num-classes 자동 설정
    if args.data_set == 'IMNET10' and args.num_classes == 1000:
        args.num_classes = 10

    device = torch.device(args.device)

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    if args.model_path:
        print(f"[Eval] 모델 로드: {args.model_path}")
        loaded = torch.load(args.model_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, dict):
            # 일반 학습 체크포인트 (state_dict dict) → 모델 생성 후 로드
            import model.build  # noqa: F401  ← EfficientViT timm 등록
            from timm.models import create_model
            sd = loaded.get('model', loaded)
            model = create_model(args.model, num_classes=args.num_classes,
                                 pretrained=False, fuse=False)
            model.load_state_dict(sd, strict=True)
            print(f"[Eval] state_dict 체크포인트 로드 완료 ({args.model})")
        else:
            # _full.pth: torch.save(model, ...) 로 저장된 전체 모델 객체
            model = loaded
    else:
        print(f"[Eval] soft-pruned 체크포인트에서 reducing 후 로드: {args.checkpoint}")
        from pruning.efficientvit_reducing import load_reduced_model
        model = load_reduced_model(
            checkpoint_path=args.checkpoint,
            num_classes=args.num_classes,
            device='cpu',
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Eval] 파라미터 수: {n_params:,}")
    model.to(device).eval()

    # ── 검증 데이터셋 ─────────────────────────────────────────────────────────
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    print(f"[Eval] 검증 샘플: {len(dataset_val)}, 클래스: {nb_classes}")

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── 평가 ──────────────────────────────────────────────────────────────────
    test_stats = evaluate(data_loader_val, model, device)
    print(f"\n[결과] Acc@1: {test_stats['acc1']:.2f}%  "
          f"Acc@5: {test_stats['acc5']:.2f}%  "
          f"Loss: {test_stats['loss']:.4f}")


if __name__ == '__main__':
    main()
