"""Soft Pruning 으로 학습된 체크포인트를 작은 Dense 모델로 변환하는 진입점.

사용법:
    python applications/efficientvit_cls/reduce_efficientvit_cls_model.py \
        --model efficientvit-b1 \
        --checkpoint /path/to/.exp/.../checkpoint/model_best.pt \
        --output  /path/to/reduced_b1.pt
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.clscore.pruning.efficientvit_reducing import main

if __name__ == "__main__":
    main()
