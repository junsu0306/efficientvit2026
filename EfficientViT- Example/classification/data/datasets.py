'''
Build trainining/testing datasets
'''
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

class SafeImageFolder(ImageFolder):
    """
    Corrupt/unreadable 이미지를 만나면 다른 샘플로 대체하는 ImageFolder.
    libjpeg segfault로 인한 core dump 방지.
    """
    def __getitem__(self, index):
        for _ in range(10):  # 최대 10번 다른 샘플로 재시도
            try:
                return super().__getitem__(index)
            except Exception:
                index = (index + 1) % len(self)
        raise RuntimeError("10회 연속 이미지 로드 실패 — 데이터셋을 확인하세요.")


class SubsetImageFolder(ImageFolder):
    """
    ImageFolder의 서브셋: 지정한 클래스 폴더만 로드하고 레이블을 0..N-1로 재매핑.

    Args:
        root: 클래스 폴더가 있는 루트 디렉토리 (예: .../imagenet/train)
        selected_classes: 사용할 클래스 폴더 이름 리스트 (예: ['00000', '00001', ...])
        transform: torchvision transforms

    동작 원리:
        1. ImageFolder로 전체 로드 → 원본 class_to_idx (폴더명 → 0..999)
        2. selected_classes에 해당하는 샘플만 필터링
        3. 레이블을 selected_classes 리스트 순서로 0..N-1 재매핑
        결과: len(selected_classes)개 클래스, 레이블 0..N-1
    """

    def __init__(self, root, selected_classes, transform=None):
        super().__init__(root, transform=transform)

        old_class_to_idx = self.class_to_idx  # {folder_name: old_label}

        # 존재하지 않는 클래스 경고
        valid = [c for c in selected_classes if c in old_class_to_idx]
        missing = set(selected_classes) - set(valid)
        if missing:
            print(f"[SubsetImageFolder] 경고: 없는 클래스 폴더 {missing}")

        # 새 레이블 매핑: valid[i] → i
        new_class_to_idx = {c: i for i, c in enumerate(valid)}
        old_to_new = {old_class_to_idx[c]: new_class_to_idx[c] for c in valid}

        # 샘플 필터링 + 레이블 재매핑
        self.samples = [
            (path, old_to_new[lbl])
            for path, lbl in self.samples
            if lbl in old_to_new
        ]
        self.targets = [s[1] for s in self.samples]
        self.classes = valid
        self.class_to_idx = new_class_to_idx


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        prefix = 'train' if is_train else 'val'
        data_dir = os.path.join(args.data_path, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = TimmDatasetTar(data_dir, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = SafeImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNETEE':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif args.data_set == 'IMNET10':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # --subset-classes 가 지정되면 해당 클래스 사용, 없으면 알파벳 순 첫 10개
        if hasattr(args, 'subset_classes') and args.subset_classes:
            selected = [c.strip() for c in args.subset_classes.split(',')]
        else:
            all_classes = sorted(os.listdir(root))
            selected = all_classes[:10]
        dataset = SubsetImageFolder(root, selected, transform=transform)
        nb_classes = len(dataset.classes)
        print(f"[IMNET10] 선택된 {nb_classes}개 클래스: {dataset.classes}")
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                                interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
