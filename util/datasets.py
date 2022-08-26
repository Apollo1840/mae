# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset_pretrain(args):

    root = os.path.join(args.data_path, 'train')
    dataset = datasets.ImageFolder(root, transform=create_transform_pretrain(args))
    print(dataset)

    return dataset


def build_dataset_finetune(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform_train(mean, std, args)
    else:
        transform = create_transform_val(mean, std, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)

    return dataset


def create_transform_pretrain(args):
    return transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )


def create_transform_train(mean, std, args):
    # this should always dispatch to transforms_imagenet_train
    return create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation='bicubic',
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=mean,
        std=std,
    )


def create_transform_val(mean, std, args):
    # get the size to resize.
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)

    t = [transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images,
         transforms.CenterCrop(args.input_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]

    return transforms.Compose(t)
