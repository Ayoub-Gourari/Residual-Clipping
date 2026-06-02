"""Dataset and loader helpers for centralized CIFAR-10 experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms


CIFAR10_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2023, 0.1994, 0.2010),
}


def cifar10_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_STATS["mean"], CIFAR10_STATS["std"]),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_STATS["mean"], CIFAR10_STATS["std"]),
        ]
    )


def make_dataset(
    dataset_name: str,
    data_dir: str | Path,
    *,
    train: bool,
    download: bool,
    fake_size: int,
):
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=str(data_dir),
            train=train,
            download=download,
            transform=cifar10_transforms(train),
        )
    if dataset_name == "fake_cifar10":
        size = fake_size if fake_size > 0 else (512 if train else 128)
        return datasets.FakeData(
            size=size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=cifar10_transforms(train),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_dataloaders(
    dataset_name: str,
    data_dir: str | Path,
    *,
    batch_size: int,
    test_batch_size: int,
    download: bool,
    loader_workers: int,
    pin_memory: bool,
    fake_train_size: int,
    fake_test_size: int,
):
    train_dataset = make_dataset(
        dataset_name,
        data_dir,
        train=True,
        download=download,
        fake_size=fake_train_size,
    )
    test_dataset = make_dataset(
        dataset_name,
        data_dir,
        train=False,
        download=download,
        fake_size=fake_test_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=loader_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
