"""Dataset and loader helpers for centralized CIFAR experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms


CIFAR_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}

DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "fake_cifar10": 10,
    "cifar100": 100,
    "fake_cifar100": 100,
}
DATASET_NAMES = tuple(DATASET_NUM_CLASSES)


def dataset_family(dataset_name: str) -> str:
    if dataset_name.startswith("fake_"):
        return dataset_name.removeprefix("fake_")
    return dataset_name


def dataset_num_classes(dataset_name: str) -> int:
    try:
        return DATASET_NUM_CLASSES[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def cifar_transforms(dataset_name: str, train: bool) -> transforms.Compose:
    stats = CIFAR_STATS[dataset_family(dataset_name)]
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )


def cifar10_transforms(train: bool) -> transforms.Compose:
    return cifar_transforms("cifar10", train)


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
            transform=cifar_transforms(dataset_name, train),
        )
    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=str(data_dir),
            train=train,
            download=download,
            transform=cifar_transforms(dataset_name, train),
        )
    if dataset_name in ("fake_cifar10", "fake_cifar100"):
        size = fake_size if fake_size > 0 else (512 if train else 128)
        return datasets.FakeData(
            size=size,
            image_size=(3, 32, 32),
            num_classes=dataset_num_classes(dataset_name),
            transform=cifar_transforms(dataset_name, train),
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
