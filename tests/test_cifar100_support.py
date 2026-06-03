from pathlib import Path

import torch

from experiments.cifar10.run import build_parser
from residual_clipping.cifar10_data import DATASET_NAMES, dataset_num_classes, make_dataset
from residual_clipping.cifar10_models import get_model
from residual_clipping.cifar10_pipeline import sweep_output_dir


def test_cifar100_is_a_supported_dataset_with_100_classes(tmp_path: Path):
    assert "cifar100" in DATASET_NAMES
    assert "fake_cifar100" in DATASET_NAMES
    assert dataset_num_classes("cifar100") == 100
    assert dataset_num_classes("fake_cifar100") == 100

    dataset = make_dataset(
        "fake_cifar100",
        tmp_path,
        train=True,
        download=False,
        fake_size=7,
    )
    image, target = dataset[0]

    assert len(dataset) == 7
    assert image.shape == torch.Size([3, 32, 32])
    assert 0 <= target < 100


def test_vgg16_uses_requested_cifar100_classifier_width():
    model = get_model("vgg16", num_classes=dataset_num_classes("cifar100"))

    assert model.classifier[-1].out_features == 100


def test_cifar100_run_parser_accepts_dataset_choice():
    args = build_parser().parse_args(
        [
            "--model",
            "vgg16",
            "--optimizer-mode",
            "sgd_momentum",
            "--dataset",
            "cifar100",
        ]
    )

    assert args.dataset == "cifar100"


def test_sweep_output_dir_uses_dataset_name():
    assert sweep_output_dir("outputs/cifar10", "cifar10") == Path("outputs/cifar10/cifar10_sweeps")
    assert sweep_output_dir("outputs/cifar100", "cifar100") == Path("outputs/cifar100/cifar100_sweeps")
