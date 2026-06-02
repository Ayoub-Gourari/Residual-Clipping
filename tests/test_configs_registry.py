import tempfile
from pathlib import Path

from residual_clipping.configs import config_to_argv, load_yaml_config
from residual_clipping.registry import get_registry_entry, load_registry, registry_frame


def test_load_yaml_config_reads_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "config.yaml"
        path.write_text("alpha: 1\nname: demo\n", encoding="utf-8")
        config = load_yaml_config(path)
        assert config == {"alpha": 1, "name": "demo"}


def test_config_to_argv_serializes_lists_and_flags():
    argv = config_to_argv(
        {
            "models": ["resnet20", "resnet18"],
            "resume": True,
            "download": False,
            "output_dir": "outputs/cifar10",
            "clip_values": [0.1, 0.3],
            "wandb_mode": "disabled",
            "unused": None,
        }
    )
    assert argv == [
        "--models",
        "resnet20,resnet18",
        "--resume",
        "--output-dir",
        "outputs/cifar10",
        "--clip-values",
        "0.1,0.3",
        "--wandb-mode",
        "disabled",
    ]


def test_registry_load_and_lookup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "registry.yaml"
        path.write_text(
            "\n".join(
                [
                    "experiments:",
                    "  - name: cifar10-resnet20-sweep",
                    "    kind: sweep",
                    "    family: cifar10",
                    "    config: configs/cifar10/resnet20_sweep.yaml",
                    "    description: Demo entry",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        entries = load_registry(path)
        assert len(entries) == 1
        assert entries[0].name == "cifar10-resnet20-sweep"
        assert entries[0].kind == "sweep"

        entry = get_registry_entry(path, "cifar10-resnet20-sweep")
        assert entry.config == "configs/cifar10/resnet20_sweep.yaml"

        frame = registry_frame(path)
        assert list(frame["name"]) == ["cifar10-resnet20-sweep"]
        assert list(frame["family"]) == ["cifar10"]


def test_registry_lookup_raises_for_missing_entry():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "registry.yaml"
        path.write_text("experiments: []\n", encoding="utf-8")
        try:
            get_registry_entry(path, "does-not-exist")
        except KeyError as exc:
            assert "does-not-exist" in str(exc)
        else:
            raise AssertionError("Expected KeyError for unknown registry entry.")
