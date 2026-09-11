"""Torch-required tests: CLI method dispatch, replay budget selection, replay loader.

These exercise the *training* entrypoint and its replay plumbing. They build run configs
only (``--validate-only``) — they never launch a training run, so no GPU, data, or DINOv2
checkpoint is needed, only that torch can be imported.

Skip behaviour: if torch is not available the module-level guard raises ``pytest.skip`` so a
CPU-only machine can still run the rest of the suite.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    from PIL import Image
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torch/PIL not available: {exc}", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continual.continual_grouped import (  # noqa: E402
    build_parser,
    config_from_namespace,
    NormalFileDataset,
    select_replay_files,
)
from src.data import get_data_transforms  # noqa: E402
from scripts.train_continual import main as train_main  # noqa: E402


# ---- method dispatch: all four paper methods are accepted by the CLI ------------ #
def test_train_continual_accepts_all_four_paper_methods():
    for method in ["shared_ft", "ewc", "shared_lora", "subspace_lora"]:
        rc = train_main(["--method", method, "--replay-buffer-size", "0",
                         "--output-dir", "/tmp/opret_out", "--validate-only"])
        assert rc == 0, method


def test_train_continual_accepts_replay_sizes():
    for n in ["0", "10", "30", "100"]:
        rc = train_main(["--method", "shared_ft", "--replay-buffer-size", n,
                         "--output-dir", "/tmp/opret_out", "--validate-only"])
        assert rc == 0, n


def test_config_builds_correct_method_and_replay():
    parser = build_parser()
    args = parser.parse_args([
        "--method", "subspace_lora", "--replay-buffer-size", "30",
        "--stream", "/tmp/stream.json", "--data-path", "/tmp/mvtec",
        "--output-dir", "/tmp/out",
    ])
    config = config_from_namespace(args, parser)
    assert config.method == "subspace_lora"
    assert config.replay_buffer_size == 30
    assert config.lora_rank == 4  # parser default


def test_train_continual_rejects_non_paper_method():
    # 'isolation' is a valid upstream method but not one of the four the paper runs train.
    with pytest.raises(SystemExit):
        train_main(["--method", "isolation", "--replay-buffer-size", "0",
                    "--output-dir", "/tmp/opret_out", "--validate-only"])


# ---- replay budget selection (pure helper factored out of continual_grouped) ---- #
def test_select_replay_files_keeps_all_under_budget():
    rng = random.Random(1)
    store = {"cat_a": [f"a{i}.png" for i in range(3)],
             "cat_b": [f"b{i}.png" for i in range(2)]}
    out = select_replay_files(store, 10, rng)
    assert sorted(out) == sorted(store["cat_a"] + store["cat_b"])


def test_select_replay_files_samples_budget_deterministically():
    store = {"cat": [f"x{i}.png" for i in range(100)]}
    f1 = select_replay_files(store, 10, random.Random(42))
    f2 = select_replay_files(store, 10, random.Random(42))
    assert len(f1) == 10 and len(set(f1)) == 10
    assert f1 == f2


def test_select_replay_files_respects_per_class_budget():
    store = {"cat_a": [f"a{i}.png" for i in range(5)],
             "cat_b": [f"b{i}.png" for i in range(5)]}
    rng = random.Random(3)
    out = select_replay_files(store, 2, rng)
    assert len(out) == 4                       # 2 per class
    assert any(p.startswith("a") for p in out)
    assert any(p.startswith("b") for p in out)


# ---- replay data loader: NormalFileDataset over tiny temp images ----------------- #
def test_normal_file_dataset_loads_and_transforms():
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            p = Path(td) / f"n{i}.png"
            Image.fromarray((np.random.default_rng(i).random((16, 16, 3)) * 255)
                            .astype("uint8")).save(p)
            paths.append(str(p))
        transform, _ = get_data_transforms(32, 16)   # resize 32, center-crop 16
        ds = NormalFileDataset(paths, transform)
        assert len(ds) == 3
        tensor, path = ds[0]
        assert torch.is_tensor(tensor)
        assert tensor.shape == (3, 16, 16)           # RGB, crop 16*16
        assert Path(path).exists()
