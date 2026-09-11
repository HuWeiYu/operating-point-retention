#!/usr/bin/env python3
"""Grouped continual anomaly detection experiments for InP-Former."""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data import MVTecDataset, RealIADDataset, get_data_transforms
from src.models import vit_encoder
from src.models.uad import INP_Former
from src.models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
from src.optimizers import StableAdamW
from src.utils import cal_anomaly_maps, get_gaussian_kernel, global_cosine_hm_adaptive, setup_seed
from src.continual.inflora_adaptation import (
    DualGPMState,
    collect_covariances,
    configure_cumulative_task,
    cumulative_lora_state_dict,
    cumulative_update_report,
    current_trainable_parameters,
    decoder_attention_target,
    inject_cumulative_subspace_lora,
    load_cumulative_lora_state,
)


EXPERT_PREFIXES = ("bottleneck.", "aggregation.", "decoder.", "prototype_token")
TASK_SPECIFIC_PREFIXES = ("bottleneck.", "aggregation.", "prototype_token")
LORA_METHODS = ("shared_lora", "isolated_lora")
ISOLATED_METHODS = ("isolation", "isolated_lora", "isolated_head")
SUBSPACE_METHODS = ("subspace_lora",)
ALL_LORA_METHODS = LORA_METHODS + SUBSPACE_METHODS
ROUTED_METHODS = ISOLATED_METHODS + SUBSPACE_METHODS
FROZEN_DECODER_METHODS = (
    "shared_lora",
    "isolated_lora",
    "isolated_head",
    "subspace_lora",
)
COMPACT_ISOLATED_METHODS = ("isolated_lora", "isolated_head")
K3_ROUTED_METHODS = COMPACT_ISOLATED_METHODS + SUBSPACE_METHODS
SHARED_METHODS = ("shared_ft", "ewc", "shared_lora", "lwf")


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class RunConfig:
    method: str
    stream: str
    data_path: str
    output_dir: str
    encoder: str
    seed: int
    input_size: int
    crop_size: int
    inp_num: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    final_learning_rate: float
    weight_decay: float
    validation_fraction: float
    threshold_quantile: float
    max_tasks: int
    max_train_batches: int
    max_eval_batches: int
    validation_audit_only: bool
    resize_mask: int
    save_pixel_maps: bool
    ewc_lambda: float
    fisher_batches: int
    save_regularizer_state: bool
    lwf_lambda: float
    lwf_gradient_audit_batches: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    subspace_lambda_start: float
    subspace_lambda_end: float
    subspace_covariance_batches: int
    compact_audit_storage: bool
    replay_buffer_size: int = 0
    replay_selector: str = "random"


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base module.")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        update = F.linear(self.dropout(inputs), self.lora_B @ self.lora_A)
        return self.base(inputs) + self.scaling * update


def inject_lora(module, rank, alpha, dropout=0.0):
    replaced = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            replaced.append(name)
        else:
            replaced.extend(
                f"{name}.{nested}"
                for nested in inject_lora(child, rank, alpha, dropout)
            )
    return replaced


def configure_shared_lora(model, rank, alpha, dropout):
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False
    replaced = inject_lora(model.decoder, rank, alpha, dropout)
    if not replaced:
        raise RuntimeError("No decoder Linear modules were found for LoRA injection.")
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(("bottleneck.", "aggregation.", "prototype_token"))
        or ".lora_A" in name
        or ".lora_B" in name
    ]
    trainable_ids = {id(parameter) for parameter in trainable}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in trainable_ids
    return trainable, replaced


def configure_frozen_decoder_head(model):
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(TASK_SPECIFIC_PREFIXES)
    ]
    trainable_ids = {id(parameter) for parameter in trainable}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in trainable_ids
    return trainable


def task_specific_head_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(TASK_SPECIFIC_PREFIXES)
    }


def load_task_specific_head_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = set(task_specific_head_state_dict(model))
    missing_required = sorted(required.intersection(missing))
    if missing_required or unexpected:
        raise RuntimeError(
            "Task-specific head state mismatch; "
            f"missing={missing_required}, unexpected={unexpected}"
        )


def task_specific_lora_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(TASK_SPECIFIC_PREFIXES)
        or (key.startswith("decoder.") and (".lora_A" in key or ".lora_B" in key))
    }


def load_task_specific_lora_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = set(task_specific_lora_state_dict(model))
    missing_required = sorted(required.intersection(missing))
    if missing_required or unexpected:
        raise RuntimeError(
            "Task-specific LoRA state mismatch; "
            f"missing={missing_required}, unexpected={unexpected}"
        )


def frozen_decoder_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("decoder.")
        and ".lora_A" not in key
        and ".lora_B" not in key
        and not key.endswith(".active_task")
    }


def configure_subspace_trainable(model, layers):
    """Train only the current cumulative B branch and current task head."""

    current_lora = current_trainable_parameters(layers.values())
    task_head = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(TASK_SPECIFIC_PREFIXES)
    ]
    selected = task_head + current_lora
    selected_ids = {id(parameter) for parameter in selected}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in selected_ids
    return selected


def audit_subspace_trainable_boundary(model, layers, task, trainable):
    selected_ids = {id(parameter) for parameter in trainable}
    required_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    names = parameter_names_for(model, trainable)
    invalid_names = []
    current_suffix = f".lora_B.{task}"
    for name in names:
        if name.startswith(TASK_SPECIFIC_PREFIXES):
            continue
        if name.endswith(current_suffix):
            continue
        invalid_names.append(name)
    old_branch_trainable = [
        name
        for name, parameter in model.named_parameters()
        if ".lora_B." in name
        and not name.endswith(current_suffix)
        and parameter.requires_grad
    ]
    fixed_a_trainable = [
        name
        for name, parameter in model.named_parameters()
        if ".lora_A." in name and parameter.requires_grad
    ]
    base_trainable = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.")
        and ".base." in name
        and parameter.requires_grad
    ]
    layer_tasks = {name: layer.current_task for name, layer in layers.items()}
    if (
        selected_ids != required_ids
        or invalid_names
        or old_branch_trainable
        or fixed_a_trainable
        or base_trainable
        or set(layer_tasks.values()) != {task}
    ):
        raise RuntimeError(
            "Invalid subspace optimizer boundary; "
            f"selected_matches_requires_grad={selected_ids == required_ids}, "
            f"invalid_names={invalid_names}, "
            f"old_branch_trainable={old_branch_trainable}, "
            f"fixed_a_trainable={fixed_a_trainable}, "
            f"base_trainable={base_trainable}, layer_tasks={layer_tasks}"
        )
    return {
        "trainable_parameter_names": names,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "current_task": task + 1,
        "adapted_layers": sorted(layers),
        "optimizer_contains_decoder_base": bool(base_trainable),
        "optimizer_contains_fixed_A": bool(fixed_a_trainable),
        "optimizer_contains_old_B": bool(old_branch_trainable),
    }


def parameter_names_for(model, parameters):
    parameter_ids = {id(parameter) for parameter in parameters}
    return sorted(
        name for name, parameter in model.named_parameters() if id(parameter) in parameter_ids
    )


def audit_lora_trainable_boundary(model, trainable):
    trainable = list(trainable)
    selected_ids = {id(parameter) for parameter in trainable}
    required_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    names = parameter_names_for(model, trainable)
    invalid_names = [
        name
        for name in names
        if not name.startswith(TASK_SPECIFIC_PREFIXES)
        and ".lora_A" not in name
        and ".lora_B" not in name
    ]
    if selected_ids != required_ids or invalid_names:
        raise RuntimeError(
            "Invalid compact optimizer boundary; "
            f"selected_matches_requires_grad={selected_ids == required_ids}, "
            f"invalid_names={invalid_names}"
        )
    return {
        "optimizer_parameter_count": sum(parameter.numel() for parameter in trainable),
        "optimizer_parameter_tensor_count": len(trainable),
        "optimizer_parameter_names_sha256": hash_strings(names),
        "optimizer_contains_decoder_base": any(".base." in name for name in names),
        "optimizer_parameter_names": names,
    }


class NormalFileDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = [str(path) for path in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            transformed = self.transform(image.convert("RGB"))
        return transformed, path


def is_realiad_dataset(dataset_name):
    normalized = str(dataset_name).lower().replace("_", "").replace("-", "")
    return normalized == "realiad"


class GroupedTestDataset(Dataset):
    def __init__(self, data_path, categories, transform, gt_transform, dataset_name="MVTec-AD"):
        self.parts = []
        self.offsets = []
        total = 0
        for category in categories:
            if is_realiad_dataset(dataset_name):
                dataset = RealIADDataset(
                    root=str(data_path),
                    category=category,
                    transform=transform,
                    gt_transform=gt_transform,
                    phase="test",
                )
            else:
                dataset = MVTecDataset(
                    root=str(Path(data_path) / category),
                    transform=transform,
                    gt_transform=gt_transform,
                    phase="test",
                )
            self.parts.append((category, dataset))
            total += len(dataset)
            self.offsets.append(total)

    def __len__(self):
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, index):
        part_index = next(i for i, end in enumerate(self.offsets) if index < end)
        start = 0 if part_index == 0 else self.offsets[part_index - 1]
        category, dataset = self.parts[part_index]
        image, mask, label, path = dataset[index - start]
        return image, mask, label, path, category


def collect_images(folder):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in Path(folder).rglob("*") if path.suffix.lower() in extensions)


def load_realiad_normal_train_files(data_path, category):
    json_path = (
        Path(data_path)
        / "realiad_jsons"
        / "realiad_jsons"
        / f"{category}.json"
    )
    if not json_path.is_file():
        raise FileNotFoundError(f"Real-IAD split JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    files = []
    for sample in split.get("train", []):
        if sample.get("anomaly_class") != "OK":
            raise RuntimeError(
                f"Real-IAD training split for {category} contains a non-normal sample."
            )
        image_path = (
            Path(data_path)
            / "realiad_1024"
            / category
            / sample["image_path"]
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Real-IAD training image not found: {image_path}")
        files.append(image_path)
    if len(files) != len(set(files)):
        raise RuntimeError(f"Real-IAD training split for {category} contains duplicates.")
    return sorted(files)


def _split_category(
    data_path,
    category,
    validation_fraction,
    seed,
    dataset_name="MVTec-AD",
):
    """Return (train_files, validation_files) for one category.

    Keeps the same deterministic shuffle as split_normal_files so that a
    per-category view is consistent with the combined train/validation split.
    """
    files = (
        load_realiad_normal_train_files(data_path, category)
        if is_realiad_dataset(dataset_name)
        else collect_images(Path(data_path) / category / "train" / "good")
    )
    if not files:
        raise FileNotFoundError(f"No normal training images found for {category}")
    category_seed = int(hashlib.sha256(f"{seed}:{category}".encode()).hexdigest()[:8], 16)
    rng = random.Random(category_seed)
    rng.shuffle(files)
    validation_count = max(1, int(round(len(files) * validation_fraction)))
    return files[validation_count:], files[:validation_count]


def split_normal_files(
    data_path,
    categories,
    validation_fraction,
    seed,
    dataset_name="MVTec-AD",
):
    train_files, validation_files = [], []
    for category in categories:
        train, validation = _split_category(
            data_path, category, validation_fraction, seed, dataset_name
        )
        train_files.extend(train)
        validation_files.extend(validation)
    return sorted(train_files), sorted(validation_files)


def hash_strings(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def state_dict_sha256(state):
    """Hash tensor names, metadata, parameters, and buffers deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def structured_state_sha256(value):
    """Hash nested checkpoint metadata and tensors deterministically."""

    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, bool):
            digest.update(b"bool:1" if item else b"bool:0")
        elif isinstance(item, (int, float, str)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))
        else:
            raise TypeError(f"Unsupported checkpoint value for hashing: {type(item)}")

    update(value)
    return digest.hexdigest()


def subspace_checkpoint_state(
    model,
    layers,
    history,
    stage,
    config,
    frozen_decoder_hash,
    task_heads,
    thresholds,
    router_prototypes,
    task_categories,
):
    state = {
        "format_version": 1,
        "method": "subspace_lora",
        "stage": int(stage),
        "task_heads": [
            {name: value.detach().cpu().clone() for name, value in head.items()}
            for head in task_heads
        ],
        "cumulative_lora": cumulative_lora_state_dict(layers, task=stage),
        "dual_gpm": history.state_dict(),
        "calibration_thresholds": [float(value) for value in thresholds],
        "router_prototypes": [
            value.detach().cpu().clone() for value in router_prototypes
        ],
        "task_categories": [list(categories) for categories in task_categories],
        "algorithm": {
            "rank": int(config.lora_rank),
            "alpha": float(config.lora_alpha),
            "dropout": float(config.lora_dropout),
            "lambda_start": float(config.subspace_lambda_start),
            "lambda_end": float(config.subspace_lambda_end),
            "targets": sorted(layers),
        },
        "frozen_decoder_sha256": frozen_decoder_hash,
    }
    state["complete_state_sha256"] = structured_state_sha256(state)
    return state


def load_subspace_checkpoint(
    model,
    layers,
    history,
    state,
    frozen_decoder_hash,
    verify_integrity=True,
):
    if state.get("format_version") != 1 or state.get("method") != "subspace_lora":
        raise RuntimeError("Unsupported subspace checkpoint format.")
    if verify_integrity:
        stored_hash = state.get("complete_state_sha256")
        unhashed = {
            key: value
            for key, value in state.items()
            if key != "complete_state_sha256"
        }
        if stored_hash != structured_state_sha256(unhashed):
            raise RuntimeError("Subspace checkpoint complete-state hash mismatch.")
        if state.get("frozen_decoder_sha256") != frozen_decoder_hash:
            raise RuntimeError("Subspace checkpoint frozen Decoder hash mismatch.")
        if state_dict_sha256(frozen_decoder_state_dict(model)) != frozen_decoder_hash:
            raise RuntimeError("Current frozen Decoder base does not match the checkpoint.")
    expected_targets = sorted(layers)
    algorithm = state.get("algorithm", {})
    if algorithm.get("targets") != expected_targets:
        raise RuntimeError("Subspace checkpoint target-layer mismatch.")
    for layer in layers.values():
        if int(algorithm.get("rank", -1)) != layer.rank:
            raise RuntimeError("Subspace checkpoint rank mismatch.")
        if float(algorithm.get("alpha", float("nan"))) != layer.alpha:
            raise RuntimeError("Subspace checkpoint alpha mismatch.")
        if float(algorithm.get("dropout", float("nan"))) != layer.dropout.p:
            raise RuntimeError("Subspace checkpoint dropout mismatch.")
    stage = int(state["stage"])
    task_heads = state.get("task_heads")
    if not isinstance(task_heads, list) or len(task_heads) != stage + 1:
        raise RuntimeError("Subspace checkpoint task-head coverage is incomplete.")
    if len(state.get("calibration_thresholds", [])) != stage + 1:
        raise RuntimeError("Subspace checkpoint calibration coverage is incomplete.")
    if len(state.get("router_prototypes", [])) != stage + 1:
        raise RuntimeError("Subspace checkpoint router coverage is incomplete.")
    if len(state.get("task_categories", [])) != stage + 1:
        raise RuntimeError("Subspace checkpoint task mapping is incomplete.")
    load_task_specific_head_state(model, task_heads[stage])
    active_task = load_cumulative_lora_state(layers, state["cumulative_lora"])
    history.load_state_dict(state["dual_gpm"])
    if active_task != stage:
        raise RuntimeError("Subspace checkpoint stage and active task disagree.")
    return active_task


def scalar_sha256(value):
    return hashlib.sha256(repr(float(value)).encode("ascii")).hexdigest()


def load_stream(path, max_tasks):
    with open(path, "r", encoding="utf-8") as handle:
        stream = json.load(handle)
    tasks = stream["tasks"]
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("The stream contains no tasks.")
    return stream, tasks


def build_encoder(name, device):
    encoder = vit_encoder.load(name).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def build_expert(encoder, encoder_name, inp_num, device, initialize=True):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported encoder architecture: {encoder_name}")

    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    prototypes = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim))])
    aggregation = nn.ModuleList(
        [
            Aggregation_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
        ]
    )
    decoder = nn.ModuleList(
        [
            Prototype_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(8)
        ]
    )
    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=aggregation,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        prototype_token=prototypes,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, decoder, aggregation, prototypes])
    if initialize:
        for module in trainable.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    return model, trainable


def expert_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(EXPERT_PREFIXES)
    }


def load_expert_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("encoder.")]
    if missing or unexpected:
        raise RuntimeError(f"Expert state mismatch; missing={missing}, unexpected={unexpected}")


def count_parameters(model):
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    expert_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(EXPERT_PREFIXES)
    )
    return encoder_parameters, expert_parameters


def expert_named_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(EXPERT_PREFIXES)
    }


def ewc_penalty(model, ewc_state):
    if not ewc_state:
        return torch.zeros((), device=next(model.parameters()).device)
    parameters = expert_named_parameters(model)
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, (anchor, fisher) in ewc_state.items():
        penalty = penalty + (fisher * (parameters[name] - anchor).square()).sum()
    return penalty


def train_expert(model, trainable, loader, config, device, ewc_state=None, replay_loader=None):
    model.train()
    model.encoder.eval()
    trainable_parameters = (
        list(trainable.parameters()) if isinstance(trainable, nn.Module) else list(trainable)
    )
    optimizer = StableAdamW(
        [{"params": trainable_parameters}],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    total_steps = max(1, config.epochs * min(len(loader), config.max_train_batches or len(loader)))
    replay_iter = cycle(replay_loader) if replay_loader is not None else None
    losses = []
    step = 0
    for epoch in range(config.epochs):
        progress = tqdm(loader, desc=f"train {epoch + 1}/{config.epochs}", ncols=90)
        for batch_index, (images, _) in enumerate(progress):
            if config.max_train_batches > 0 and batch_index >= config.max_train_batches:
                break
            ratio = step / max(1, total_steps - 1)
            learning_rate = config.final_learning_rate + 0.5 * (
                config.learning_rate - config.final_learning_rate
            ) * (1.0 + np.cos(np.pi * ratio))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            images = images.to(device, non_blocking=True)
            if replay_iter is not None:
                replay_images, _ = next(replay_iter)
                images = torch.cat(
                    [images, replay_images.to(device, non_blocking=True)], dim=0
                )
            encoded, decoded, gather_loss = model(images)
            loss = global_cosine_hm_adaptive(encoded, decoded, y=3) + 0.2 * gather_loss
            if ewc_state:
                loss = loss + 0.5 * config.ewc_lambda * ewc_penalty(model, ewc_state)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.1)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", lr=f"{learning_rate:.2e}")
    return losses


def collect_subspace_covariances(model, layers, loader, config, device):
    """Collect only layerwise second moments on current-task normal inputs."""

    model.eval()
    model.encoder.eval()
    for layer in layers.values():
        layer.start_covariance_collection(reset=True)
    batches = 0
    try:
        with torch.no_grad():
            for batch_index, (images, _) in enumerate(
                tqdm(loader, desc="subspace covariance", ncols=90)
            ):
                if (
                    config.subspace_covariance_batches > 0
                    and batch_index >= config.subspace_covariance_batches
                ):
                    break
                model(images.to(device, non_blocking=True))
                batches += 1
    finally:
        for layer in layers.values():
            layer.stop_covariance_collection()
    if batches == 0:
        raise RuntimeError("Subspace covariance collection processed no batches.")
    return collect_covariances(layers, reset=True), batches


def gradient_cosine(first_loss, second_loss, parameters):
    first_gradients = torch.autograd.grad(
        first_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second_gradients = torch.autograd.grad(
        second_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    dot = torch.zeros((), device=first_loss.device)
    first_norm = torch.zeros((), device=first_loss.device)
    second_norm = torch.zeros((), device=first_loss.device)
    for first, second in zip(first_gradients, second_gradients):
        if first is None or second is None:
            continue
        dot = dot + (first * second).sum()
        first_norm = first_norm + first.square().sum()
        second_norm = second_norm + second.square().sum()
    denominator = first_norm.sqrt() * second_norm.sqrt()
    if denominator.item() == 0:
        return float("nan")
    return float((dot / denominator).detach().cpu())


def train_expert_lwf(
    model,
    teacher_model,
    trainable,
    loader,
    config,
    device,
    teacher_threshold,
):
    if teacher_model is None:
        raise ValueError("LwF requires a frozen previous-stage teacher.")
    model.train()
    model.encoder.eval()
    teacher_model.eval()
    trainable_parameters = (
        list(trainable.parameters()) if isinstance(trainable, nn.Module) else list(trainable)
    )
    optimizer = StableAdamW(
        [{"params": trainable_parameters}],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    total_steps = max(1, config.epochs * min(len(loader), config.max_train_batches or len(loader)))
    with preserve_rng_state():
        gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    losses = []
    ad_losses = []
    distillation_losses = []
    gradient_cosines = []
    zero_gradient_audit_batches = 0
    incoming_teacher_scores = []
    step = 0
    gradient_audit_attempts = 0
    max_gradient_audit_attempts = config.lwf_gradient_audit_batches + 4
    for epoch in range(config.epochs):
        progress = tqdm(loader, desc=f"train {epoch + 1}/{config.epochs}", ncols=90)
        for batch_index, (images, _) in enumerate(progress):
            if config.max_train_batches > 0 and batch_index >= config.max_train_batches:
                break
            ratio = step / max(1, total_steps - 1)
            learning_rate = config.final_learning_rate + 0.5 * (
                config.learning_rate - config.final_learning_rate
            ) * (1.0 + np.cos(np.pi * ratio))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_maps = anomaly_maps(
                    teacher_model,
                    images,
                    config.resize_mask,
                    gaussian,
                )
            encoded, decoded, gather_loss = model(images)
            ad_loss = global_cosine_hm_adaptive(encoded, decoded, y=3) + 0.2 * gather_loss
            if config.lwf_lambda == 0.0:
                with torch.no_grad():
                    student_maps = anomaly_maps_from_features(
                        encoded,
                        [feature.detach() for feature in decoded],
                        images.shape[-1],
                        config.resize_mask,
                        gaussian,
                    )
                distillation_loss = F.mse_loss(student_maps, teacher_maps)
                loss = ad_loss
            else:
                student_maps = anomaly_maps_from_features(
                    encoded,
                    decoded,
                    images.shape[-1],
                    config.resize_mask,
                    gaussian,
                )
                distillation_loss = F.mse_loss(student_maps, teacher_maps)
                loss = ad_loss + config.lwf_lambda * distillation_loss
            if (
                config.lwf_lambda > 0.0
                and config.lwf_gradient_audit_batches > 0
                and len(gradient_cosines) < config.lwf_gradient_audit_batches
                and gradient_audit_attempts < max_gradient_audit_attempts
            ):
                cosine = gradient_cosine(ad_loss, distillation_loss, trainable_parameters)
                gradient_audit_attempts += 1
                if np.isfinite(cosine):
                    gradient_cosines.append(cosine)
                else:
                    zero_gradient_audit_batches += 1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.1)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ad_losses.append(float(ad_loss.detach().cpu()))
            distillation_losses.append(float(distillation_loss.detach().cpu()))
            if epoch == 0:
                incoming_teacher_scores.extend(
                    image_scores(teacher_maps).detach().cpu().numpy().tolist()
                )
            step += 1
            progress.set_postfix(
                loss=f"{np.mean(losses[-20:]):.4f}",
                distill=f"{np.mean(distillation_losses[-20:]):.4f}",
                lr=f"{learning_rate:.2e}",
            )
    teacher_scores = np.asarray(incoming_teacher_scores, dtype=np.float64)
    return {
        "losses": losses,
        "ad_losses": ad_losses,
        "distillation_losses": distillation_losses,
        "gradient_cosines": gradient_cosines,
        "zero_gradient_audit_batches": zero_gradient_audit_batches,
        "incoming_teacher_scores": teacher_scores,
        "incoming_teacher_score_mean": (
            float(teacher_scores.mean()) if teacher_scores.size else float("nan")
        ),
        "incoming_teacher_fpr_at_previous_threshold": (
            float((teacher_scores > teacher_threshold).mean())
            if teacher_scores.size
            else float("nan")
        ),
        "incoming_teacher_mean_score_over_previous_threshold": (
            float(teacher_scores.mean() / teacher_threshold)
            if teacher_scores.size and teacher_threshold > 0
            else float("nan")
        ),
    }


def estimate_fisher(model, loader, config, device):
    model.train()
    model.encoder.eval()
    parameters = expert_named_parameters(model)
    fisher = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    batches = 0
    for batch_index, (images, _) in enumerate(tqdm(loader, desc="fisher", ncols=90)):
        if config.fisher_batches > 0 and batch_index >= config.fisher_batches:
            break
        images = images.to(device, non_blocking=True)
        encoded, decoded, gather_loss = model(images)
        loss = global_cosine_hm_adaptive(encoded, decoded, y=3) + 0.2 * gather_loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                fisher[name].add_(parameter.grad.detach().square())
        batches += 1
    if batches == 0:
        raise RuntimeError("No batches were available for Fisher estimation.")
    return {
        name: value.div_(batches).cpu()
        for name, value in fisher.items()
    }, batches


def update_online_ewc(model, previous_state, current_fisher):
    parameters = expert_named_parameters(model)
    updated = {}
    for name, parameter in parameters.items():
        fisher = current_fisher[name]
        if previous_state:
            fisher = fisher + previous_state[name][1]
        updated[name] = (parameter.detach().cpu().clone(), fisher)
    return updated


def move_ewc_state(ewc_state, device):
    if not ewc_state:
        return None
    return {
        name: (anchor.to(device), fisher.to(device))
        for name, (anchor, fisher) in ewc_state.items()
    }


@torch.no_grad()
def routing_features(encoder, images):
    tokens = encoder.prepare_tokens(images)
    for block in encoder.blocks:
        tokens = block(tokens)
    register_count = getattr(encoder, "num_register_tokens", 0)
    patches = tokens[:, 1 + register_count :, :]
    return F.normalize(patches.mean(dim=1), dim=1)


@torch.no_grad()
def compute_centroid(encoder, loader, device, max_batches=0):
    features = []
    encoder.eval()
    for batch_index, (images, _) in enumerate(tqdm(loader, desc="centroid", ncols=90)):
        if max_batches > 0 and batch_index >= max_batches:
            break
        features.append(routing_features(encoder, images.to(device, non_blocking=True)).cpu())
    if not features:
        raise RuntimeError("No routing features were collected.")
    return F.normalize(torch.cat(features).mean(dim=0), dim=0).cpu()


def route_by_centroid(features, centroids):
    matrix = torch.stack(centroids).to(features.device)
    return torch.argmax(features @ matrix.T, dim=1)


@torch.no_grad()
def compute_multi_prototypes(encoder, loader, device, clusters, seed, max_batches=0):
    features = []
    encoder.eval()
    for batch_index, (images, _) in enumerate(
        tqdm(loader, desc=f"router k={clusters}", ncols=90)
    ):
        if max_batches > 0 and batch_index >= max_batches:
            break
        features.append(routing_features(encoder, images.to(device, non_blocking=True)).cpu())
    if not features:
        raise RuntimeError("No routing features were collected.")
    feature_array = torch.cat(features).numpy()
    if feature_array.shape[0] < clusters:
        raise RuntimeError(
            f"Router requires at least {clusters} samples, got {feature_array.shape[0]}."
        )
    centers = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit(
        feature_array
    ).cluster_centers_
    return F.normalize(torch.from_numpy(centers), dim=1)


def route_by_multi_prototype(features, task_prototypes):
    prototype_matrix = torch.cat(task_prototypes).to(features.device)
    prototype_tasks = torch.cat(
        [
            torch.full((len(prototypes),), task, dtype=torch.long)
            for task, prototypes in enumerate(task_prototypes)
        ]
    ).to(features.device)
    nearest = torch.argmax(features @ prototype_matrix.T, dim=1)
    return prototype_tasks[nearest]


def anomaly_maps_from_features(encoded, decoded, image_size, resize_mask, gaussian):
    maps, _ = cal_anomaly_maps(encoded, decoded, image_size)
    if resize_mask > 0:
        maps = F.interpolate(maps, size=resize_mask, mode="bilinear", align_corners=False)
    return gaussian(maps)


@torch.no_grad()
def anomaly_maps(model, images, resize_mask, gaussian):
    encoded, decoded, _ = model(images)
    return anomaly_maps_from_features(
        encoded,
        decoded,
        images.shape[-1],
        resize_mask,
        gaussian,
    )


def image_scores(maps, max_ratio=0.01):
    flat = maps.flatten(1)
    count = max(1, int(flat.shape[1] * max_ratio))
    return torch.topk(flat, k=count, dim=1).values.mean(dim=1)


def safe_metric(function, labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(function(labels, scores))


def compute_metrics(labels, scores, masks, maps):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    return {
        "I-AUROC": safe_metric(roc_auc_score, labels, scores),
        "I-AP": safe_metric(average_precision_score, labels, scores),
        "P-AUROC": safe_metric(roc_auc_score, masks.reshape(-1), maps.reshape(-1)),
        "P-AP": safe_metric(average_precision_score, masks.reshape(-1), maps.reshape(-1)),
    }


def validate_bundle(bundle, expected_categories=None, require_label_coverage=True):
    keys = ("paths", "categories", "labels", "scores", "routes", "masks", "maps")
    lengths = {key: len(bundle[key]) for key in keys}
    if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
        raise RuntimeError(f"Invalid prediction bundle lengths: {lengths}")

    paths = list(bundle["paths"])
    if len(paths) != len(set(paths)):
        raise RuntimeError("Prediction bundle contains duplicate image paths.")

    labels = np.asarray(bundle["labels"], dtype=np.int64)
    scores = np.asarray(bundle["scores"], dtype=np.float64)
    maps = np.asarray(bundle["maps"], dtype=np.float32)
    masks = np.asarray(bundle["masks"], dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Prediction bundle contains labels outside {0, 1}.")
    if not np.isfinite(scores).all() or not np.isfinite(maps).all():
        raise RuntimeError("Prediction bundle contains non-finite scores or anomaly maps.")
    if maps.shape != masks.shape:
        raise RuntimeError(f"Prediction map/mask shape mismatch: {maps.shape} != {masks.shape}")

    categories = np.asarray(bundle["categories"])
    observed_categories = set(categories.tolist())
    if expected_categories is not None and observed_categories != set(expected_categories):
        raise RuntimeError(
            f"Test category mismatch: observed={sorted(observed_categories)}, "
            f"expected={sorted(expected_categories)}"
        )
    if require_label_coverage:
        for category in sorted(observed_categories):
            category_labels = labels[categories == category]
            if set(np.unique(category_labels).tolist()) != {0, 1}:
                raise RuntimeError(
                    f"Category {category} does not contain both normal and anomalous test samples."
                )


def threshold_rates(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(scores) > threshold
    normal = labels == 0
    anomaly = labels == 1
    fpr = float(predictions[normal].mean()) if normal.any() else float("nan")
    fnr = float((~predictions[anomaly]).mean()) if anomaly.any() else float("nan")
    return fpr, fnr


def save_prediction_bundle(path, bundle, save_pixel_maps):
    arrays = {
        "paths": np.asarray(bundle["paths"]),
        "categories": np.asarray(bundle["categories"]),
        "labels": np.asarray(bundle["labels"], dtype=np.int8),
        "scores": np.asarray(bundle["scores"], dtype=np.float32),
        "routes": np.asarray(bundle["routes"], dtype=np.int16),
    }
    if save_pixel_maps:
        arrays["masks"] = np.asarray(bundle["masks"], dtype=np.uint8)
        arrays["maps"] = np.asarray(bundle["maps"], dtype=np.float16)
    np.savez_compressed(path, **arrays)


def save_reference_bundle(path, bundle):
    np.savez_compressed(
        path,
        paths=np.asarray(bundle["paths"]),
        categories=np.asarray(bundle["categories"]),
        labels=np.asarray(bundle["labels"], dtype=np.int8),
        scores=np.asarray(bundle["scores"], dtype=np.float64),
        routes=np.asarray(bundle["routes"], dtype=np.int16),
        masks=np.asarray(bundle["masks"], dtype=np.uint8),
        maps=np.asarray(bundle["maps"], dtype=np.float32),
    )


def load_reference_bundle(path):
    with np.load(path) as bundle:
        return {
            "paths": bundle["paths"].tolist(),
            "categories": bundle["categories"].tolist(),
            "labels": bundle["labels"].tolist(),
            "scores": bundle["scores"].copy(),
            "routes": bundle["routes"].copy(),
            "masks": bundle["masks"].copy(),
            "maps": bundle["maps"].copy(),
        }


def routing_only_bundle(bundle):
    return {
        "paths": list(bundle["paths"]),
        "labels": list(bundle["labels"]),
        "routes": list(bundle["routes"]),
    }


def evaluate_task(
    model,
    loader,
    device,
    config,
    true_task,
    route_centroids=None,
    route_prototypes=None,
    expert_states=None,
    expert_state_loader=load_expert_state,
):
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    bundle = {key: [] for key in ("paths", "categories", "labels", "scores", "routes", "masks", "maps")}
    model.eval()
    for batch_index, (images, masks, labels, paths, categories) in enumerate(
        tqdm(loader, desc=f"eval task {true_task}", ncols=90)
    ):
        if config.max_eval_batches > 0 and batch_index >= config.max_eval_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device)
        if config.resize_mask > 0:
            masks = F.interpolate(masks, size=config.resize_mask, mode="nearest")
        masks = (masks > 0.5).to(torch.uint8)

        if route_centroids is None and route_prototypes is None:
            routes = torch.full((images.shape[0],), true_task, device=device, dtype=torch.long)
            maps = anomaly_maps(model, images, config.resize_mask, gaussian)
        else:
            features = routing_features(model.encoder, images)
            routes = (
                route_by_centroid(features, route_centroids)
                if route_prototypes is None
                else route_by_multi_prototype(features, route_prototypes)
            )
            maps = torch.empty(
                (images.shape[0], 1, masks.shape[-2], masks.shape[-1]),
                device=device,
                dtype=images.dtype,
            )
            for expert_index in routes.unique().tolist():
                selected = routes == expert_index
                expert_state_loader(model, expert_states[expert_index])
                maps[selected] = anomaly_maps(model, images[selected], config.resize_mask, gaussian)

        scores = image_scores(maps)
        bundle["paths"].extend(paths)
        bundle["categories"].extend(categories)
        bundle["labels"].extend(labels.cpu().numpy().tolist())
        bundle["scores"].extend(scores.cpu().numpy().tolist())
        bundle["routes"].extend(routes.cpu().numpy().tolist())
        bundle["masks"].extend(masks[:, 0].cpu().numpy())
        bundle["maps"].extend(maps[:, 0].cpu().numpy())
    return bundle


def category_from_path(path, categories):
    path_parts = set(Path(path).parts)
    matches = [category for category in categories if category in path_parts]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not identify exactly one category for validation path {path}; "
            f"matches={matches}"
        )
    return matches[0]


def evaluate_validation_normals(
    model,
    loader,
    device,
    config,
    true_task,
    categories,
):
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    bundle = {
        key: []
        for key in ("paths", "categories", "labels", "scores", "routes", "masks", "maps")
    }
    model.eval()
    with torch.no_grad():
        for batch_index, (images, paths) in enumerate(
            tqdm(loader, desc=f"validation task {true_task}", ncols=90)
        ):
            if config.max_eval_batches > 0 and batch_index >= config.max_eval_batches:
                break
            images = images.to(device, non_blocking=True)
            maps = anomaly_maps(model, images, config.resize_mask, gaussian)
            scores = image_scores(maps)
            routes = torch.full(
                (images.shape[0],),
                true_task,
                device=device,
                dtype=torch.long,
            )
            bundle["paths"].extend(paths)
            bundle["categories"].extend(
                category_from_path(path, categories) for path in paths
            )
            bundle["labels"].extend([0] * images.shape[0])
            bundle["scores"].extend(scores.cpu().numpy().tolist())
            bundle["routes"].extend(routes.cpu().numpy().tolist())
            bundle["masks"].extend(
                np.zeros(
                    (images.shape[0], maps.shape[-2], maps.shape[-1]),
                    dtype=np.uint8,
                )
            )
            bundle["maps"].extend(maps[:, 0].cpu().numpy())
    return bundle


def summarize_validation_bundle(bundle, threshold):
    scores = np.asarray(bundle["scores"], dtype=np.float64)
    if not scores.size:
        raise RuntimeError("Cannot summarize an empty validation bundle.")
    return {
        "validation_images": int(scores.size),
        "validation_score_mean": float(scores.mean()),
        "validation_score_std": float(scores.std()),
        "validation_score_q95": float(np.quantile(scores, 0.95)),
        "validation_score_q99": float(np.quantile(scores, 0.99)),
        "validation_score_max": float(scores.max()),
        "FPR@fixed": float((scores > threshold).mean()),
        "mean_score_over_fixed_threshold": (
            float(scores.mean() / threshold) if threshold > 0 else float("nan")
        ),
        "max_score_over_fixed_threshold": (
            float(scores.max() / threshold) if threshold > 0 else float("nan")
        ),
    }


def summarize_bundle(bundle, threshold, true_task):
    metrics = compute_metrics(bundle["labels"], bundle["scores"], bundle["masks"], bundle["maps"])
    fpr, fnr = threshold_rates(bundle["labels"], bundle["scores"], threshold)
    labels = np.asarray(bundle["labels"])
    routes = np.asarray(bundle["routes"])
    correct = routes == true_task
    anomaly = labels == 1
    wrong_anomaly = anomaly & (~correct)
    scores = np.asarray(bundle["scores"])
    metrics.update(
        {
            "FPR@fixed": fpr,
            "FNR@fixed": fnr,
            "Route-Acc": float(correct.mean()),
            "Route-Acc-Normal": float(correct[labels == 0].mean()) if (labels == 0).any() else float("nan"),
            "Route-Acc-Anomaly": float(correct[labels == 1].mean()) if (labels == 1).any() else float("nan"),
            "Wrong-Route-Anomaly-FNR": (
                float((scores[wrong_anomaly] <= threshold).mean())
                if wrong_anomaly.any()
                else float("nan")
            ),
        }
    )
    return metrics


def summarize_categories(bundle, threshold, true_task):
    categories = np.asarray(bundle["categories"])
    rows = []
    for category in sorted(set(categories.tolist())):
        selected = categories == category
        subset = {
            key: np.asarray(value)[selected]
            for key, value in bundle.items()
        }
        rows.append({"category": category, **summarize_bundle(subset, threshold, true_task)})
    return rows


def compare_bundles(reference, current):
    if reference["paths"] != current["paths"]:
        raise RuntimeError("Prediction order changed; drift comparison is invalid.")
    if len(reference["paths"]) != len(set(reference["paths"])):
        raise RuntimeError("Reference predictions contain duplicate paths.")
    for key in ("categories", "labels"):
        if list(reference[key]) != list(current[key]):
            raise RuntimeError(f"Prediction {key} changed; drift comparison is invalid.")
    ref_scores = np.asarray(reference["scores"], dtype=np.float64)
    cur_scores = np.asarray(current["scores"], dtype=np.float64)
    ref_maps = np.asarray(reference["maps"], dtype=np.float32)
    cur_maps = np.asarray(current["maps"], dtype=np.float32)
    if not all(np.isfinite(array).all() for array in (ref_scores, cur_scores, ref_maps, cur_maps)):
        raise RuntimeError("Cannot compare bundles containing non-finite predictions.")
    score_diff = np.abs(cur_scores - ref_scores)
    map_diff = np.abs(cur_maps - ref_maps)
    normal = np.asarray(reference["labels"]) == 0
    anomaly = ~normal
    denominator = float(ref_scores.std() + 1e-12)
    return {
        "max_abs_score_diff": float(score_diff.max(initial=0.0)),
        "mean_abs_score_diff": float(score_diff.mean()),
        "max_abs_pixel_map_diff": float(map_diff.max(initial=0.0)),
        "mean_abs_pixel_map_diff": float(map_diff.mean()),
        "normalized_score_drift": float(score_diff.mean() / denominator),
        "normal_score_drift": float(score_diff[normal].mean()) if normal.any() else float("nan"),
        "anomaly_score_drift": float(score_diff[anomaly].mean()) if anomaly.any() else float("nan"),
    }


def route_flip_metrics(reference, current):
    if list(reference["paths"]) != list(current["paths"]):
        raise RuntimeError("Prediction order changed; route comparison is invalid.")
    if list(reference["labels"]) != list(current["labels"]):
        raise RuntimeError("Prediction labels changed; route comparison is invalid.")
    reference_routes = np.asarray(reference["routes"], dtype=np.int64)
    current_routes = np.asarray(current["routes"], dtype=np.int64)
    labels = np.asarray(reference["labels"], dtype=np.int64)
    flipped = reference_routes != current_routes
    normal = labels == 0
    anomaly = labels == 1
    return {
        "route_flip_rate": float(flipped.mean()),
        "route_flip_rate_normal": float(flipped[normal].mean()) if normal.any() else float("nan"),
        "route_flip_rate_anomaly": float(flipped[anomaly].mean()) if anomaly.any() else float("nan"),
    }


def write_stage_csv(path, rows):
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_run_notes(path, config, stream, rows, gate):
    lines = [
        f"# {Path(config.output_dir).name}",
        "",
        "## Configuration",
        "",
        f"- Method: `{config.method}`",
        f"- Stream: `{stream['name']}`",
        f"- Seed: `{config.seed}`",
        f"- Epochs/task: `{config.epochs}`",
        f"- Train batch cap: `{config.max_train_batches or 'none'}`",
        f"- Eval batch cap: `{config.max_eval_batches or 'none'}`",
        f"- Compact audit storage: `{config.compact_audit_storage}`",
        "",
        "## Status",
        "",
    ]
    if config.max_train_batches or config.max_eval_batches or config.epochs < 50:
        lines.append("- This is a smoke/debug run and must not be cited as a formal paper result.")
    else:
        lines.append("- Formal run candidate; inspect consistency checks before using in the paper.")
    if gate is not None:
        lines.append(
            f"- Oracle retention gate: `{'PASS' if gate else 'FAIL'}` "
            "(max score/map difference threshold 1e-6)."
        )
    lines.extend(["", "## Stage Metrics", ""])
    for row in rows:
        lines.append(
            f"- Stage {row['stage']}, task {row['eval_task']}, route `{row['route']}`: "
            f"I-AUROC={row['I-AUROC']:.4f}, P-AUROC={row['P-AUROC']:.4f}, "
            f"route_acc={row['Route-Acc']:.4f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def runtime_metadata(config, stream_path, script_path):
    metadata = {
        "started_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "command": sys.argv,
        "working_directory": os.getcwd(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "stream_sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest(),
        "formal_candidate_conditions": {
            "epochs_at_least_50": config.epochs >= 50,
            "all_tasks": config.max_tasks == 0,
            "uncapped_training": config.max_train_batches == 0,
            "uncapped_evaluation": config.max_eval_batches == 0,
            "pixel_maps_saved": config.save_pixel_maps,
            "official_test_evaluation": not config.validation_audit_only,
        },
        "compact_audit_storage": config.compact_audit_storage,
        "validation_audit_only": config.validation_audit_only,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    metadata["formal_candidate"] = all(metadata["formal_candidate_conditions"].values())
    return metadata


def score_normal_files(model, files, transform, device, config):
    """Score a list of normal image paths with the CURRENT model state (no_grad) -> {path: score}.

    This is the Goal C score-anchor signal. At replay-build time the model still holds the
    *previous* stage's trained state (the state about to be overwritten by the next task), so
    scoring prior normals here measures how far each normal sits from the normal-memory decision
    boundary of a memory that is about to be lost. Higher score = harder normal = more valuable to
    keep under a limited replay budget.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    model.eval()
    scores = {}
    if not files:
        return scores
    loader = DataLoader(
        NormalFileDataset(files, transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            maps = anomaly_maps(model, images, config.resize_mask, gaussian)
            sc = image_scores(maps)
            for path, s in zip(paths, sc.cpu().numpy().tolist()):
                scores[str(path)] = float(s)
    return scores


def _anchor_select_replay(model, stored_category_train_files, transform, device, config):
    """Select top-K prior normals per category by anchor score (descending), same per-class budget
    as the random path (config.replay_buffer_size). Returns (replay_files, anchor_scores)."""
    all_prior = [p for fs in stored_category_train_files.values() for p in fs]
    anchor_scores = score_normal_files(model, all_prior, transform, device, config)
    replay_files = []
    for prior_category, prior_files in stored_category_train_files.items():
        if len(prior_files) <= config.replay_buffer_size:
            replay_files.extend(prior_files)
        else:
            ranked = sorted(
                prior_files,
                key=lambda p: anchor_scores.get(str(p), float("inf")),
                reverse=True,
            )
            replay_files.extend(ranked[: config.replay_buffer_size])
    return replay_files, anchor_scores


def select_replay_files(stored_category_train_files, buffer_size, rng):
    """Per-class replay budget selection for the 'random' rule.

    For each prior-normal category, keep every image if the category holds at most
    ``buffer_size`` files, otherwise uniformly sample ``buffer_size`` of them using
    ``rng`` (an already-seeded ``random.Random``). Returns the flat list of replay paths.
    """
    replay_files = []
    for prior_files in stored_category_train_files.values():
        if len(prior_files) <= buffer_size:
            replay_files.extend(prior_files)
        else:
            replay_files.extend(rng.sample(prior_files, buffer_size))
    return replay_files


def run(config):
    setup_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for this experiment.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = Path(config.stream)
    metadata = runtime_metadata(config, stream_path, Path(__file__).resolve())
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    stream, tasks = load_stream(config.stream, config.max_tasks)
    dataset_name = stream.get("dataset", "MVTec-AD")
    with open(output_dir / "stream.json", "w", encoding="utf-8") as handle:
        json.dump(stream, handle, indent=2)

    transform, gt_transform = get_data_transforms(config.input_size, config.crop_size)
    encoder = build_encoder(config.encoder, device)
    model, trainable = build_expert(encoder, config.encoder, config.inp_num, device)
    teacher_model = None
    subspace_layers = {}
    subspace_history = None
    if config.method == "lwf":
        with torch.random.fork_rng(devices=[device.index or 0]):
            teacher_model, _ = build_expert(
                encoder,
                config.encoder,
                config.inp_num,
                device,
                initialize=False,
            )
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
    lora_modules = []
    if config.method in LORA_METHODS:
        trainable, lora_modules = configure_shared_lora(
            model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    elif config.method in SUBSPACE_METHODS:
        for parameter in model.decoder.parameters():
            parameter.requires_grad = False
        subspace_layers = inject_cumulative_subspace_lora(
            model.decoder,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            total_tasks=len(tasks),
            dropout=config.lora_dropout,
            target_filter=decoder_attention_target,
            prefix="decoder",
        )
        if not subspace_layers:
            raise RuntimeError("No Decoder attention projections were selected.")
        lora_modules = sorted(subspace_layers)
        subspace_history = DualGPMState(
            total_tasks=len(tasks),
            lambda_start=config.subspace_lambda_start,
            lambda_end=config.subspace_lambda_end,
        )
        trainable = configure_frozen_decoder_head(model)
    elif config.method == "isolated_head":
        trainable = configure_frozen_decoder_head(model)
    encoder_parameters, expert_parameters = count_parameters(model)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    )
    if config.method in SUBSPACE_METHODS:
        trainable_parameters += sum(
            layer.lora_B[0].numel() for layer in subspace_layers.values()
        )
    if config.method in COMPACT_ISOLATED_METHODS:
        task_specific_state = task_specific_lora_state_dict(model)
    elif config.method in SUBSPACE_METHODS:
        task_specific_state = task_specific_head_state_dict(model)
    else:
        task_specific_state = {}
    task_specific_parameters = sum(value.numel() for value in task_specific_state.values())
    task_specific_non_lora_parameters = sum(
        value.numel()
        for name, value in task_specific_state.items()
        if ".lora_A" not in name and ".lora_B" not in name
    )
    frozen_decoder_parameters = sum(
        value.numel() for value in frozen_decoder_state_dict(model).values()
    )
    parameter_report = {
        "encoder_parameters": encoder_parameters,
        "expert_parameters_per_task": expert_parameters,
        "expert_size_fp32_mb": expert_parameters * 4 / 1024**2,
        "trainable_parameters": trainable_parameters,
        "trainable_size_fp32_mb": trainable_parameters * 4 / 1024**2,
        "lora_parameters": lora_parameters,
        "lora_size_fp32_mb": lora_parameters * 4 / 1024**2,
        "lora_parameter_scope": (
            "all preallocated task branches"
            if config.method in SUBSPACE_METHODS
            else "all injected LoRA parameters"
        ),
        "lora_linear_modules": len(lora_modules),
        "subspace_lora_parameters_per_task": (
            sum(
                layer.lora_A[0].numel() + layer.lora_B[0].numel()
                for layer in subspace_layers.values()
            )
            if subspace_layers
            else 0
        ),
        "subspace_trainable_B_parameters_per_stage": (
            sum(layer.lora_B[0].numel() for layer in subspace_layers.values())
            if subspace_layers
            else 0
        ),
        "task_specific_parameters_per_task": task_specific_parameters,
        "task_specific_size_fp32_mb_per_task": task_specific_parameters * 4 / 1024**2,
        "task_specific_non_lora_parameters_per_task": task_specific_non_lora_parameters,
        "task_specific_non_lora_size_fp32_mb_per_task": (
            task_specific_non_lora_parameters * 4 / 1024**2
        ),
        "frozen_shared_decoder_parameters": frozen_decoder_parameters,
        "full_isolated_expert_parameters_per_task": (
            expert_parameters - lora_parameters
            if config.method in ALL_LORA_METHODS
            else expert_parameters
        ),
        "ewc_state_size_fp32_mb": (
            2 * expert_parameters * 4 / 1024**2 if config.method == "ewc" else 0.0
        ),
        "centroid_routing_bytes_per_task": int(getattr(encoder, "embed_dim", 0) * 4),
        "k3_routing_bytes_per_task": int(getattr(encoder, "embed_dim", 0) * 3 * 4),
        "threshold_calibration_bytes_per_task": 8,
    }
    with open(output_dir / "parameters.json", "w", encoding="utf-8") as handle:
        json.dump(parameter_report, handle, indent=2)

    if config.method in COMPACT_ISOLATED_METHODS:
        state_getter = task_specific_lora_state_dict
        state_loader = load_task_specific_lora_state
    elif config.method in SUBSPACE_METHODS:
        state_getter = task_specific_head_state_dict
        state_loader = load_task_specific_head_state
    else:
        state_getter = expert_state_dict
        state_loader = load_expert_state
    initial_state = state_getter(model)
    shared_state = copy.deepcopy(initial_state)
    expert_states = []
    expert_hashes = []
    centroids = []
    router_prototypes = []
    thresholds = []
    threshold_hashes = []
    task_validation_files = []
    reference_bundles = {}
    previous_stage_bundles = {}
    routing_reference_bundles = {}
    validation_reference_bundles = {}
    validation_previous_stage_bundles = {}
    reference_dir = output_dir / "references"
    previous_stage_reference_dir = output_dir / "previous_stage_references"
    validation_reference_dir = output_dir / "validation_references"
    validation_previous_stage_reference_dir = (
        output_dir / "validation_previous_stage_references"
    )
    if config.compact_audit_storage:
        reference_dir.mkdir(exist_ok=True)
        previous_stage_reference_dir.mkdir(exist_ok=True)
        validation_reference_dir.mkdir(exist_ok=True)
        validation_previous_stage_reference_dir.mkdir(exist_ok=True)
    metric_rows = []
    category_metric_rows = []
    drift_rows = []
    incremental_drift_rows = []
    routing_drift_rows = []
    validation_metric_rows = []
    validation_drift_rows = []
    validation_incremental_drift_rows = []
    invariant_rows = []
    gate_pass = True
    ewc_state = None
    frozen_decoder_hash = (
        state_dict_sha256(frozen_decoder_state_dict(model))
        if config.method in FROZEN_DECODER_METHODS
        else None
    )
    shared_frozen_decoder_path = None
    if config.method in SUBSPACE_METHODS:
        shared_frozen_decoder_path = output_dir / "shared_frozen_decoder.pth"
        torch.save(frozen_decoder_state_dict(model), shared_frozen_decoder_path)
        parameter_report.update(
            {
                "shared_frozen_decoder_checkpoint_bytes": (
                    shared_frozen_decoder_path.stat().st_size
                ),
                "subspace_target_modules": sorted(subspace_layers),
                "subspace_basis_bytes_initial": 0,
            }
        )
        with open(output_dir / "parameters.json", "w", encoding="utf-8") as handle:
            json.dump(parameter_report, handle, indent=2)
    optimizer_audits = []
    stored_category_train_files = {}

    for stage, categories in enumerate(tasks):
        stage_dir = output_dir / f"stage_{stage + 1:02d}"
        stage_dir.mkdir(exist_ok=True)
        train_files, validation_files = split_normal_files(
            config.data_path,
            categories,
            config.validation_fraction,
            config.seed,
            dataset_name=dataset_name,
        )
        task_validation_files.append(validation_files)
        train_loader = DataLoader(
            NormalFileDataset(train_files, transform),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        validation_loader = DataLoader(
            NormalFileDataset(validation_files, transform),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # --- Experience replay of prior-stage normal categories (data anchor) ---
        replay_loader = None
        if config.replay_buffer_size > 0:
            replay_files = []
            if stored_category_train_files:
                anchor_scores = {}
                if config.replay_selector == "score_anchor":
                    replay_files, anchor_scores = _anchor_select_replay(
                        model, stored_category_train_files, transform, device, config
                    )
                    print(
                        f"[replay] stage={stage + 1} SELECTOR=score_anchor "
                        f"scored={len(anchor_scores)} prior normals "
                        f"buffer={config.replay_buffer_size} selected={len(replay_files)}",
                        flush=True,
                    )
                else:
                    buffer_rng = random.Random(
                        int(
                            hashlib.sha256(
                                f"replay:{config.seed}:{config.replay_buffer_size}".encode()
                            ).hexdigest()[:8],
                            16,
                        )
                    )
                    replay_files.extend(
                        select_replay_files(
                            stored_category_train_files,
                            config.replay_buffer_size,
                            buffer_rng,
                        )
                    )
            if len(replay_files) >= config.batch_size:
                replay_loader = DataLoader(
                    NormalFileDataset(replay_files, transform),
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    drop_last=True,
                    pin_memory=True,
                )
                print(
                    f"[replay] stage={stage + 1} buffer_size_per_class="
                    f"{config.replay_buffer_size} total_replay_files={len(replay_files)}",
                    flush=True,
                )
            elif config.replay_buffer_size > 0:
                print(
                    f"[replay] stage={stage + 1} skipped: "
                    f"total_replay_files={len(replay_files)} < batch_size={config.batch_size}",
                    flush=True,
                )
        current_category_train_files = {}
        for category in categories:
            current_category_train_files[category], _ = _split_category(
                config.data_path,
                category,
                config.validation_fraction,
                config.seed,
                dataset_name,
            )
        stored_category_train_files.update(current_category_train_files)

        if config.method in SHARED_METHODS:
            load_expert_state(model, shared_state)
        elif config.method in ISOLATED_METHODS:
            source_state = expert_states[-1] if expert_states else initial_state
            state_loader(model, source_state)
        elif config.method in SUBSPACE_METHODS:
            source_state = expert_states[-1] if expert_states else initial_state
            state_loader(model, source_state)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        subspace_projection_report = {}
        subspace_history_report = {}
        subspace_pre_covariance_batches = 0
        subspace_post_covariance_batches = 0
        if config.method in SUBSPACE_METHODS:
            with preserve_rng_state():
                pre_covariances, subspace_pre_covariance_batches = (
                    collect_subspace_covariances(
                        model,
                        subspace_layers,
                        train_loader,
                        config,
                        device,
                    )
                )
            subspace_projection_report = configure_cumulative_task(
                subspace_layers,
                task=stage,
                covariances=pre_covariances,
                history=subspace_history,
            )
            del pre_covariances
            trainable = configure_subspace_trainable(model, subspace_layers)
            optimizer_audit = audit_subspace_trainable_boundary(
                model, subspace_layers, stage, trainable
            )
        elif config.method in COMPACT_ISOLATED_METHODS:
            optimizer_audit = audit_lora_trainable_boundary(model, trainable)
        else:
            optimizer_audit = None
        incoming_validation_teacher_bundle = None
        if config.method == "lwf" and stage > 0:
            load_expert_state(teacher_model, shared_state)
            with preserve_rng_state():
                incoming_validation_teacher_bundle = evaluate_validation_normals(
                    teacher_model,
                    validation_loader,
                    device,
                    config,
                    stage,
                    categories,
                )
            validate_bundle(
                incoming_validation_teacher_bundle,
                expected_categories=categories if config.max_eval_batches == 0 else None,
                require_label_coverage=False,
            )
            training_record = train_expert_lwf(
                model,
                teacher_model,
                trainable,
                train_loader,
                config,
                device,
                teacher_threshold=thresholds[-1],
            )
        else:
            device_ewc_state = move_ewc_state(ewc_state, device)
            losses = train_expert(
                model,
                trainable,
                train_loader,
                config,
                device,
                ewc_state=device_ewc_state,
                replay_loader=replay_loader,
            )
            del device_ewc_state
            training_record = {
                "losses": losses,
                "ad_losses": losses,
                "distillation_losses": [],
                "gradient_cosines": [],
                "zero_gradient_audit_batches": 0,
                "incoming_teacher_scores": np.asarray([], dtype=np.float64),
                "incoming_teacher_score_mean": float("nan"),
                "incoming_teacher_fpr_at_previous_threshold": float("nan"),
                "incoming_teacher_mean_score_over_previous_threshold": float("nan"),
            }
        subspace_update_norms = {}
        if config.method in SUBSPACE_METHODS:
            with preserve_rng_state():
                post_covariances, subspace_post_covariance_batches = (
                    collect_subspace_covariances(
                        model,
                        subspace_layers,
                        train_loader,
                        config,
                        device,
                    )
                )
            subspace_history_report = subspace_history.update(stage, post_covariances)
            del post_covariances
            subspace_update_norms = cumulative_update_report(
                subspace_layers, task=stage
            )
        training_peak_memory_bytes = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        trained_state = state_getter(model)
        if config.method in SHARED_METHODS:
            shared_state = trained_state
        else:
            expert_states.append(trained_state)
            expert_hashes.append(state_dict_sha256(trained_state))
        if config.method in COMPACT_ISOLATED_METHODS + SUBSPACE_METHODS:
            current_decoder_hash = state_dict_sha256(frozen_decoder_state_dict(model))
            optimizer_audit.update(
                {
                    "stage": stage + 1,
                    "old_task_snapshots": len(expert_states) - 1,
                    "old_task_snapshots_are_cpu_tensors": all(
                        value.device.type == "cpu"
                        for state in expert_states[:-1]
                        for value in state.values()
                    ),
                    "frozen_decoder_sha256": current_decoder_hash,
                    "frozen_decoder_hash_match": current_decoder_hash
                    == frozen_decoder_hash,
                }
            )
            optimizer_audits.append(optimizer_audit)
            if config.method in COMPACT_ISOLATED_METHODS:
                gate_pass = gate_pass and optimizer_audit["frozen_decoder_hash_match"]
                gate_pass = gate_pass and not optimizer_audit[
                    "optimizer_contains_decoder_base"
                ]
            elif not optimizer_audit["frozen_decoder_hash_match"]:
                raise RuntimeError("Subspace LoRA changed the frozen Decoder base.")

        fisher_batches = 0
        if config.method == "ewc":
            current_fisher, fisher_batches = estimate_fisher(model, train_loader, config, device)
            ewc_state = update_online_ewc(model, ewc_state, current_fisher)
            if config.save_regularizer_state:
                torch.save(ewc_state, stage_dir / "ewc_state.pth")

        centroid = compute_centroid(
            encoder, train_loader, device, max_batches=config.max_train_batches
        )
        centroids.append(centroid)
        if config.method in K3_ROUTED_METHODS:
            prototypes = compute_multi_prototypes(
                encoder,
                train_loader,
                device,
                clusters=3,
                seed=config.seed,
                max_batches=config.max_train_batches,
            )
            router_prototypes.append(prototypes)

        validation_scores = []
        model.eval()
        gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        with torch.no_grad():
            for batch_index, (images, _) in enumerate(validation_loader):
                if config.max_eval_batches > 0 and batch_index >= config.max_eval_batches:
                    break
                maps = anomaly_maps(model, images.to(device), config.resize_mask, gaussian)
                validation_scores.extend(image_scores(maps).cpu().numpy().tolist())
        threshold = float(np.quantile(validation_scores, config.threshold_quantile))
        thresholds.append(threshold)
        threshold_hashes.append(scalar_sha256(threshold))
        incoming_validation_teacher_summary = (
            summarize_validation_bundle(
                incoming_validation_teacher_bundle,
                thresholds[-2],
            )
            if incoming_validation_teacher_bundle is not None
            else {}
        )

        if config.method in SUBSPACE_METHODS:
            checkpoint_state = subspace_checkpoint_state(
                model,
                subspace_layers,
                subspace_history,
                stage,
                config,
                frozen_decoder_hash,
                expert_states,
                thresholds,
                router_prototypes,
                tasks[: stage + 1],
            )
            checkpoint_hash = checkpoint_state["complete_state_sha256"]
        else:
            checkpoint_state = (
                shared_state
                if config.method in SHARED_METHODS
                else expert_states[-1]
            )
            checkpoint_hash = state_dict_sha256(checkpoint_state)
        checkpoint_path = stage_dir / "expert.pth"
        torch.save(checkpoint_state, checkpoint_path)
        task_head_path = None
        if config.method in SUBSPACE_METHODS:
            task_head_path = stage_dir / "task_head.pth"
            torch.save(expert_states[-1], task_head_path)
        if training_record["incoming_teacher_scores"].size:
            np.save(
                stage_dir / "incoming_teacher_scores.npy",
                training_record["incoming_teacher_scores"],
            )
        if incoming_validation_teacher_bundle is not None:
            np.save(
                stage_dir / "incoming_validation_teacher_scores.npy",
                np.asarray(
                    incoming_validation_teacher_bundle["scores"],
                    dtype=np.float64,
                ),
            )
        np.save(stage_dir / "centroid.npy", centroid.numpy())
        if config.method in K3_ROUTED_METHODS:
            np.save(stage_dir / "router_k3.npy", prototypes.numpy())
        complete_state_verification_ms = None
        if config.method in SUBSPACE_METHODS:
            verification_started = time.perf_counter()
            load_subspace_checkpoint(
                model,
                subspace_layers,
                subspace_history,
                checkpoint_state,
                frozen_decoder_hash,
                verify_integrity=True,
            )
            complete_state_verification_ms = (
                time.perf_counter() - verification_started
            ) * 1000
        load_started = time.perf_counter()
        for _ in range(10):
            if config.method in SUBSPACE_METHODS:
                load_subspace_checkpoint(
                    model,
                    subspace_layers,
                    subspace_history,
                    checkpoint_state,
                    frozen_decoder_hash,
                    verify_integrity=False,
                )
            else:
                state_loader(model, checkpoint_state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        mean_state_load_ms = (time.perf_counter() - load_started) * 1000 / 10
        with open(stage_dir / "train.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "categories": categories,
                    "train_images": len(train_files),
                    "validation_images": len(validation_files),
                    "train_paths_sha256": hash_strings(train_files),
                    "validation_paths_sha256": hash_strings(validation_files),
                    "threshold": threshold,
                    "expert_state_sha256": checkpoint_hash,
                    "threshold_sha256": threshold_hashes[-1],
                    "losses": training_record["losses"],
                    "ad_losses": training_record["ad_losses"],
                    "distillation_losses": training_record["distillation_losses"],
                    "lwf_lambda": config.lwf_lambda if config.method == "lwf" else None,
                    "lwf_gradient_cosines": training_record["gradient_cosines"],
                    "lwf_zero_gradient_audit_batches": training_record[
                        "zero_gradient_audit_batches"
                    ],
                    "incoming_teacher_score_mean": training_record[
                        "incoming_teacher_score_mean"
                    ],
                    "incoming_teacher_fpr_at_previous_threshold": training_record[
                        "incoming_teacher_fpr_at_previous_threshold"
                    ],
                    "incoming_teacher_mean_score_over_previous_threshold": training_record[
                        "incoming_teacher_mean_score_over_previous_threshold"
                    ],
                    "incoming_validation_teacher_score_mean": (
                        incoming_validation_teacher_summary.get(
                            "validation_score_mean",
                            float("nan"),
                        )
                    ),
                    "incoming_validation_teacher_fpr_at_previous_threshold": (
                        incoming_validation_teacher_summary.get(
                            "FPR@fixed",
                            float("nan"),
                        )
                    ),
                    "incoming_validation_teacher_mean_score_over_previous_threshold": (
                        incoming_validation_teacher_summary.get(
                            "mean_score_over_fixed_threshold",
                            float("nan"),
                        )
                    ),
                    "incoming_validation_teacher_paths_sha256": (
                        hash_strings(incoming_validation_teacher_bundle["paths"])
                        if incoming_validation_teacher_bundle is not None
                        else None
                    ),
                    "ewc_lambda": config.ewc_lambda if config.method == "ewc" else None,
                    "fisher_batches": fisher_batches,
                    "lora_rank": (
                        config.lora_rank
                        if config.method in ALL_LORA_METHODS
                        else None
                    ),
                    "lora_alpha": (
                        config.lora_alpha
                        if config.method in ALL_LORA_METHODS
                        else None
                    ),
                    "lora_dropout": (
                        config.lora_dropout
                        if config.method in ALL_LORA_METHODS
                        else None
                    ),
                    "optimizer_steps": len(training_record["losses"]),
                    "subspace_covariance_batches_pre_training": (
                        subspace_pre_covariance_batches
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "subspace_covariance_batches_post_training": (
                        subspace_post_covariance_batches
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "subspace_projection_report": (
                        subspace_projection_report
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "subspace_history_update_report": (
                        subspace_history_report
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "subspace_update_norms": (
                        subspace_update_norms
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "subspace_basis_bytes": (
                        subspace_history.storage_bytes_fp32()
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "complete_checkpoint_sha256": (
                        checkpoint_hash
                        if config.method in SUBSPACE_METHODS
                        else None
                    ),
                    "task_head_checkpoint_size_bytes": (
                        task_head_path.stat().st_size
                        if task_head_path is not None
                        else None
                    ),
                    "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                    "training_peak_memory_bytes": training_peak_memory_bytes,
                    "mean_state_load_ms": mean_state_load_ms,
                    "complete_state_verification_ms": (
                        complete_state_verification_ms
                    ),
                    "optimizer_audit": optimizer_audit,
                    "elapsed_seconds": time.time() - started,
                },
                handle,
                indent=2,
            )

        for eval_task in range(stage + 1):
            validation_eval_loader = DataLoader(
                NormalFileDataset(task_validation_files[eval_task], transform),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            if config.method in SHARED_METHODS:
                load_expert_state(model, shared_state)
                validation_route_name = {
                    "shared_ft": "shared",
                    "ewc": "ewc",
                    "shared_lora": "lora",
                    "lwf": "lwf",
                }[config.method]
            else:
                state_loader(model, expert_states[eval_task])
                validation_route_name = "oracle"
            validation_bundle = evaluate_validation_normals(
                model,
                validation_eval_loader,
                device,
                config,
                eval_task,
                tasks[eval_task],
            )
            validate_bundle(
                validation_bundle,
                expected_categories=(
                    tasks[eval_task] if config.max_eval_batches == 0 else None
                ),
                require_label_coverage=False,
            )
            validation_summary = summarize_validation_bundle(
                validation_bundle,
                thresholds[eval_task],
            )
            validation_row = {
                "stage": stage + 1,
                "eval_task": eval_task + 1,
                "route": validation_route_name,
                "is_current_task": eval_task == stage,
                "official_test_images_used": False,
                **validation_summary,
            }
            if eval_task == stage and incoming_validation_teacher_bundle is not None:
                teacher_mean = incoming_validation_teacher_summary[
                    "validation_score_mean"
                ]
                student_mean = validation_summary["validation_score_mean"]
                validation_row.update(
                    {
                        "incoming_teacher_score_mean": teacher_mean,
                        "current_score_reduction": teacher_mean - student_mean,
                        "current_relative_score_reduction": (
                            1.0 - student_mean / teacher_mean
                            if teacher_mean > 0
                            else float("nan")
                        ),
                        "incoming_teacher_fpr_at_previous_threshold": (
                            incoming_validation_teacher_summary["FPR@fixed"]
                        ),
                    }
                )
            validation_metric_rows.append(validation_row)
            save_prediction_bundle(
                stage_dir
                / f"task_{eval_task + 1:02d}_validation_{validation_route_name}.npz",
                validation_bundle,
                config.save_pixel_maps and not config.compact_audit_storage,
            )
            if eval_task in validation_previous_stage_bundles:
                previous_validation_bundle = (
                    load_reference_bundle(
                        validation_previous_stage_bundles[eval_task]
                    )
                    if config.compact_audit_storage
                    else validation_previous_stage_bundles[eval_task]
                )
                validation_incremental_drift_rows.append(
                    {
                        "stage": stage + 1,
                        "eval_task": eval_task + 1,
                        "route": validation_route_name,
                        "official_test_images_used": False,
                        **compare_bundles(
                            previous_validation_bundle,
                            validation_bundle,
                        ),
                    }
                )
                if config.compact_audit_storage:
                    del previous_validation_bundle
            if config.compact_audit_storage:
                previous_validation_path = (
                    validation_previous_stage_reference_dir
                    / f"task_{eval_task + 1:02d}_float32.npz"
                )
                save_reference_bundle(previous_validation_path, validation_bundle)
                validation_previous_stage_bundles[eval_task] = (
                    previous_validation_path
                )
            else:
                validation_previous_stage_bundles[eval_task] = validation_bundle
            if eval_task not in validation_reference_bundles:
                if config.compact_audit_storage:
                    validation_reference_path = (
                        validation_reference_dir
                        / f"task_{eval_task + 1:02d}_{validation_route_name}_float32.npz"
                    )
                    save_reference_bundle(
                        validation_reference_path,
                        validation_bundle,
                    )
                    validation_reference_bundles[eval_task] = (
                        validation_reference_path
                    )
                else:
                    validation_reference_bundles[eval_task] = validation_bundle
            else:
                validation_reference_bundle = (
                    load_reference_bundle(validation_reference_bundles[eval_task])
                    if config.compact_audit_storage
                    else validation_reference_bundles[eval_task]
                )
                validation_drift_rows.append(
                    {
                        "stage": stage + 1,
                        "eval_task": eval_task + 1,
                        "route": validation_route_name,
                        "official_test_images_used": False,
                        **compare_bundles(
                            validation_reference_bundle,
                            validation_bundle,
                        ),
                    }
                )
                if config.compact_audit_storage:
                    del validation_reference_bundle
            if config.compact_audit_storage:
                del validation_bundle

        for eval_task in (
            () if config.validation_audit_only else range(stage + 1)
        ):
            test_loader = DataLoader(
                GroupedTestDataset(
                    config.data_path,
                    tasks[eval_task],
                    transform,
                    gt_transform,
                    dataset_name=dataset_name,
                ),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            if config.method in SHARED_METHODS:
                load_expert_state(model, shared_state)
                bundle = evaluate_task(model, test_loader, device, config, eval_task)
                route_name = {
                    "shared_ft": "shared",
                    "ewc": "ewc",
                    "shared_lora": "lora",
                    "lwf": "lwf",
                }[config.method]
            else:
                state_loader(model, expert_states[eval_task])
                loaded_hash = state_dict_sha256(state_getter(model))
                bundle = evaluate_task(model, test_loader, device, config, eval_task)
                route_name = "oracle"
                invariant_rows.append(
                    {
                        "stage": stage + 1,
                        "eval_task": eval_task + 1,
                        "expert_state_sha256": expert_hashes[eval_task],
                        "loaded_expert_sha256": loaded_hash,
                        "expert_hash_match": loaded_hash == expert_hashes[eval_task],
                        "threshold_sha256": scalar_sha256(thresholds[eval_task]),
                        "threshold_hash_match": (
                            scalar_sha256(thresholds[eval_task])
                            == threshold_hashes[eval_task]
                        ),
                        "latest_cumulative_lora_sha256": (
                            state_dict_sha256(
                                cumulative_lora_state_dict(
                                    subspace_layers, task=stage
                                )
                            )
                            if config.method in SUBSPACE_METHODS
                            else None
                        ),
                    }
                )
                gate_pass = gate_pass and loaded_hash == expert_hashes[eval_task]
                gate_pass = gate_pass and (
                    scalar_sha256(thresholds[eval_task])
                    == threshold_hashes[eval_task]
                )

            validate_bundle(
                bundle,
                expected_categories=tasks[eval_task] if config.max_eval_batches == 0 else None,
                require_label_coverage=config.max_eval_batches == 0,
            )
            metrics = summarize_bundle(bundle, thresholds[eval_task], eval_task)
            row = {"stage": stage + 1, "eval_task": eval_task + 1, "route": route_name, **metrics}
            metric_rows.append(row)
            category_metric_rows.extend(
                {
                    "stage": stage + 1,
                    "eval_task": eval_task + 1,
                    "route": route_name,
                    **category_row,
                }
                for category_row in summarize_categories(
                    bundle, thresholds[eval_task], eval_task
                )
            )
            save_prediction_bundle(
                stage_dir / f"task_{eval_task + 1:02d}_{route_name}.npz",
                bundle,
                config.save_pixel_maps and not config.compact_audit_storage,
            )
            if eval_task in previous_stage_bundles:
                previous_bundle = (
                    load_reference_bundle(previous_stage_bundles[eval_task])
                    if config.compact_audit_storage
                    else previous_stage_bundles[eval_task]
                )
                incremental_drift_rows.append(
                    {
                        "stage": stage + 1,
                        "eval_task": eval_task + 1,
                        "route": route_name,
                        **compare_bundles(previous_bundle, bundle),
                    }
                )
                if config.compact_audit_storage:
                    del previous_bundle
            if config.compact_audit_storage:
                previous_path = (
                    previous_stage_reference_dir / f"task_{eval_task + 1:02d}_float32.npz"
                )
                save_reference_bundle(previous_path, bundle)
                previous_stage_bundles[eval_task] = previous_path
            else:
                previous_stage_bundles[eval_task] = bundle
            if eval_task not in reference_bundles:
                if config.compact_audit_storage:
                    reference_path = (
                        reference_dir / f"task_{eval_task + 1:02d}_{route_name}_float32.npz"
                    )
                    save_reference_bundle(reference_path, bundle)
                    reference_bundles[eval_task] = reference_path
                else:
                    reference_bundles[eval_task] = bundle
            else:
                reference_bundle = (
                    load_reference_bundle(reference_bundles[eval_task])
                    if config.compact_audit_storage
                    else reference_bundles[eval_task]
                )
                drift = compare_bundles(reference_bundle, bundle)
                if config.compact_audit_storage:
                    del reference_bundle
                drift_rows.append(
                    {"stage": stage + 1, "eval_task": eval_task + 1, "route": route_name, **drift}
                )
                if config.method in ISOLATED_METHODS:
                    gate_pass = gate_pass and drift["max_abs_score_diff"] < 1e-6
                    gate_pass = gate_pass and drift["max_abs_pixel_map_diff"] < 1e-6

            if config.method in ROUTED_METHODS:
                auto_bundle = evaluate_task(
                    model,
                    test_loader,
                    device,
                    config,
                    eval_task,
                    route_centroids=centroids if config.method == "isolation" else None,
                    route_prototypes=(
                        router_prototypes
                        if config.method in K3_ROUTED_METHODS
                        else None
                    ),
                    expert_states=expert_states,
                    expert_state_loader=state_loader,
                )
                validate_bundle(
                    auto_bundle,
                    expected_categories=tasks[eval_task] if config.max_eval_batches == 0 else None,
                    require_label_coverage=config.max_eval_batches == 0,
                )
                auto_metrics = summarize_bundle(auto_bundle, thresholds[eval_task], eval_task)
                metric_rows.append(
                    {
                        "stage": stage + 1,
                        "eval_task": eval_task + 1,
                        "route": (
                            "centroid" if config.method == "isolation" else "k3_patch"
                        ),
                        **auto_metrics,
                    }
                )
                category_metric_rows.extend(
                    {
                        "stage": stage + 1,
                        "eval_task": eval_task + 1,
                        "route": (
                            "centroid" if config.method == "isolation" else "k3_patch"
                        ),
                        **category_row,
                    }
                    for category_row in summarize_categories(
                        auto_bundle, thresholds[eval_task], eval_task
                    )
                )
                save_prediction_bundle(
                    stage_dir
                    / (
                        f"task_{eval_task + 1:02d}_centroid.npz"
                        if config.method == "isolation"
                        else f"task_{eval_task + 1:02d}_k3_patch.npz"
                    ),
                    auto_bundle,
                    config.save_pixel_maps and not config.compact_audit_storage,
                )
                if eval_task not in routing_reference_bundles:
                    routing_reference_bundles[eval_task] = (
                        routing_only_bundle(auto_bundle)
                        if config.compact_audit_storage
                        else auto_bundle
                    )
                else:
                    routing_drift_rows.append(
                        {
                            "stage": stage + 1,
                            "eval_task": eval_task + 1,
                            "route": (
                                "centroid"
                                if config.method == "isolation"
                                else "k3_patch"
                            ),
                            **route_flip_metrics(
                                routing_reference_bundles[eval_task], auto_bundle
                            ),
                        }
                    )
                if config.compact_audit_storage:
                    del auto_bundle

            if config.compact_audit_storage:
                del bundle

        write_stage_csv(output_dir / "stage_metrics.csv", metric_rows)
        write_stage_csv(output_dir / "category_metrics.csv", category_metric_rows)
        write_stage_csv(output_dir / "drift_metrics.csv", drift_rows)
        write_stage_csv(
            output_dir / "incremental_drift_metrics.csv",
            incremental_drift_rows,
        )
        write_stage_csv(output_dir / "routing_drift_metrics.csv", routing_drift_rows)
        write_stage_csv(output_dir / "invariant_metrics.csv", invariant_rows)
        write_stage_csv(
            output_dir / "validation_metrics.csv",
            validation_metric_rows,
        )
        write_stage_csv(
            output_dir / "validation_drift_metrics.csv",
            validation_drift_rows,
        )
        write_stage_csv(
            output_dir / "validation_incremental_drift_metrics.csv",
            validation_incremental_drift_rows,
        )

    metadata["completed_at_unix"] = time.time()
    metadata["elapsed_seconds"] = metadata["completed_at_unix"] - metadata["started_at_unix"]
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    checkpoint_paths = [
        output_dir / f"stage_{stage + 1:02d}" / "expert.pth"
        for stage in range(len(tasks))
    ]
    subspace_storage = None
    if config.method in SUBSPACE_METHODS:
        task_head_paths = [
            output_dir / f"stage_{stage + 1:02d}" / "task_head.pth"
            for stage in range(len(tasks))
        ]
        router_raw_bytes = sum(
            prototype.numel() * prototype.element_size()
            for prototype in router_prototypes
        )
        deployable_state_bytes = (
            shared_frozen_decoder_path.stat().st_size
            + checkpoint_paths[-1].stat().st_size
        )
        subspace_storage = {
            "shared_frozen_decoder_bytes": shared_frozen_decoder_path.stat().st_size,
            "final_complete_checkpoint_bytes": checkpoint_paths[-1].stat().st_size,
            "all_task_head_checkpoint_bytes": sum(
                path.stat().st_size for path in task_head_paths
            ),
            "aggregate_basis_bytes_fp32": subspace_history.storage_bytes_fp32(),
            "router_raw_bytes": router_raw_bytes,
            "calibration_bytes": len(thresholds) * 8,
            "deployable_state_bytes": deployable_state_bytes,
            "deployable_state_scope": (
                "shared frozen Decoder plus final checkpoint containing every "
                "acquired head, cumulative LoRA, DualGPM bases, thresholds, "
                "router prototypes, and task mapping"
            ),
            "final_complete_state_sha256": checkpoint_state[
                "complete_state_sha256"
            ],
        }
        parameter_report["subspace_basis_bytes_final"] = (
            subspace_history.storage_bytes_fp32()
        )
        parameter_report["subspace_deployable_state_bytes"] = deployable_state_bytes
        with open(output_dir / "parameters.json", "w", encoding="utf-8") as handle:
            json.dump(parameter_report, handle, indent=2)
    summary = {
        "method": config.method,
        "tasks_completed": len(tasks),
        "oracle_retention_gate_pass": (
            gate_pass
            if config.method in ISOLATED_METHODS
            else None
        ),
        "parameters": parameter_report,
        "frozen_decoder_sha256": frozen_decoder_hash,
        "optimizer_audits": optimizer_audits,
        "compact_audit_storage": config.compact_audit_storage,
        "validation_audit_only": config.validation_audit_only,
        "total_checkpoint_size_bytes": sum(path.stat().st_size for path in checkpoint_paths),
        "subspace_storage": subspace_storage,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_run_notes(
        output_dir / "run_notes.md",
        config,
        stream,
        metric_rows,
        gate_pass if config.method in ISOLATED_METHODS else None,
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "shared_ft",
            "isolation",
            "ewc",
            "shared_lora",
            "isolated_lora",
            "isolated_head",
            "subspace_lora",
            "lwf",
        ],
        required=True,
    )
    parser.add_argument("--stream", required=True)
    parser.add_argument("--data-path", default="../mvtec")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--validation-audit-only", action="store_true")
    parser.add_argument("--resize-mask", type=int, default=256)
    parser.add_argument("--no-save-pixel-maps", action="store_true")
    parser.add_argument("--ewc-lambda", type=float, default=0.0)
    parser.add_argument("--fisher-batches", type=int, default=0)
    parser.add_argument("--save-regularizer-state", action="store_true")
    parser.add_argument("--lwf-lambda", type=float, default=1.0)
    parser.add_argument("--lwf-gradient-audit-batches", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--subspace-lambda-start", type=float, default=0.98)
    parser.add_argument("--subspace-lambda-end", type=float, default=1.0)
    parser.add_argument("--subspace-covariance-batches", type=int, default=0)
    parser.add_argument("--compact-audit-storage", action="store_true")
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=0,
        help="Per-class replay images retained for prior categories; 0 disables replay.",
    )
    parser.add_argument(
        "--replay-selector",
        choices=["random", "score_anchor"],
        default="random",
        help=(
            "How prior-normal replay images are chosen per class. 'random' (default) is the "
            "existing uniform-random draw. 'score_anchor' keeps the SAME per-class budget but "
            "retains the top-K normals by highest anomaly score under the previous-stage model "
            "state (the state about to be overwritten) — a Goal C candidate replay selector."
        ),
    )
    return parser


def config_from_namespace(args, parser):
    if args.lwf_lambda < 0:
        parser.error("--lwf-lambda must be non-negative.")
    if args.lwf_gradient_audit_batches < 0:
        parser.error("--lwf-gradient-audit-batches must be non-negative.")
    if not 0.0 < args.subspace_lambda_start <= 1.0:
        parser.error("--subspace-lambda-start must be in (0, 1].")
    if not args.subspace_lambda_start <= args.subspace_lambda_end <= 1.0:
        parser.error(
            "--subspace-lambda-end must be in [subspace-lambda-start, 1]."
        )
    if args.subspace_covariance_batches < 0:
        parser.error("--subspace-covariance-batches must be non-negative.")
    if args.replay_buffer_size < 0:
        parser.error("--replay-buffer-size must be non-negative.")
    return RunConfig(
        method=args.method,
        stream=str(Path(args.stream).resolve()),
        data_path=str(Path(args.data_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        encoder=args.encoder,
        seed=args.seed,
        input_size=args.input_size,
        crop_size=args.crop_size,
        inp_num=args.inp_num,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        threshold_quantile=args.threshold_quantile,
        max_tasks=args.max_tasks,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        validation_audit_only=args.validation_audit_only,
        resize_mask=args.resize_mask,
        save_pixel_maps=not args.no_save_pixel_maps,
        ewc_lambda=args.ewc_lambda,
        fisher_batches=args.fisher_batches,
        save_regularizer_state=args.save_regularizer_state,
        lwf_lambda=args.lwf_lambda,
        lwf_gradient_audit_batches=args.lwf_gradient_audit_batches,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        subspace_lambda_start=args.subspace_lambda_start,
        subspace_lambda_end=args.subspace_lambda_end,
        subspace_covariance_batches=args.subspace_covariance_batches,
        compact_audit_storage=args.compact_audit_storage,
        replay_buffer_size=args.replay_buffer_size,
        replay_selector=args.replay_selector,
    )


def parse_args():
    parser = build_parser()
    return config_from_namespace(parser.parse_args(), parser)


if __name__ == "__main__":
    run(parse_args())
