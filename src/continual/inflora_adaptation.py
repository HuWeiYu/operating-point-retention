"""Mechanism-faithful primitives for an InfLoRA-style AD baseline.

This module intentionally does not launch experiments or depend on MVTec.  It
implements the two semantics that must be unit-tested before a GPU pilot:

1. task-indexed LoRA branches are frozen after acquisition but are summed in
   the latest deployed function; and
2. the new branch's fixed input factor is initialized from current-task
   activation covariance after a DualGPM-style historical projection.

Historical state consists only of aggregate layerwise bases and mode labels;
individual activation rows are never retained by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_PROJECT_MODES = ("remove", "retain")


class CumulativeSubspaceLoRALinear(nn.Module):
    """Frozen Linear layer with task-indexed, cumulatively applied LoRA.

    ``A_t`` is fixed after projected-covariance initialization and only ``B_t``
    is trainable at task ``t``.  At deployment task ``t``, the effective update
    is ``sum_{i=0}^t B_i A_i``.  Thus adding a new branch can change old-input
    outputs even though every old branch is frozen.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        total_tasks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("CumulativeSubspaceLoRALinear requires nn.Linear.")
        if rank <= 0 or rank > base.in_features:
            raise ValueError("rank must be in [1, base.in_features].")
        if total_tasks <= 0:
            raise ValueError("total_tasks must be positive.")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.total_tasks = int(total_tasks)
        self.dropout = nn.Dropout(dropout)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.rank, base.in_features, **factory),
                    requires_grad=False,
                )
                for _ in range(self.total_tasks)
            ]
        )
        self.lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(base.out_features, self.rank, **factory),
                    requires_grad=False,
                )
                for _ in range(self.total_tasks)
            ]
        )
        for factor in self.lora_A:
            nn.init.kaiming_uniform_(factor, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.register_buffer("active_task", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer(
            "_covariance_sum",
            torch.zeros(
                base.in_features,
                base.in_features,
                device=base.weight.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_covariance_count", torch.tensor(0, dtype=torch.int64), persistent=False
        )
        self._collect_covariance = False

    @property
    def current_task(self) -> int:
        return int(self.active_task.item())

    def start_covariance_collection(self, reset: bool = True) -> None:
        if reset:
            self._covariance_sum.zero_()
            self._covariance_count.zero_()
        self._collect_covariance = True

    def stop_covariance_collection(self) -> None:
        self._collect_covariance = False

    def covariance(self, reset: bool = False) -> torch.Tensor:
        count = int(self._covariance_count.item())
        if count == 0:
            raise RuntimeError("No activations were collected for this layer.")
        result = (self._covariance_sum / count).detach().cpu().to(torch.float64)
        if reset:
            self._covariance_sum.zero_()
            self._covariance_count.zero_()
        return result

    def configure_task_from_covariance(
        self,
        task: int,
        covariance: torch.Tensor,
        historical_basis: Optional[torch.Tensor] = None,
        project_mode: str = "remove",
    ) -> Dict[str, float]:
        """Initialize fixed ``A_task`` and expose only ``B_task`` to training."""

        if not 0 <= task < self.total_tasks:
            raise IndexError(f"task {task} outside [0, {self.total_tasks}).")
        covariance = _validate_covariance(covariance, self.base.in_features)
        projected = project_covariance(covariance, historical_basis, project_mode)
        left, singular_values, _ = torch.linalg.svd(projected, full_matrices=False)
        directions = left[:, : self.rank].T / math.sqrt(3.0)

        with torch.no_grad():
            self.lora_A[task].copy_(
                directions.to(self.lora_A[task].device, self.lora_A[task].dtype)
            )
            self.lora_B[task].zero_()
            self.active_task.fill_(task)

        for factor in self.lora_A:
            factor.requires_grad_(False)
        for index, factor in enumerate(self.lora_B):
            factor.requires_grad_(index == task)

        total_norm = float(torch.linalg.vector_norm(covariance))
        projected_norm = float(torch.linalg.vector_norm(projected))
        retained_fraction = projected_norm / max(total_norm, torch.finfo(torch.float64).eps)
        top_energy = float(singular_values[: self.rank].square().sum())
        all_energy = float(singular_values.square().sum())
        return {
            "covariance_frobenius_norm": total_norm,
            "projected_frobenius_norm": projected_norm,
            "projected_norm_fraction": retained_fraction,
            "top_rank_energy_fraction": top_energy
            / max(all_energy, torch.finfo(torch.float64).eps),
        }

    def effective_update(self, task: Optional[int] = None) -> torch.Tensor:
        task = self.current_task if task is None else int(task)
        if task < 0:
            return torch.zeros_like(self.base.weight)
        if task >= self.total_tasks:
            raise IndexError(f"task {task} outside [0, {self.total_tasks}).")
        return torch.stack(
            [self.lora_B[index] @ self.lora_A[index] for index in range(task + 1)]
        ).sum(dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._collect_covariance:
            with torch.no_grad():
                rows = inputs.detach().reshape(-1, inputs.shape[-1]).to(torch.float32)
                self._covariance_sum.add_(rows.T @ rows)
                self._covariance_count.add_(rows.shape[0])
        if self.current_task < 0:
            return self.base(inputs)
        update = F.linear(self.dropout(inputs), self.effective_update())
        return self.base(inputs) + self.scaling * update


def _validate_covariance(covariance: torch.Tensor, dimension: int) -> torch.Tensor:
    covariance = torch.as_tensor(covariance, dtype=torch.float64, device="cpu")
    if covariance.shape != (dimension, dimension):
        raise ValueError(
            f"Expected covariance {(dimension, dimension)}, got {tuple(covariance.shape)}."
        )
    if not torch.isfinite(covariance).all():
        raise ValueError("Covariance contains non-finite values.")
    return covariance


def project_covariance(
    covariance: torch.Tensor,
    historical_basis: Optional[torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Apply the official DualGPM remove/retain projection convention."""

    if historical_basis is None:
        return covariance
    if mode not in VALID_PROJECT_MODES:
        raise ValueError(f"Unknown project mode: {mode}.")
    basis = torch.as_tensor(historical_basis, dtype=torch.float64, device="cpu")
    if basis.ndim != 2 or basis.shape[0] != covariance.shape[0]:
        raise ValueError("Historical basis has an incompatible shape.")
    projector_times_covariance = basis @ (basis.T @ covariance)
    if mode == "remove":
        return covariance - projector_times_covariance
    return projector_times_covariance


def _energy_rank(singular_values: torch.Tensor, threshold: float) -> int:
    energy = singular_values.square()
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps:
        return 1
    cumulative = torch.cumsum(energy / total, dim=0)
    return max(int((cumulative < threshold).sum()), 1)


def _orthonormal_span(matrix: torch.Tensor, columns: int) -> torch.Tensor:
    if columns <= 0:
        return matrix.new_zeros((matrix.shape[0], 0))
    left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return left[:, :columns]


def _compact_dual_basis(basis: torch.Tensor, mode: str) -> tuple[torch.Tensor, str]:
    dimension, columns = basis.shape
    basis = _orthonormal_span(basis, columns)
    if mode == "remove" and columns > dimension / 2:
        left, _, _ = torch.linalg.svd(basis, full_matrices=True)
        return left[:, columns:], "retain"
    if mode == "retain" and columns > dimension / 2:
        raise RuntimeError("A retain basis must encode at most half the dimensions.")
    return basis, mode


def update_dual_gpm_basis(
    activation_covariance: torch.Tensor,
    threshold: float,
    previous_basis: Optional[torch.Tensor] = None,
    previous_mode: Optional[str] = None,
) -> tuple[torch.Tensor, str, Dict[str, float]]:
    """Update an aggregate DualGPM basis without retaining activation rows."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")
    covariance = torch.as_tensor(
        activation_covariance, dtype=torch.float64, device="cpu"
    )
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("activation_covariance must be square.")
    dimension = covariance.shape[0]
    left_full, singular_full, _ = torch.linalg.svd(covariance, full_matrices=False)
    total_energy = float(singular_full.square().sum())

    if previous_basis is None:
        rank = _energy_rank(singular_full, threshold)
        basis, mode = _compact_dual_basis(left_full[:, :rank], "remove")
    else:
        if previous_mode not in VALID_PROJECT_MODES:
            raise ValueError("previous_mode must be remove or retain.")
        basis = torch.as_tensor(previous_basis, dtype=torch.float64, device="cpu")
        if basis.ndim != 2 or basis.shape[0] != dimension:
            raise ValueError("previous_basis has an incompatible shape.")
        projected = basis @ (basis.T @ covariance)

        if previous_mode == "remove":
            residual = covariance - projected
            left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
            residual_energy = float(singular.square().sum())
            accumulated = (total_energy - residual_energy) / max(
                total_energy, torch.finfo(torch.float64).eps
            )
            ratios = singular.square() / max(
                total_energy, torch.finfo(torch.float64).eps
            )
            rank = 0
            for ratio in ratios:
                if accumulated < threshold:
                    accumulated += float(ratio)
                    rank += 1
                else:
                    break
            combined = (
                torch.cat((basis, left[:, :rank]), dim=1) if rank else basis.clone()
            )
            basis, mode = _compact_dual_basis(combined, "remove")
        else:
            left, singular, _ = torch.linalg.svd(projected, full_matrices=False)
            retained_energy = float(singular.square().sum())
            accumulated = retained_energy / max(
                total_energy, torch.finfo(torch.float64).eps
            )
            ratios = singular.square() / max(
                total_energy, torch.finfo(torch.float64).eps
            )
            rank = 0
            for ratio in ratios:
                if accumulated >= 1.0 - threshold:
                    accumulated -= float(ratio)
                    rank += 1
                else:
                    break
            remaining = max(basis.shape[1] - rank, 0)
            if remaining:
                reduced = basis - left[:, :rank] @ (left[:, :rank].T @ basis)
                basis = _orthonormal_span(reduced, remaining)
            else:
                basis = covariance.new_zeros((dimension, 0))
            mode = "retain"
            basis, mode = _compact_dual_basis(basis, mode)

    represented = basis @ (basis.T @ covariance) if basis.shape[1] else torch.zeros_like(covariance)
    represented_fraction = float(torch.linalg.vector_norm(represented)) / max(
        float(torch.linalg.vector_norm(covariance)), torch.finfo(torch.float64).eps
    )
    return basis, mode, {
        "dimension": float(dimension),
        "basis_columns": float(basis.shape[1]),
        "basis_bytes_fp32": float(basis.numel() * 4),
        "represented_norm_fraction": represented_fraction,
        "activation_spectral_energy": total_energy,
    }


@dataclass
class DualGPMState:
    """Serializable aggregate historical state for all adapted Linear layers."""

    total_tasks: int
    lambda_start: float = 0.98
    lambda_end: float = 1.0
    bases: Dict[str, torch.Tensor] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)

    def threshold(self, task: int) -> float:
        if not 0 <= task < self.total_tasks:
            raise IndexError(f"task {task} outside [0, {self.total_tasks}).")
        return self.lambda_start + (
            (self.lambda_end - self.lambda_start) * task / self.total_tasks
        )

    def update(
        self, task: int, covariances: Mapping[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        reports: Dict[str, Dict[str, float]] = {}
        threshold = self.threshold(task)
        for name, covariance in covariances.items():
            basis, mode, report = update_dual_gpm_basis(
                covariance,
                threshold,
                previous_basis=self.bases.get(name),
                previous_mode=self.modes.get(name),
            )
            self.bases[name] = basis.to(torch.float32).contiguous()
            self.modes[name] = mode
            reports[name] = report
        return reports

    def state_dict(self) -> Dict[str, object]:
        return {
            "total_tasks": self.total_tasks,
            "lambda_start": self.lambda_start,
            "lambda_end": self.lambda_end,
            "bases": {name: value.detach().cpu().clone() for name, value in self.bases.items()},
            "modes": dict(self.modes),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["total_tasks"]) != self.total_tasks:
            raise RuntimeError("DualGPM total_tasks mismatch.")
        if float(state["lambda_start"]) != self.lambda_start:
            raise RuntimeError("DualGPM lambda_start mismatch.")
        if float(state["lambda_end"]) != self.lambda_end:
            raise RuntimeError("DualGPM lambda_end mismatch.")
        bases = state["bases"]
        modes = state["modes"]
        if not isinstance(bases, Mapping) or not isinstance(modes, Mapping):
            raise TypeError("Malformed DualGPM state.")
        self.bases = {
            str(name): torch.as_tensor(value).detach().cpu().clone().to(torch.float32)
            for name, value in bases.items()
        }
        self.modes = {str(name): str(value) for name, value in modes.items()}
        if set(self.bases) != set(self.modes):
            raise RuntimeError("DualGPM basis/mode keys do not match.")
        invalid = {name: mode for name, mode in self.modes.items() if mode not in VALID_PROJECT_MODES}
        if invalid:
            raise RuntimeError(f"Invalid DualGPM modes: {invalid}")

    def storage_bytes_fp32(self) -> int:
        return sum(value.numel() * 4 for value in self.bases.values())


def inject_cumulative_subspace_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    total_tasks: int,
    dropout: float = 0.0,
    target_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    prefix: str = "",
) -> Dict[str, CumulativeSubspaceLoRALinear]:
    """Replace selected Linear children and return their stable full names."""

    replaced: Dict[str, CumulativeSubspaceLoRALinear] = {}
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and (
            target_filter is None or target_filter(full_name, child)
        ):
            wrapped = CumulativeSubspaceLoRALinear(
                child,
                rank=rank,
                alpha=alpha,
                total_tasks=total_tasks,
                dropout=dropout,
            )
            setattr(module, name, wrapped)
            replaced[full_name] = wrapped
        else:
            replaced.update(
                inject_cumulative_subspace_lora(
                    child,
                    rank=rank,
                    alpha=alpha,
                    total_tasks=total_tasks,
                    dropout=dropout,
                    target_filter=target_filter,
                    prefix=full_name,
                )
            )
    return replaced


def decoder_attention_target(name: str, _: nn.Linear) -> bool:
    """Pilot target: cross-attention query and combined key/value projections."""

    return name.endswith(".attn.q") or name.endswith(".attn.kv")


def collect_covariances(
    layers: Mapping[str, CumulativeSubspaceLoRALinear], reset: bool = True
) -> Dict[str, torch.Tensor]:
    return {name: layer.covariance(reset=reset) for name, layer in layers.items()}


def configure_cumulative_task(
    layers: Mapping[str, CumulativeSubspaceLoRALinear],
    task: int,
    covariances: Mapping[str, torch.Tensor],
    history: DualGPMState,
) -> Dict[str, Dict[str, float]]:
    if set(layers) != set(covariances):
        raise RuntimeError("Layer/covariance keys do not match.")
    reports: Dict[str, Dict[str, float]] = {}
    for name, layer in layers.items():
        reports[name] = layer.configure_task_from_covariance(
            task,
            covariances[name],
            historical_basis=history.bases.get(name),
            project_mode=history.modes.get(name, "remove"),
        )
    return reports


def current_trainable_parameters(
    layers: Iterable[CumulativeSubspaceLoRALinear],
) -> list[nn.Parameter]:
    parameters = [
        parameter
        for layer in layers
        for parameter in layer.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("No current cumulative LoRA branch is trainable.")
    return parameters


def cumulative_lora_state_dict(
    layers: Mapping[str, CumulativeSubspaceLoRALinear],
    task: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Serialize acquired cumulative branches without future placeholders.

    The returned mapping is deliberately flat so it can be hashed with the
    same deterministic tensor-state audit used by the continual runner.
    """

    if not layers:
        raise ValueError("At least one cumulative LoRA layer is required.")
    active_tasks = {layer.current_task for layer in layers.values()}
    if len(active_tasks) != 1:
        raise RuntimeError(f"Cumulative LoRA active tasks disagree: {active_tasks}.")
    active_task = active_tasks.pop() if task is None else int(task)
    if active_task < 0:
        raise RuntimeError("No cumulative LoRA task has been acquired yet.")

    state: Dict[str, torch.Tensor] = {}
    for name, layer in layers.items():
        if active_task >= layer.total_tasks:
            raise IndexError(
                f"task {active_task} outside [0, {layer.total_tasks}) for {name}."
            )
        state[f"{name}.active_task"] = torch.tensor(active_task, dtype=torch.int64)
        for index in range(active_task + 1):
            state[f"{name}.lora_A.{index}"] = (
                layer.lora_A[index].detach().cpu().clone()
            )
            state[f"{name}.lora_B.{index}"] = (
                layer.lora_B[index].detach().cpu().clone()
            )
    return state


def load_cumulative_lora_state(
    layers: Mapping[str, CumulativeSubspaceLoRALinear],
    state: Mapping[str, torch.Tensor],
) -> int:
    """Restore acquired branches and return the common active task index."""

    if not layers:
        raise ValueError("At least one cumulative LoRA layer is required.")
    expected_prefixes = set(layers)
    state_keys = set(state)
    active_tasks = set()
    required_keys = set()

    with torch.no_grad():
        for name, layer in layers.items():
            active_key = f"{name}.active_task"
            if active_key not in state:
                raise RuntimeError(f"Missing cumulative LoRA key: {active_key}.")
            active_task = int(torch.as_tensor(state[active_key]).item())
            if not 0 <= active_task < layer.total_tasks:
                raise RuntimeError(
                    f"Invalid active task {active_task} for cumulative layer {name}."
                )
            active_tasks.add(active_task)
            required_keys.add(active_key)
            for index in range(active_task + 1):
                for factor_name, factors in (
                    ("lora_A", layer.lora_A),
                    ("lora_B", layer.lora_B),
                ):
                    key = f"{name}.{factor_name}.{index}"
                    if key not in state:
                        raise RuntimeError(f"Missing cumulative LoRA key: {key}.")
                    source = torch.as_tensor(state[key])
                    if tuple(source.shape) != tuple(factors[index].shape):
                        raise RuntimeError(
                            f"Shape mismatch for {key}: {tuple(source.shape)} != "
                            f"{tuple(factors[index].shape)}."
                        )
                    factors[index].copy_(
                        source.to(device=factors[index].device, dtype=factors[index].dtype)
                    )
                    required_keys.add(key)
            layer.active_task.fill_(active_task)
            for factor in layer.lora_A:
                factor.requires_grad_(False)
            for factor in layer.lora_B:
                factor.requires_grad_(False)

    if len(active_tasks) != 1:
        raise RuntimeError(f"Restored cumulative LoRA tasks disagree: {active_tasks}.")
    unexpected = sorted(state_keys - required_keys)
    if unexpected:
        unknown_prefixes = sorted(
            key for key in unexpected if not any(key.startswith(f"{name}.") for name in expected_prefixes)
        )
        raise RuntimeError(
            "Unexpected cumulative LoRA state keys; "
            f"unexpected={unexpected}, unknown_prefixes={unknown_prefixes}."
        )
    return active_tasks.pop()


def cumulative_update_report(
    layers: Mapping[str, CumulativeSubspaceLoRALinear], task: Optional[int] = None
) -> Dict[str, object]:
    """Report current-branch and cumulative update norms for auditing."""

    per_layer: Dict[str, Dict[str, float]] = {}
    current_squared = 0.0
    cumulative_squared = 0.0
    for name, layer in layers.items():
        active_task = layer.current_task if task is None else int(task)
        if active_task < 0:
            raise RuntimeError("No cumulative LoRA task has been acquired yet.")
        current = layer.lora_B[active_task] @ layer.lora_A[active_task]
        cumulative = layer.effective_update(active_task)
        current_norm = float(torch.linalg.vector_norm(current.detach()).cpu())
        cumulative_norm = float(torch.linalg.vector_norm(cumulative.detach()).cpu())
        current_squared += current_norm**2
        cumulative_squared += cumulative_norm**2
        per_layer[name] = {
            "current_branch_frobenius_norm": current_norm,
            "cumulative_update_frobenius_norm": cumulative_norm,
        }
    return {
        "current_branch_global_frobenius_norm": math.sqrt(current_squared),
        "cumulative_update_global_frobenius_norm": math.sqrt(cumulative_squared),
        "layers": per_layer,
    }
