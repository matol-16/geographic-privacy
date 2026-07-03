"""
Shared helpers for the plotting package.

Pure data utilities (no matplotlib/cartopy import) so they can be reused by every
plot module without pulling heavy geo dependencies: results loading, metric-tensor
extraction, sample summarisation, attack-name display, and a small JSON dumper used
to persist the numbers behind each plot (success rates, displacement summaries).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _load_attack_results(results_dir, dataset_name, attack_types):
    """Load per-attack results, falling back to a saved evaluation state if needed."""
    if results_dir is None:
        raise ValueError("results_dir must be provided when all_results/results are not supplied")

    loaded_results = {}
    missing_attack_types = []
    for attack_type in attack_types:
        results_path = os.path.join(results_dir, f"{dataset_name}_{attack_type}_results.pt")
        if os.path.exists(results_path):
            loaded_results[attack_type] = torch.load(results_path)
        else:
            missing_attack_types.append(attack_type)

    if not missing_attack_types:
        return loaded_results

    state_candidates = sorted(
        Path(results_dir).glob(f"{dataset_name}_seed*_eval_state*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not state_candidates:
        missing_paths = [os.path.join(results_dir, f"{dataset_name}_{attack_type}_results.pt") for attack_type in missing_attack_types]
        raise FileNotFoundError(
            "Missing saved results files and no evaluation state was found. "
            f"Missing: {missing_paths}"
        )

    state = torch.load(state_candidates[0], map_location="cpu")
    state_results = state.get("results") if isinstance(state, dict) else None
    if not isinstance(state_results, dict):
        missing_paths = [os.path.join(results_dir, f"{dataset_name}_{attack_type}_results.pt") for attack_type in missing_attack_types]
        raise FileNotFoundError(
            "Missing saved results files and the latest evaluation state does not contain results. "
            f"Missing: {missing_paths}"
        )

    for attack_type in missing_attack_types:
        if attack_type in state_results:
            loaded_results[attack_type] = state_results[attack_type]

    still_missing = [attack_type for attack_type in attack_types if attack_type not in loaded_results]
    if still_missing:
        missing_paths = [os.path.join(results_dir, f"{dataset_name}_{attack_type}_results.pt") for attack_type in still_missing]
        raise FileNotFoundError(
            "Missing saved results files and the available evaluation state does not cover all requested attack types. "
            f"Missing: {missing_paths}"
        )

    return loaded_results


def _sanitize_lon_lat(coords):
    """Return valid [lat, lon] rows only, wrapping lon to [-180, 180]."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected coords shape [N, 2] with [lat, lon]")

    arr = arr.copy()
    arr[:, 1] = ((arr[:, 1] + 180.0) % 360.0) - 180.0
    arr[:, 0] = np.clip(arr[:, 0], -90.0, 90.0)

    valid = np.isfinite(arr).all(axis=1)
    return arr[valid], valid


def _plot_valid_path(ax, lat_lon_traj, **plot_kwargs):
    """Plot only contiguous valid trajectory segments to avoid Shapely warnings."""
    traj = np.asarray(lat_lon_traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2:
        return

    traj = traj.copy()
    traj[:, 1] = ((traj[:, 1] + 180.0) % 360.0) - 180.0
    traj[:, 0] = np.clip(traj[:, 0], -90.0, 90.0)
    valid = np.isfinite(traj).all(axis=1)

    start = None
    for i, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = i
        if (not is_valid or i == len(valid) - 1) and start is not None:
            end = i if not is_valid else i + 1
            if end - start >= 2:
                seg = traj[start:end]
                ax.plot(seg[:, 1], seg[:, 0], **plot_kwargs)
            start = None


def _metric_aliases(metric_name):
    if metric_name == "final_step_displacement_predicted":
        return ["final_step_displacement_predicted", "final_step_displacement", "final_step_displacement_clean"]
    if metric_name == "final_step_displacement_true":
        return ["final_step_displacement_true"]
    if metric_name == "final_step_displacement":
        return ["final_step_displacement", "final_step_displacement_predicted", "final_step_displacement_clean"]
    return [metric_name]


def _get_metric_tensor(attack_results, metric_name):
    for alias in _metric_aliases(metric_name):
        tensor = attack_results.get(alias)
        if isinstance(tensor, torch.Tensor):
            return tensor
    return None


def _get_metric_samples_by_budget(attack_results, metric_name):
    tensor = _get_metric_tensor(attack_results, metric_name)
    if tensor is not None:
        tensor = tensor.detach().cpu()
        return [tensor[i].reshape(-1).numpy() for i in range(tensor.shape[0])]

    restart_results = attack_results.get("restart_results")
    if restart_results is None:
        return None

    samples_by_budget = []
    for budget_results in restart_results:
        budget_samples = []
        for image_results in budget_results:
            if not image_results:
                continue
            for restart_result in image_results:
                value = restart_result.get(metric_name)
                if value is None:
                    for alias in _metric_aliases(metric_name):
                        value = restart_result.get(alias)
                        if value is not None:
                            break
                if value is not None:
                    budget_samples.append(float(value))
        samples_by_budget.append(np.asarray(budget_samples, dtype=np.float64))
    return samples_by_budget


def _summarize_samples(samples: np.ndarray, attack_name, budget) -> tuple[float, float, float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    # drop nan results and log the amount that was dropped
    valid_finite = finite[~np.isnan(finite)]
    nb_dropped = finite.size - valid_finite.size
    if nb_dropped > 0:
        print(f"Budget {budget:.3f}, attack {attack_name}: Dropped {nb_dropped} samples out of {valid_finite.size} total samples for metric summarization.")

    with np.errstate(all="ignore"):
        mean_value = float(np.mean(finite))
        median_value = float(np.median(finite))
        q25_value = float(np.quantile(finite, 0.25))
        q75_value = float(np.quantile(finite, 0.75))
    return mean_value, median_value, q25_value, q75_value


def _select_displacement_metric(gps_true: bool) -> str:
    return "final_step_displacement_true" if gps_true else "final_step_displacement_predicted"


def select_closest_budget(attack_budgets, target_eps: float) -> float:
    """Return the value in ``attack_budgets`` closest to ``target_eps``.

    Used by the ablation plots (robustness, model-transfer, sampling-steps,
    DTD-variance) to pin every panel to a single attack budget instead of
    drawing one line/box per budget.
    """
    arr = np.asarray(attack_budgets, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("attack_budgets is empty")
    idx = int(np.argmin(np.abs(arr - target_eps)))
    return float(arr[idx])


def _display_attack_name(attack_name: str) -> str:
    normalized = str(attack_name).lower()
    # DTD = Diffusion Trajectory Deviation (raw cosine-similarity objective).
    if normalized in ("dtd", "diffusion_cosine_neg"):
        return "DTD"
    if normalized == "diffusion":
        return "Diffusion"
    if normalized in ("sampling", "diffusion_salman", "salman"):
        return "Sampling"
    if normalized == "diffusion_l2":
        return "L2"
    if normalized == "ace":
        return "ACE"
    if normalized == "unidef":
        return "UniDef"
    if normalized == "unidef_nofdje":
        return "UniDef (no FDJE)"
    if normalized == "encoder":
        return "Encoder"
    if normalized == "geoshield":
        return "GeoShield"
    if normalized == "training_loss":
        return "AdvDM"
    return str(attack_name).replace("_", " ").title()


def save_plot_json(plot_dir: Optional[str], filename_stem: str, payload: dict) -> Optional[str]:
    """Persist the numbers behind a plot as JSON next to the PNG.

    ``filename_stem`` should match the PNG stem so the two files sit side by side.
    Returns the written path, or None when ``plot_dir`` is None (interactive mode).
    """
    if plot_dir is None:
        return None
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{filename_stem}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved plot data to: {path}")
    return path
