"""
Utility functions for adversarial attack training and evaluation.

Organized into sections:
1. Device and noise utilities (GPU/CPU handling, RNG)
2. Image processing utilities (preprocessing, perturbation, conversion)
3. Embedding utilities (model-dependent embedding extraction)
4. Pipeline utilities (paired evaluation with shared noise)
5. Configuration utilities (argument handling)
"""

from __future__ import annotations

import inspect
import random
from typing import Any, Callable, Dict, Optional

import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from utils.adversarial_metrics import evaluate_displacement_metrics


############################################################################################
# Section 1: Device and Noise Utilities
############################################################################################


def resolve_torch_device(device: Any = "cuda") -> str:
    """Return a valid torch device string, falling back to CPU when CUDA is unavailable."""
    resolved = str(device)
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return resolved


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for deterministic experiment setup."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_shared_initial_noise(
    batch_size: int,
    device: Any = "cuda",
    seed: int = 1234,
) -> torch.Tensor:
    """Sample initial diffusion noise once so source/perturbed runs are comparable."""
    resolved_device = resolve_torch_device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    return torch.randn(int(batch_size), 3, device=resolved_device, generator=generator)


############################################################################################
# Section 2: Image Processing Utilities
############################################################################################


def conditional_preprocessing(source_image, pipeline, device="cuda"):
    """Preprocess image based on model type (DINOv2 vs CLIP)."""
    if source_image.mode != "RGB":
        source_image = source_image.convert("RGB")

    if ("YFCC" in pipeline.model_path) or ("iNaturalist" in pipeline.model_path): 
        # DINOv2-based model
        tensor = (
            pipeline.cond_preprocessing.augmentation(source_image)
            .unsqueeze(0)
            .to(device)
        )  # [1, 3, H, W]
    else:  
        # CLIP-based model
        tensor = pipeline.cond_preprocessing.processor(
            images=source_image, return_tensors="pt"
        )["pixel_values"].to(device)  # [1, 3, H, W]
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL image with proper denormalization."""
    # Unnormalize (assuming normalization with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    unnormalized_tensor = unnormalize(tensor.squeeze(0)).clamp(0, 1)
    
    # Convert to PIL image
    pil_image = transforms.ToPILImage()(unnormalized_tensor.cpu())
    return pil_image


def add_perturbation_to_image(image: Image.Image, perturbation: torch.Tensor, pipeline) -> Image.Image:
    """Apply perturbation to image and convert back to PIL format."""
    # Convert the image to a tensor
    image_tensor = conditional_preprocessing(image, pipeline, device=perturbation.device)  # [1, 3, H, W]
    # Add the perturbation to the image tensor
    perturbed_tensor = image_tensor + perturbation
    # Convert back to PIL
    perturbed_image = tensor_to_pil(perturbed_tensor)
    return perturbed_image


############################################################################################
# Section 3: Embedding Utilities
############################################################################################


def compute_embedding(image_tensor, batch_size, pipeline, device="cuda", track_grad=True):
    """Extract embedding from image tensor, handling model-specific variations."""
    if ("YFCC" in pipeline.model_path) or ("iNaturalist" in pipeline.model_path):
        # DINOv2 model
        emb_single = pipeline.cond_preprocessing.emb_model(image_tensor)
    else:
        # CLIP model
        input_dict = {"pixel_values": image_tensor}
        if track_grad:
            outputs = pipeline.cond_preprocessing.emb_model(**input_dict)
        else:
            with torch.no_grad():
                outputs = pipeline.cond_preprocessing.emb_model(**input_dict)
        emb_single = outputs.last_hidden_state[:, 0]
    
    # Repeat to batch size
    emb = emb_single.repeat(batch_size, 1)
    return emb


def model_dependent_embedding(image_tensor, pipeline, track_grad=True):
    """Extract single embedding from image tensor (no batching)."""
    if "YFCC" in pipeline.model_path or "iNaturalist" in pipeline.model_path:
        # DINOv2 model
        if track_grad:
            z_source = pipeline.cond_preprocessing.emb_model(image_tensor)
        else:
            with torch.no_grad():
                z_source = pipeline.cond_preprocessing.emb_model(image_tensor)
    else:  
        # CLIP model
        if track_grad:
            z_source = pipeline.cond_preprocessing.emb_model(image_tensor)["last_hidden_state"][:, 0]
        else:
            with torch.no_grad():
                z_source = pipeline.cond_preprocessing.emb_model(pixel_values=image_tensor)["last_hidden_state"][:, 0]
    
    return z_source


############################################################################################
# Section 4: Pipeline Utilities
############################################################################################


def run_paired_pipeline_with_shared_noise(
    pipeline,
    source_image,
    perturbed_image,
    batch_size: int = 256,
    cfg: float = 10.0,
    num_steps: Optional[int] = None,
    seed: int = 1234,
    device: Any = "cuda",
) -> Dict[str, Any]:
    """Run source and perturbed images with identical initial noise and return trajectories + metrics."""
    x_n = make_shared_initial_noise(batch_size=batch_size, device=device, seed=seed)

    eval_kwargs = {
        "batch_size": int(batch_size),
        "cfg": float(cfg),
        "x_N": x_n,
        "return_trajectories": True,
    }
    if num_steps is not None:
        eval_kwargs["num_steps"] = int(num_steps)

    with torch.inference_mode():
        gps_source, traj_source = pipeline(source_image, **eval_kwargs)
        gps_perturbed, traj_perturbed = pipeline(perturbed_image, **eval_kwargs)
    metrics = evaluate_displacement_metrics(traj_source, traj_perturbed)

    return {
        "gps_source": gps_source,
        "gps_perturbed": gps_perturbed,
        "traj_source": traj_source,
        "traj_perturbed": traj_perturbed,
        "metrics": metrics,
    }


############################################################################################
# Section 5: Configuration Utilities
############################################################################################


def filter_kwargs_for(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of kwargs accepted by func."""
    sig = inspect.signature(func)
    params = sig.parameters

    # Pass through untouched when func accepts arbitrary kwargs.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    accepted = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def expand_per_budget_kwargs(attack_kwargs: list[Dict[str, Any]], n_budgets: int) -> list[Dict[str, Any]]:
    """Normalize per-budget kwargs to length n_budgets."""
    if len(attack_kwargs) not in (1, n_budgets):
        raise ValueError(
            f"attack_kwargs must have length 1 or n_budgets={n_budgets}, got {len(attack_kwargs)}"
        )
    if len(attack_kwargs) == 1 and n_budgets > 1:
        return [dict(attack_kwargs[0]) for _ in range(n_budgets)]
    return [dict(kwargs) for kwargs in attack_kwargs]