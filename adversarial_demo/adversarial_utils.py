
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional

from torchvision import transforms
import torch
from PIL import Image
from adversarial_metrics import evaluate_displacement_metrics


def filter_kwargs_for(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of kwargs accepted by func."""
    sig = inspect.signature(func)
    params = sig.parameters

    # Pass through untouched when func accepts arbitrary kwargs.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    accepted = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def resolve_torch_device(device: Any = "cuda") -> str:
    """Return a valid torch device string, falling back to CPU when CUDA is unavailable."""
    resolved = str(device)
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return resolved


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

def add_perturbation_to_image(image: Image, perturbation: torch.Tensor, pipeline):
    # Convert the image to a tensor
    image_tensor = conditional_preprocessing(image, pipeline, device=perturbation.device)  # [1, 3, H, W]
    # Add the perturbation to the image tensor
    perturbed_tensor = image_tensor + perturbation

    # Convert the perturbed tensor back to an image. cond_prepocesser has no deprocess method, so we need to implement it ourselves. We just need to unnormalize the image and convert it back to PIL format.
    perturbed_image = tensor_to_pil(perturbed_tensor.squeeze(0))

    return perturbed_image

def tensor_to_pil(tensor: torch.Tensor):
    # Unnormalize the tensor (assuming it was normalized with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5])
    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    unnormalized_tensor = unnormalize(tensor.squeeze(0)).clamp(0, 1)

    # Convert to PIL image
    pil_image = transforms.ToPILImage()(unnormalized_tensor.cpu())
    return pil_image


def conditional_preprocessing(source_image, pipeline, device="cuda"):
    if ("YFCC" in pipeline.model_path) or ("iNaturalist" in pipeline.model_path): #dinov2
        if source_image.mode != "RGB":
            source_image = source_image.convert("RGB")
        tensor = (
            pipeline.cond_preprocessing.augmentation(source_image)
            .unsqueeze(0)
            .to(device)
        )  # [1, 3, H, W]
        
    else: #clip 
        tensor = pipeline.cond_preprocessing.processor(
            images=source_image, return_tensors="pt"
        )["pixel_values"].to(device)  # [1, 3, H, W]
    return tensor


def compute_embedding(image_tensor, batch_size, pipeline, device="cuda", track_grad=True):
    #Conditional embedding depends on the embedder
    if ("YFCC" in pipeline.model_path) or ("iNaturalist" in pipeline.model_path):
        emb_single = pipeline.cond_preprocessing.emb_model(image_tensor)
    else:
        input_dict = {"pixel_values": image_tensor}
        if track_grad:
            outputs = pipeline.cond_preprocessing.emb_model(**input_dict)
        else:
            with torch.no_grad():
                outputs = pipeline.cond_preprocessing.emb_model(**input_dict)
        emb_single = outputs.last_hidden_state[:, 0]
    emb = emb_single.repeat(batch_size, 1)
    return emb


def model_dependent_embedding(image_tensor, pipeline, track_grad=True):
    if "YFCC" in pipeline.model_path or "iNaturalist" in pipeline.model_path:
        if track_grad:
            z_source = pipeline.cond_preprocessing.emb_model(image_tensor)
        else:
            with torch.no_grad():
                z_source = pipeline.cond_preprocessing.emb_model(image_tensor)
    else: #clip
        if track_grad:
            z_source = pipeline.cond_preprocessing.emb_model(image_tensor)["last_hidden_state"][:, 0]
        else:
            with torch.no_grad():
                z_source = pipeline.cond_preprocessing.emb_model(pixel_values=image_tensor)["last_hidden_state"][:, 0]
    return z_source


def expand_per_budget_kwargs(attack_kwargs: list[Dict[str, Any]], n_budgets: int) -> list[Dict[str, Any]]:
    """Normalize per-budget kwargs to length n_budgets."""
    if len(attack_kwargs) not in (1, n_budgets):
        raise ValueError(
            f"attack_kwargs must have length 1 or n_budgets={n_budgets}, got {len(attack_kwargs)}"
        )
    if len(attack_kwargs) == 1 and n_budgets > 1:
        return [dict(attack_kwargs[0]) for _ in range(n_budgets)]
    return [dict(kwargs) for kwargs in attack_kwargs]