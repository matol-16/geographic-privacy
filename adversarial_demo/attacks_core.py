"""
Core base classes and utilities for attack training.

Consolidates common patterns from diffusion and encoder attacks:
- Multi-restart training loops
- Restart evaluation and metric selection
- Standardized result formatting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image

from adversarial_metrics import select_displacement_score
from adversarial_utils import (
    add_perturbation_to_image,
    conditional_preprocessing,
    run_paired_pipeline_with_shared_noise,
)


class RestartEvaluator:
    """Evaluate a single attack restart and track metrics."""

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
        batch_size: int = 256,
        cfg: float = 10.0,
        num_steps: Optional[int] = None,
        seed_offset: int = 0,
        device: str = "cuda",
    ):
        self.pipeline = pipeline
        self.source_image = source_image
        self.batch_size = batch_size
        self.cfg = cfg
        self.num_steps = num_steps
        self.seed_offset = seed_offset
        self.device = device

    def evaluate(
        self,
        delta: torch.Tensor,
        restart_idx: int = 0,
        base_seed: int = 1234,
    ) -> Dict[str, Any]:
        """
        Evaluate a perturbation by running it through the pipeline with shared noise.
        
        Returns dict with gps_source, gps_perturbed, traj_source, traj_perturbed, metrics.
        """
        perturbed_image = add_perturbation_to_image(self.source_image, delta.detach(), self.pipeline)
        
        eval_result = run_paired_pipeline_with_shared_noise(
            pipeline=self.pipeline,
            source_image=self.source_image,
            perturbed_image=perturbed_image,
            batch_size=self.batch_size,
            cfg=self.cfg,
            num_steps=self.num_steps,
            seed=int(base_seed) + restart_idx + self.seed_offset,
            device=self.device,
        )
        return eval_result


class RestartManager:
    """Manage multi-restart attack training with best-restart selection."""

    def __init__(
        self,
        num_restarts: int = 1,
        selection_metric: str = "mean_step_displacement",
        print_results: bool = True,
    ):
        if int(num_restarts) < 1:
            raise ValueError("num_restarts must be >= 1")
        
        self.num_restarts = int(num_restarts)
        self.selection_metric = selection_metric
        self.print_results = print_results
        
        # Tracking
        self.best_delta: Optional[torch.Tensor] = None
        self.best_history: List[float] = []
        self.best_restart: Optional[int] = None
        self.best_score = -float("inf")
        self.best_metrics: Optional[Dict[str, float]] = None
        self.restart_summaries: List[Dict[str, Any]] = []

    def update_best(
        self,
        restart_idx: int,
        delta: torch.Tensor,
        history: List[float],
        metrics: Dict[str, float],
        loss_value: float = None,
    ) -> None:
        """
        Update best restart tracking if this restart improved the score.
        
        Args:
            restart_idx: Index of this restart (0-based)
            delta: The learned perturbation tensor
            history: Training loss history for this restart
            metrics: Displacement metrics dict with mean_step_displacement and final_step_displacement
            loss_value: Optional final loss for display (uses history[-1] if not provided)
        """
        mean_step_disp = metrics["mean_step_displacement"]
        final_step_disp = metrics["final_step_displacement"]
        
        score = select_displacement_score(
            mean_step_disp=mean_step_disp,
            final_step_disp=final_step_disp,
            metric_name=self.selection_metric,
        )
        
        final_loss = float(loss_value) if loss_value is not None else (float(history[-1]) if history else float("inf"))
        min_loss = float(min(history)) if history else float("inf")
        
        summary = {
            "restart": restart_idx,
            "mean_step_displacement": float(mean_step_disp),
            "final_step_displacement": float(final_step_disp),
            "score": float(score),
            "final_loss": final_loss,
            "min_loss": min_loss,
        }
        self.restart_summaries.append(summary)
        
        if self.print_results:
            print(
                f"[restart {restart_idx + 1}/{self.num_restarts}] "
                f"final_loss={summary['final_loss']:.6f}, "
                f"min_loss={summary['min_loss']:.6f}, "
                f"mean_step_disp={summary['mean_step_displacement']:.6f}, "
                f"final_step_disp={summary['final_step_displacement']:.6f}, "
                f"selection_score={summary['score']:.6f}"
            )
        
        if score > self.best_score:
            self.best_score = float(score)
            self.best_delta = delta.detach().clone()
            self.best_history = list(history)
            self.best_restart = restart_idx
            self.best_metrics = dict(metrics)

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize restart selection and return summary.
        
        Returns dict with best_delta, best_restart, best_score, restart_summaries.
        """
        if self.best_delta is None or self.best_restart is None:
            raise RuntimeError("No restart produced a valid perturbation")
        
        if self.print_results and self.num_restarts > 1:
            print(
                f"Selected restart {self.best_restart + 1}/{self.num_restarts} using "
                f"{self.selection_metric} (score={self.best_score:.6f})"
            )
        
        return {
            "best_delta": self.best_delta,
            "best_history": self.best_history,
            "best_restart": int(self.best_restart),
            "best_score": float(self.best_score),
            "best_metrics": self.best_metrics,
            "restart_summaries": self.restart_summaries,
        }


class AttackResult:
    """Standardized attack result formatting."""

    @staticmethod
    def format_result(
        attack_type: str,
        delta: torch.Tensor,
        source_tensor: torch.Tensor,
        history: List[float],
        best_metrics: Optional[Dict[str, float]] = None,
        best_restart: int = 0,
        restart_summaries: Optional[List[Dict[str, Any]]] = None,
        attack_mode: str = "untargeted",
        z_source: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format attack results into standardized dict.
        
        Args:
            attack_type: "encoder" or "diffusion"
            delta: Learned perturbation tensor
            source_tensor: Source image tensor
            history: Training loss history
            best_metrics: Best restart metrics dict
            best_restart: Index of best restart
            restart_summaries: List of restart summary dicts
            attack_mode: "targeted" or "untargeted"
            z_source: Source embedding (for encoder attacks)
            z_target: Target embedding (for encoder attacks)
            config: Attack configuration dict
            
        Returns:
            Standardized result dict
        """
        final_loss = float(history[-1]) if history else float("inf")
        min_loss = float(min(history)) if history else float("inf")
        
        result = {
            "attack_type": attack_type,
            "delta": delta,
            "source_tensor": source_tensor,
            "history": history,
            "final_loss": final_loss,
            "min_loss": min_loss,
            "best_metrics": best_metrics,
            "best_restart": int(best_restart),
            "restart_summaries": restart_summaries or [],
        }
        
        # Add encoder-specific fields
        if attack_mode is not None:
            result["attack_mode"] = attack_mode
        if z_source is not None:
            result["z_source"] = z_source
        if z_target is not None:
            result["z_target"] = z_target
        
        # Add config if provided
        if config is not None:
            result["config"] = config
        
        return result


class AttackBase(ABC):
    """
    Base class for attack implementations.
    
    Provides common attack lifecycle:
    1. prepare_inputs(source_image) -> source_tensor
    2. Loop over restarts:
        3. initialize_delta() -> delta
        4. Loop over steps:
            5. compute_loss(delta, step) -> loss
            6. optimizer step
        7. evaluate_restart(delta) -> metrics
        8. update best tracking
    9. finalize() -> result dict
    """

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
        num_restarts: int = 1,
        selection_metric: str = "mean_step_displacement",
        print_results: bool = True,
        device: str = "cuda",
    ):
        self.pipeline = pipeline
        self.source_image = source_image
        self.device = device
        
        self.source_tensor = conditional_preprocessing(source_image, pipeline, device=device)
        self.restart_manager = RestartManager(
            num_restarts=num_restarts,
            selection_metric=selection_metric,
            print_results=print_results,
        )

    @abstractmethod
    def initialize_delta(self, restart_idx: int) -> torch.Tensor:
        """Initialize perturbation for this restart. Must set requires_grad=True."""
        pass

    @abstractmethod
    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Execute one optimization step.
        
        Returns:
            Loss value for this step
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return attack configuration dict for result."""
        pass

    def train_restart(
        self,
        restart_idx: int,
        n_steps: int,
        optimizer_fn: Callable[[list], torch.optim.Optimizer],
        show_progress: bool = True,
        early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
    ) -> tuple[torch.Tensor, List[float]]:
        """
        Train a single restart with optional early stopping.
        
        Args:
            restart_idx: Index of this restart (0-based)
            n_steps: Number of optimization steps
            optimizer_fn: Function that takes [delta] and returns optimizer
            show_progress: Whether to show progress bar
            early_stopping_patience: Stop if loss doesn't improve for N steps (0=disabled)
            
        Returns:
            (delta, history) where history is list of loss values
        """
        import tqdm
        
        delta = self.initialize_delta(restart_idx)
        optimizer = optimizer_fn([delta])
        
        history: List[float] = []
        best_loss = float('inf')
        patience_counter = 0
        
        step_iter = range(int(n_steps))
        pbar = None
        if show_progress:
            pbar = tqdm.trange(
                int(n_steps),
                desc=f"Attack training (restart {restart_idx + 1}/{self.restart_manager.num_restarts})",
            )
            step_iter = pbar
        
        for step in step_iter:
            loss = self.run_step(delta, step, optimizer)
            history.append(float(loss))
            
            # Early stopping logic
            if early_stopping_patience > 0:
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if pbar is not None:
                            pbar.close()
                        if show_progress:
                            print(f"  → Early stopping at step {step+1}/{n_steps} (loss plateau)")
                        return delta, history
            
            if pbar is not None and (step + 1) % 20 == 0:  # Log every 20 steps
                pbar.set_postfix(loss=f"{loss:.6f}")
        
        if pbar is not None:
            pbar.close()
        return delta, history

    def evaluate_restart(
        self,
        delta: torch.Tensor,
        restart_idx: int,
        batch_size: int = 256,
        cfg: float = 10.0,
        num_steps: Optional[int] = None,
        seed: int = 1234,
    ) -> Dict[str, Any]:
        """Evaluate restart with shared noise."""
        evaluator = RestartEvaluator(
            pipeline=self.pipeline,
            source_image=self.source_image,
            batch_size=batch_size,
            cfg=cfg,
            num_steps=num_steps,
            device=self.device,
        )
        return evaluator.evaluate(delta, restart_idx=restart_idx, base_seed=seed)

    def finalize_result(
        self,
        attack_type: str,
        attack_mode: str = "untargeted",
        z_source: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Finalize and format attack result.
        
        Args:
            attack_type: "encoder" or "diffusion"
            attack_mode: "targeted" or "untargeted"
            z_source: Source embedding (encoder only)
            z_target: Target embedding (encoder only)
            
        Returns:
            Standardized result dict
        """
        restart_summary = self.restart_manager.finalize()
        
        return AttackResult.format_result(
            attack_type=attack_type,
            delta=restart_summary["best_delta"],
            source_tensor=self.source_tensor,
            history=restart_summary["best_history"],
            best_metrics=restart_summary["best_metrics"],
            best_restart=restart_summary["best_restart"],
            restart_summaries=restart_summary["restart_summaries"],
            attack_mode=attack_mode,
            z_source=z_source,
            z_target=z_target,
            config=self.get_config(),
        )
