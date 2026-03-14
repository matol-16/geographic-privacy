# Then, we can evaluate the trained perturbation by adding it to the source image, and running the pipeline forward with the perturbed image as conditioning input.
# We especially want to visualize the diffusion trajectory. This is possible via the "return_trajectories" argument of the ddim_sampler. 
#This requires changing the _call method of the PlonkPipeline to return trajectories when this argument is set to True.

from plonk.pipe import PlonkPipeline
import torch
import numpy as np
from PIL import Image

class PlonkPipelineTrajectory(PlonkPipeline):
    def __call__(
        self,
        images,
        batch_size=None,
        x_N=None,
        num_steps=None,
        scheduler=None,
        cfg=0,
        generator=None,
        return_trajectories=False
    ):
        """
        Extends the __call__ method of the PlonkPipeline by allowing to track trajectories.
        The rest of the code is identical
        """
        # Set up batch size and initial noise
        shape = [3]
        if not isinstance(images, list):
            images = [images]
        if x_N is None:
            if batch_size is None:
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
            x_N = torch.randn(
                batch_size, *shape, device=self.device, generator=generator
            )
        else:
            x_N = x_N.to(self.device)
            if x_N.ndim == 3:
                x_N = x_N.unsqueeze(0)
            batch_size = x_N.shape[0]

        # Set up batch with conditioning
        batch = {"y": x_N}
        batch["img"] = images
        batch = self.cond_preprocessing(batch)
        if len(images) > 1:
            assert len(images) == batch_size
        else:
            batch["emb"] = batch["emb"].repeat(batch_size, 1)

        # Use default sampler/scheduler if not provided
        sampler = self.sampler
        if scheduler is None:
            scheduler = self.scheduler
        # Sample from model
        traj = None
        if num_steps is None:
            if return_trajectories:
                output, traj = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
            else:
                output = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
        else:
            if return_trajectories:
                output, traj = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    num_steps=num_steps,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
            else:
                output = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    num_steps=num_steps,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )

        # Apply postprocessing to final samples
        output = self.postprocessing(output)
        output = np.degrees(output.detach().cpu().numpy())
        
        # Apply postprocessing to each trajectory step if requested.
        # Keep the same conversion path as `output`: cartesian -> radians -> degrees.
        if traj is not None:
            if isinstance(traj, torch.Tensor):
                traj_steps = [traj[i] for i in range(traj.shape[0])]
            else:
                traj_steps = list(traj)

            traj_gps = []
            for batch_traj in traj_steps:
                if not isinstance(batch_traj, torch.Tensor):
                    batch_traj = torch.as_tensor(batch_traj, device=self.device)
                batch_traj_gps = self.postprocessing(batch_traj)
                batch_traj_gps = np.degrees(batch_traj_gps.detach().cpu().numpy())
                traj_gps.append(batch_traj_gps)
            traj = np.stack(traj_gps, axis=0)
        
        if return_trajectories and traj is not None:
            return output, traj
        else:
            return output
