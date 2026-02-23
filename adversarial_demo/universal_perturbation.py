from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange

from plonk.data.data import Baseline


def get_yfcc4k_dataloader(
    dataset_root,
    image_size=336,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = Baseline(path=str(Path(dataset_root)), which="yfcc4k", transforms=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
    )
    return dataloader


def train_universal_encoder_perturbation(
    pipeline,
    dataloader,
    target_image,
    device,
    num_steps=200,
    lr=5e-2,
    eps_max=1.0,
    recon_weight=0.1,
    show_progress=True,
):
    emb_model = pipeline.cond_preprocessing.emb_model
    emb_model.eval()
    pipeline.network.eval().requires_grad_(False)

    for param in emb_model.parameters():
        param.requires_grad_(False)

    first_batch = next(iter(dataloader))
    first_images = first_batch["img"]
    delta = torch.zeros(
        (1, first_images.shape[1], first_images.shape[2], first_images.shape[3]),
        device=device,
        requires_grad=True,
    )

    with torch.no_grad():
        if isinstance(target_image, torch.Tensor):
            if target_image.ndim == 4:
                target_img = transforms.ToPILImage()(target_image[0].detach().cpu())
            else:
                target_img = transforms.ToPILImage()(target_image.detach().cpu())
        elif isinstance(target_image, Image.Image):
            target_img = target_image
        elif isinstance(target_image, list) and len(target_image) > 0:
            target_img = target_image[0]
        else:
            raise ValueError("target_image must be a PIL image, tensor, or non-empty list of images")

        target_batch = {"img": [target_img]}
        target_batch = pipeline.cond_preprocessing(target_batch)
        z_target = target_batch["emb"].to(device, non_blocking=True)

    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    history = []
    batch_iterator = iter(dataloader)
    step_iterator = trange(num_steps, desc="Universal perturbation", disable=not show_progress)
    for _ in step_iterator:
        optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader)
            batch = next(batch_iterator)

        step_images = batch["img"].to(device, non_blocking=True)
        perturbed = step_images + delta

        z_adv = emb_model(perturbed)
        loss_embed = torch.nn.functional.mse_loss(z_adv, z_target.expand_as(z_adv))
        loss_recon = torch.nn.functional.l1_loss(perturbed, step_images)
        loss = loss_embed + recon_weight * loss_recon

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            delta.clamp_(-eps_max, eps_max)

        history.append(
            {
                "loss": float(loss.item()),
                "loss_embed": float(loss_embed.item()),
                "loss_recon": float(loss_recon.item()),
            }
        )

        if show_progress:
            step_iterator.set_postfix(
                bs=step_images.shape[0],
                loss=f"{loss.item():.6f}",
                embed=f"{loss_embed.item():.6f}",
                recon=f"{loss_recon.item():.6f}",
            )

    return delta.detach(), history
