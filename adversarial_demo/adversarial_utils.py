
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

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