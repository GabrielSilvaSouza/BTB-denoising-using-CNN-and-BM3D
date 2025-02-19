import torch
import numpy as np
from bm3d import bm3d

def denoiser_bm3d(noisy_img_tensor):
    """Applies BM3D denoising to a PyTorch tensor image."""
    # Convert PyTorch tensor to NumPy array
    noisy_img = noisy_img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    
    # Apply BM3D denoising
    denoised_img = bm3d(noisy_img, 0.2)
    
    # Convert back to PyTorch tensor
    return torch.from_numpy(denoised_img).permute(2, 0, 1).unsqueeze(0).float()

def denoiser_ffcnn(noisy_img_tensor, model_path):
    """Applies FFCNN denoising using a pre-trained model."""
    from ffcnn import FFCNN 
    
    model = FFCNN()
    model.load_state_dict(torch.load(model_path, map_location="gpu"))
    model.eval()

    with torch.no_grad():
        denoised_img_tensor = model(noisy_img_tensor)
    
    return denoised_img_tensor
