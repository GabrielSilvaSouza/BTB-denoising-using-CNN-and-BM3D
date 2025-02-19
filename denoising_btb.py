import torch
import numpy as np

def fast_iterative_denoising_btb(y, f, mu_sequence, T, delta):
    """
    Fast Iterative Denoising Algorithm (BTB)

    Parameters:
        y: Input image (noisy, as a PyTorch tensor)
        f: Denoiser function
        mu_sequence: Sequence of step sizes
        T: Maximum number of iterations
        delta: Convergence threshold

    Returns:
        x_denoised: Denoised image
    """
    x_prev = y.clone()  # Use .clone() instead of .copy() for PyTorch tensors
    
    for t in range(T):
        # Apply denoiser function to previous denoised image
        x_next_tensor = f(x_prev)

        # Convert tensor back to numpy array
        x_next = x_next_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
        
        # Convert NumPy array back to PyTorch tensor for consistent processing
        x_next_tensor = torch.from_numpy(x_next).permute(2, 0, 1).unsqueeze(0).float()

        # Update current denoised image using hybrid Banach contraction principle
        x_current = (1 - mu_sequence[t]) * x_prev + mu_sequence[t] * x_next_tensor
        
        # Check convergence criterion
        if torch.norm(x_current - x_prev) < delta:
            break  # Stop iteration if convergence criterion is met
        
        x_prev = x_current  # Update previous denoised image
        
    return x_current


