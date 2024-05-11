import cv2
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from skimage import img_as_float, io, restoration

def fast_iterative_denoising_btb(y, f, mu_sequence, T, delta):
    """
    Fast Iterative Denoising Algorithm (BTB)

    Parameters:
        y: Input image (noisy)
        f: Denoiser function
        mu_sequence: Sequence of step sizes
        T: Maximum number of iterations
        delta: Convergence threshold

    Returns:
        x_denoised: Denoised image
    """
    x_prev = y.copy()  # Initialize previous denoised image as noisy image

    for t in range(T):
        x_next = f(x_prev)  # Apply denoiser function to previous denoised image
        
        # Update current denoised image using hybrid Banach contraction principle
        x_current = (1 - mu_sequence[t]) * x_prev + mu_sequence[t] * x_next
        
        # Check convergence criterion
        if cp.linalg.norm(x_current - x_prev) < delta:
            break  # Stop iteration if convergence criterion is met
        
        x_prev = x_current  # Update previous denoised image
        
    return x_current

def add_gaussian_noise(image, sigma):
    """
    Add Gaussian noise to an image

    Parameters:
        image: Input image
        sigma: Standard deviation of Gaussian noise

    Returns:
        image_noisy: Noisy image
    """
    noise = np.random.normal(0, sigma, image.shape)  # Generate Gaussian noise
    image_noisy = image + noise  # Add noise to image
    image_noisy = np.clip(image_noisy, 0, 1)  # Clip noisy image to [0, 1]

    return image_noisy

img_path = "d:\\synthetic\\seismic_data__I_3.png"

image = cv2.imread(img_path, cv2.IMREAD_COLOR)
image = cp.array(image)

noise = add_gaussian_noise(image.get(), 0.1)  # Get NumPy array for noise

noisy_image = image + noise  # Add noise directly to CuPy array
noisy_image = img_as_float(noisy_image.get())

def denoiser_nlm(image):
    return cp.array(restoration.denoise_nl_means(image, h=0.1, sigma=0.1, fast_mode=True, patch_size=5, patch_distance=6))

mu_sequence = cp.linspace(0.1, 0.1, 8)

T = 8
delta = 1e-4
denoised_image = fast_iterative_denoising_btb(noisy_image, denoiser_nlm, mu_sequence, T, delta)

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(noisy_image, cmap='gray')  # No need to use .get() here
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(denoised_image.get(), cmap='gray')  # Use .get() to convert CuPy array to NumPy for display
plt.title('Denoised Image (BTB)')
plt.axis('off')

plt.show()
