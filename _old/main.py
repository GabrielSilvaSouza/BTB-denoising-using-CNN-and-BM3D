
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d
from btb import fast_iterative_denoising_btb
from denoisers import denoiser_bm3d, denoiser_ffcnn 



def prompt():
    input_model = input("""Choose the model to use in Fast Interative Denoising: \n 
                        1. FFCNN \n 
                        2. BM3D \n 
                        Enter the number of the model: """)
    img_path = input("Enter the path of the noisy image: ")
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    noisy_img = image.astype(np.float32) / 255.0
    noisy_img_tensor = torch.from_numpy(noisy_img).permute(2, 0, 1).unsqueeze(0).float()
    
    return input_model, noisy_img, noisy_img_tensor


def model_selected(input_model):
    if input_model == "1":
        model_path = input("Select the path of your .pth file: ")
        denoiser = lambda img: denoiser_ffcnn(img, model_path)  

    elif input_model == "2":
        denoiser = denoiser_bm3d 

    return denoiser


def denoising_results(noisy_img_tensor, denoiser):
    mu_sequence = np.linspace(0.1, 0.2, 10)
    T = 10
    delta = 1e-4

    denoised_image = fast_iterative_denoising_btb(noisy_img_tensor, denoiser, mu_sequence, T, delta)
    denoised_image_np = denoised_image.squeeze(0).permute(1, 2, 0).detach().numpy()

    return denoised_image_np

def plot_results(noisy_img, denoised_image_np):
    if denoised_image_np.shape[-1] == 3:
        denoised_image_np = np.clip(denoised_image_np, 0, 1) 
    elif len(denoised_image_np.shape) == 2:
        cmap = 'gray' 
    else:
        cmap = None 

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image_np, cmap="gray")
    plt.title('Denoised Image (BTB)')
    plt.axis('off')

    plt.savefig("denoising_results.png", dpi=300, bbox_inches='tight')
    plt.close()  

    print("Plot saved as 'denoising_results.png'")


if __name__ == "__main__":
    input_model, noisy_img, noisy_img_tensor = prompt()
    denoiser = model_selected(input_model)
    denoised_image_np = denoising_results(noisy_img_tensor, denoiser)
    plot_results(noisy_img, denoised_image_np)