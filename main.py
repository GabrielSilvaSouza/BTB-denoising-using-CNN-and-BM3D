import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d
from btb import fast_iterative_denoising_btb
from denoisers import denoiser_bm3d, denoiser_ffcnn


class DenoisingPipeline:
    def __init__(self):
        self.input_model = None
        self.noisy_img = None
        self.noisy_img_tensor = None
        self.denoiser = None
        self.denoised_image_np = None

    def prompt(self):
        self.input_model = input("""Choose the model to use in Fast Iterative Denoising: \n 
                        1. FFCNN \n 
                        2. BM3D \n 
                        Enter the number of the model: """)
        img_path = input("\n Enter the path of the noisy image: ")
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        self.noisy_img = image.astype(np.float32) / 255.0
        self.noisy_img_tensor = torch.from_numpy(self.noisy_img).permute(2, 0, 1).unsqueeze(0).float()
    
    def select_model(self):
        if self.input_model == "1":
            model_path = input("Select the path of your .pth file: ")
            self.denoiser = lambda img: denoiser_ffcnn(img, model_path)
        elif self.input_model == "2":
            self.denoiser = denoiser_bm3d

    def denoise(self):
        mu_sequence = np.linspace(0.1, 0.2, 10)
        T = 10
        delta = 1e-4

        denoised_image = fast_iterative_denoising_btb(self.noisy_img_tensor, self.denoiser, mu_sequence, T, delta)
        self.denoised_image_np = denoised_image.squeeze(0).permute(1, 2, 0).detach().numpy()
    
    def plot_results(self):
        if self.denoised_image_np.shape[-1] == 3:
            self.denoised_image_np = np.clip(self.denoised_image_np, 0, 1)
            cmap = None
        elif len(self.denoised_image_np.shape) == 2:
            cmap = 'gray'
        else:
            cmap = None

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.noisy_img, cmap='gray')
        plt.title('Noisy Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.denoised_image_np, cmap=cmap)
        plt.title('Denoised Image (BTB)')
        plt.axis('off')

        plt.savefig("denoising_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Plot saved as 'denoising_results.png'")

    def run(self):
        self.prompt()
        self.select_model()
        self.denoise()
        self.plot_results()


if __name__ == "__main__":
    pipeline = DenoisingPipeline()
    pipeline.run()
