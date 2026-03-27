# SIGK - Project 1: Image Modification

**Dataset:** DIV2K | **Framework:** PyTorch | **Language:** Python

Two image restoration tasks were implemented and evaluated against classical baselines using SNE, PSNR, SSIM and LPIPS metrics on the DIV2K validation set (3 598 patches of 256×256).

---

## 1. Super-Resolution

**Author:** Filip Langiewicz

U-Net-based model (**SRUNet**) with residual blocks, PixelShuffle upsampling and a bicubic residual skip connection. Trained on 32 000 patches, validated on 3 598 patches.

| Variant | Params    | Input LR | Output HR |
|---------|:---------:|:--------:|:---------:|
| SR ×4   | 2 289 923 | 64×64    | 256×256   |
| SR ×8   | 2 326 915 | 32×32    | 256×256   |

Training: Adam, L1 loss, StepLR (step=30, γ=0.5), 150 / 200 epochs, batch size 32.

### Results - ×4

| Method                 | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓      |
|------------------------|:-----------:|:----------:|:----------:|:----------:|
| cv2_resize_bicubic_x4  | 29.47       | 0.7554     | 0.3369     | 642.70     |
| **SRUNet_x4**          | **30.52**   | **0.7906** | **0.3153** | **538.52** |

### Results - ×8

| Method                 | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓        |
|------------------------|:-----------:|:----------:|:----------:|:------------:|
| cv2_resize_bicubic_x8  | 26.52       | 0.6301     | 0.4886     | 1159.15      |
| **SRUNet_x8**          | **27.13**   | **0.6565** | **0.4686** | **1043.12**  |

### Visual comparison - ×4
*Left to right: LR input (64×64) · SRUNet output · Bicubic · GT HR (256×256)*

![SR x4 comparison](figures/comparison_srx4.png)

### Visual comparison - ×8
*Left to right: LR input (32×32) · SRUNet output · Bicubic · GT HR (256×256)*

![SR x8 comparison](figures/comparison_srx8.png)

---

## 2. Denoising

**Author:** Dominika Boguszewska

Attention-based model (**RIDNet**) with dilated convolutions, residual blocks and channel attention (EAM modules). Trained to remove Gaussian noise at σ ∈ {0.01, 0.03}.

Training: Adam, MSE loss, StepLR (step=5, γ=0.5), 13 epochs (early stopping), batch size 4.

### Results

| Method                  | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓      |
|-------------------------|:-----------:|:----------:|:----------:|:----------:|
| Noisy input             | 33.65       | 0.8471     | 0.1509     | 378.45     |
| denoise_bilateral       | 34.07       | 0.9058     | 0.1800     | 381.21     |
| **RIDNet**              | **40.80**   | **0.9731** | **0.0938** | **78.32**  |

### Visual comparison
*Left to right: Clean · Noisy · RIDNet output · Bilateral*

![Denoising comparison](figures/denoising_comparison.png)

---

> `inf` PSNR values (uniform patches) were replaced with 42 before averaging.
