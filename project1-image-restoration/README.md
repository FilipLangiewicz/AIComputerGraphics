# Wyniki

## Zwiększanie rozdzielczości (Super-Resolution)

### Skala ×4

| Metoda                     | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓      |
|----------------------------|:-----------:|:----------:|:----------:|:----------:|
| cv2_resize_bicubic_x4     | 29.47       | 0.7554     | 0.3369     | 642.70     |
| **SRUNet_x4** (nasz model) | **30.52**   | **0.7906** | **0.3153** | **538.52** |

### Skala ×8

| Metoda                     | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓        |
|----------------------------|:-----------:|:----------:|:----------:|:------------:|
| cv2_resize_bicubic_x8     | 26.52       | 0.6301     | 0.4886     | 1159.15      |
| **SRUNet_x8** (nasz model) | **27.13**   | **0.6565** | **0.4686** | **1043.12**  |

## Odszumianie (Denoising)

| Metoda                       | PSNR (dB) ↑ |   SSIM ↑   |  LPIPS ↓   |   SNE ↓    |
|------------------------------|:-----------:|:----------:|:----------:|:----------:|
| Pary czysty-zaszumiony obraz |   33.6471   |   0.8471   |   0.1509   |  378.4478  |
| denoise_bilateral (skimage)  |   34.0724   |   0.9058   |    0.18    |  381.2101  |
| RIDNet (nasz model)          | **40.8019** | **0.9731** | **0.0938** | **78.321** |
