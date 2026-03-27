# SIGK - Artificial Intelligence in Computer Graphics

**Course:** Sztuczna Inteligencja w Grafice Komputerowej  
**Framework:** PyTorch | **Language:** Python

---

## Table of Contents

- [Project 1 - Super-Resolution & Denoising](#project-1--super-resolution--denoising)
- [Project 2 - TBD](#project-2--tbd)
- [Project 3 - TBD](#project-3--tbd)
- [Project 4 - TBD](#project-4--tbd)
- [Project 5 - TBD](#project-5--tbd)

---

## Project 1 - Super-Resolution & Denoising

> Full report: [`project1/SUMMARY.md`](project1-image-restoration/SUMMARY.md)

### Super-Resolution (SRUNet)
U-Net with residual blocks and PixelShuffle upsampling. Reconstructs HR images (256×256) from LR inputs at ×4 (64×64) and ×8 (32×32) scale.

| Method                | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----------------------|:------:|:------:|:-------:|
| Bicubic ×4            | 29.47  | 0.7554 | 0.3369  |
| **SRUNet ×4**         | **30.52** | **0.7906** | **0.3153** |
| Bicubic ×8            | 26.52  | 0.6301 | 0.4886  |
| **SRUNet ×8**         | **27.13** | **0.6565** | **0.4686** |

### Denoising (RIDNet)
Residual attention network with dilated convolutions and channel attention (EAM). Removes Gaussian noise at σ ∈ {0.01, 0.03}.

| Method             | PSNR ↑    | SSIM ↑    | LPIPS ↓   |
|--------------------|:---------:|:---------:|:---------:|
| Noisy input        | 33.65     | 0.8471    | 0.1509    |
| Bilateral filter   | 34.07     | 0.9058    | 0.1800    |
| **RIDNet**         | **40.80** | **0.9731**| **0.0938**|

---

## Project 2 - TBD

> *Coming soon.*

---

## Project 3 - TBD

> *Coming soon.*

---

## Project 4 - TBD

> *Coming soon.*

---

## Project 5 - TBD

> *Coming soon.*
