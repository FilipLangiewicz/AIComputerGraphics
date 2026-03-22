# SIGK - Projekt 1: Modyfikacja obrazów

## Opis projektu

Celem projektu było zaprojektowanie i wytrenowanie sieci neuronowej do realizacji wybranych zadań modyfikacji obrazów, a następnie porównanie jej jakości z metodami bazowymi przy użyciu metryk SNE, PSNR, SSIM oraz LPIPS.

Jako zbiór danych wykorzystano DIV2K - wysokiej jakości zbiór 900 obrazów w rozdzielczości HD i powyżej, powszechnie stosowany w tego typu zagadnieniach.

W ramach projektu zdecydowaliśmy się zrealizować dwa zadania:

1. Zwiększanie rozdzielczości (Super-Resolution) - rekonstrukcja obrazów HR (256×256) z obrazów LR o rozdzielczościach 64×64 (skala x4) oraz 32×32 (skala x8).

2. Odszumianie (Denoising) - usuwanie szumu gaussowskiego (o współczynnikach σ = 0.01 oraz σ = 0.03) z obrazów.

---

## 1. Zwiększanie rozdzielczości (Super-Resolution)
Autor: Filip Langiewicz
### 1.1 Przygotowanie danych

Obserwacje ze zbioru treningowego DIV2K pocięto na patche 256×256 (z każdego obrazka w zbiorze treningowym uzyskano 40 losowych patchy), a następnie przeskalowano w dół przy użyciu `cv2.resize` z interpolacją `cv2.INTER_AREA`, uzyskując wersje LR o rozdzielczościach **64×64** (skala x4) oraz **32×32** (skala x8). Dodatkowo przeprowadzono augmentację uzyskanych patchy poprzez losowe odwracanie wycinków w pionie (z prawdopodobieństwem 1/2) oraz w poziomie (z prawdopodobieństwem 1/2).

Patche w zbiorze walidacyjnym uzyskano poprzez pocięcie obrazów ze zbioru walidacyjnego DIV2K na nienachodzące na siebie fragmenty o rozmiarze 256×256.

| Zbiór     | Liczba próbek |
|-----------|:-------------:|
| Treningowy   | 32 000        |
| Walidacyjny | 3 598         |

---

### 1.2 Architektura - SRUNet

Zaproponowano architekturę **SRUNet** - sieć w stylu U-Net z blokami resztkowymi (`ResBlock`) i upsamplingiem przez `PixelShuffle`. Każdy blok enkodera zmniejsza rozdzielczość przez konwolucję ze `stride=2`, a dekoder odbudowuje przestrzenną rozdzielczość przez `PixelShuffle`. Końcowe głowice SR (`sr_up`) podnoszą rozdzielczość do docelowej. Do wyjścia dodawany jest residual skip z wejścia LR interpolowanego bikubicznie.


| Wariant | `base_ch` | `n_bottleneck` | Parametry   | Wejście LR | Wyjście HR |
|---------|:---------:|:--------------:|:-----------:|:----------:|:----------:|
| SR ×4   | 32        | 4              | 2 289 923   | 64×64      | 256×256    |
| SR ×8   | 32        | 4              | 2 326 915   | 32×32      | 256×256    |

---

### 1.3 Trening

| Parametr       | SR ×4                   | SR ×8                   |
|----------------|:-----------------------:|:-----------------------:|
| Epoki          | 150                     | 200                     |
| Batch size     | 32                      | 32                      |
| Optymalizator  | Adam                    | Adam                    |
| LR startowe    | 1e-4                    | 1e-4                    |
| Scheduler      | StepLR (step=30, γ=0.5) | StepLR (step=30, γ=0.5) |
| Funkcja straty | L1                      | L1                      |
| Walidacja co   | 10 epok                 | 10 epok                 |
| Checkpoint     | best PSNR on valid      | best PSNR on valid      |

Sieć dla skali ×4 trenowała się poprawnie do około 100 epoki. Wówczas uzyskano najlepszy wynik metryki PSNR dla zbioru walidacyjnego. W następnych epokach widoczny jest efekt przeuczenia sieci - maleje błąd na zbiorze treningowym, natomiast powiększa się błąd dla zbioru walidacyjnego. W związku z tym postanowiono jako najlepsze rozwiązanie wybrać sieć z wagami z 100 epoki.

Analogiczna sytuacja miała miejsca w przypadku skali ×8. Sieć zaczęła się przeuczać już w około 30 epoce, w związku z tym wykorzystano wagi z tego momentu treningu.

---

### 1.4 Wyniki - skala ×4

| Metoda                     | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓      |
|----------------------------|:-----------:|:----------:|:----------:|:----------:|
| cv2_resize_bicubic_x4     | 29.47       | 0.7554     | 0.3369     | 642.70     |
| **SRUNet_x4** (nasz model) | **30.52**   | **0.7906** | **0.3153** | **538.52** |

### 1.5 Wyniki - skala ×8

| Metoda                     | PSNR (dB) ↑ | SSIM ↑     | LPIPS ↓    | SNE ↓        |
|----------------------------|:-----------:|:----------:|:----------:|:------------:|
| cv2_resize_bicubic_x8     | 26.52       | 0.6301     | 0.4886     | 1159.15      |
| **SRUNet_x8** (nasz model) | **27.13**   | **0.6565** | **0.4686** | **1043.12**  |

> Metryki obliczone na pełnym zbiorze walidacyjnym DIV2K (3598 patchy 256×256).  
> Wartości `inf` na potrzeby obliczenia metryki PSNR zastąpiono wartością 42 przed uśrednieniem.

---

### 1.6 Porównanie wizualne - SR ×4

*Od lewej: LR wejście (64×64), wyjście SRUNet (256×256), Bicubic (256×256), GT HR (256×256)*

![Porównanie SR x4](figures/comparison_srx4.png)

Jak widać na obrazkach różnica jest dostrzegalna gołym okiem. Sieć nauczyła się niektórych kształtów i konturów, co daje lepsze obrazki wyjściowe niż zwykła interpolacja. Zdjęcia z rozdzielczością czterokrotnie zwiększoną przez sieć są dużo mniej romzyte i zawierają więcej ostrych krawędzi.

---

### 1.7 Porównanie wizualne - SR ×8

*Od lewej: LR wejście (32×32), wyjście SRUNet (256×256), Bicubic (256×256), GT HR (256×256)*

![Porównanie SR x8](figures/comparison_srx8.png)

W tym przypadku sieć również poprawiła rozdzielczość zdjęć lepiej niż bazowa metoda z biblioteki cv2. Efekty są jednak mniej widoczne niż przy czterokrotnym zwiększaniu rozdzielczości. Mimo tego, dalej dostrzegalna jest różnica pomiędzy interpolacją a naszą siecią.

---

### 1.8 Wnioski

- SRUNet poprawia wszystkie cztery metryki względem bazowej interpolacji bikubicznej dla obu skal.
- Dla ×4 poprawa PSNR wynosi **+1.05 dB**, SSIM **+0.0352**, LPIPS **-0.0216** - model lepiej rekonstruuje drobne tekstury i krawędzie.
- Dla ×8 poprawa jest mniejsza (**+0.61 dB** PSNR), co jest spodziewane - 8-krotna utrata rozdzielczości niesie ze sobą znacznie mniej informacji wejściowych.
- Połączenia resztkowe z interpolacją bikubiczną stabilizują trening od pierwszych epok i zapewniają sensowne wyjście bazowe nawet bez uczenia.
- Wartości metryki LPIPS sugerują, że model percepcyjnie rekonstruuje globalną strukturę obrazu, choć drobne detale pozostają trudne do odtworzenia.

---

## 2. Odszumianie (Denoising)
Autor: Dominika Boguszewska

---
