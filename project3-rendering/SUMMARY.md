# SIGK - Projekt 3: Rendering

## Opis projektu


Celem projektu było stworzenie modelu sztucznej inteligencji, który realizuje model oświetlenia Phonga dla zadanej sceny 3D. Model na wejściu przyjmuje wektor parametrów sceny (pozycja obiektu, kolor rozproszenia, współczynnik połyskliwości, pozycja światła) i generuje odpowiadający obraz 128×128 px.

---

## Przygotowanie zbioru danych
Autor: Dominika Boguszewska


## Sieć dyfuzyjna
Autor: Dominika Boguszewska



## Sieć GAN
Autor: Filip Langiewicz


### Podział danych

| Zbiór      | Indeksy       | Liczba próbek |
|------------|---------------|---------------|
| Treningowy | 0 – 2399      | 2 400         |
| Testowy    | 2400 – 2999   | 600           |

### Normalizacja parametrów

Parametry sceny są normalizowane do zakresu `[−1, 1]`:

| Parametr wejściowy        | Transformacja                                          | Wymiar |
|---------------------------|--------------------------------------------------------|--------|
| Pozycja obiektu (x, y, z) | `t / TRANS_SCALE`                                      | 3      |
| Kolor rozproszenia (r,g,b)| `v * 2.0 − 1.0`                                        | 3      |
| Połyskliwość              | `((s − SHINE_MIN) / (SHINE_MAX − SHINE_MIN)) * 2 − 1` | 1      |
| Pozycja światła (rel.)    | `(light_pos − model_pos) / (2 * LIGHT_SCALE)`         | 3      |

> **Uwaga:** pozycja światła jest kodowana **relatywnie** względem pozycji obiektu, co ułatwia modelowi generalizację.

Wektor warunkujący ma wymiar **10** (`cond_dim = 10`).

---

### Architektura modeli

### Generator

Generator warunkowy buduje obraz od zera, startując od małej reprezentacji
i stopniowo zwiększając rozdzielczość. Na wejściu otrzymuje dwa wektory:
wektor szumu `z` (wymiar 8) wprowadzający losowość oraz wektor parametrów
sceny `c` (wymiar 10). Po konkatenacji powstaje wektor 18-wymiarowy.

Następnie sieć pięciokrotnie podwaja rozdzielczość za pomocą transpozycji
konwolucyjnych (ConvTranspose2d, stride=2), aż do osiągnięcia docelowych
128×128 px. Każdy blok upsamplingu zawiera BatchNorm i aktywację ReLU,
poza ostatnim, który stosuje Tanh (wyjście w zakresie `[−1, 1]`).

| Parametr          | Wartość       |
|-------------------|---------------|
| `noise_dim`       | 8             |
| `cond_dim`        | 10            |
| `features_g`      | 64            |
| Liczba parametrów | **7 107 459** |

Wagi inicjalizowane są metodą DCGAN: warstwy konwolucyjne z rozkładu
`N(0, 0.02)`, BatchNorm z `N(1, 0.02)` z biasem zerowym.

---

### Dyskryminator

Dyskryminator warunkowy ocenia, czy dany obraz jest prawdziwy, biorąc pod
uwagę kontekst sceny (`c`). Obraz i wektor warunkujący przetwarzane są
**oddzielnymi gałęziami**, których wyjścia łączone są dopiero w końcowej
głowicy klasyfikatora.

**Gałąź obrazu** redukuje przestrzennie wejście `3 × 128 × 128` przez
cztery warstwy konwolucyjne ze stride=2, uzyskując tensor `128 × 8 × 8`,
który spłaszczany jest do wektora o długości 8 192. Każda warstwa stosuje
LeakyReLU(0.2) oraz **spectral normalization**, która stabilizuje trening
ograniczając stałą Lipschitza dyskryminatora.

**Gałąź warunkowa** rzutuje wektor `c` (dim=10) na przestrzeń 128 za
pomocą jednej warstwy liniowej ze spectral normalization i LeakyReLU.

Oba wektory są konkatenowane (łączny wymiar: 8 320) i przekazywane do
głowicy, która przez dwie warstwy liniowe (z Dropout(0.3)) produkuje
jeden logit klasyfikacji.


| Parametr          | Wartość       |
|-------------------|---------------|
| `cond_dim`        | 10            |
| `features_d`      | 16            |
| Liczba parametrów | **8 695 937** |

---
### Funkcja straty

Projekt używa wariantu **LSGAN** (Least Squares GAN) z dodatkową stratą rekonstrukcyjną L1 ważoną maską pierwszoplanową.

### Strata dyskryminatora

Strata dyskryminatora (LSGAN) jest średnią z dwóch składników MSE:

| Składnik        | Wejście do D         | Etykieta docelowa | Waga |
|-----------------|----------------------|-------------------|------|
| Próbki prawdziwe | `D(x_real, c)`      | **0.9** (smoothing) | 0.5 |
| Próbki fałszywe | `D(x_fake, c)`      | **0.0**             | 0.5 |

```python
L_D = 0.5 * (MSE(D(x_real, c), 0.9) + MSE(D(x_fake, c), 0.0))
```

Wartość docelowa dla próbek prawdziwych wynosi **0.9** (label smoothing).

### Strata generatora

Generator minimalizuje sumę dwóch składników: straty adversarialnej
(chce „oszukać" dyskryminator) oraz ważonej straty rekonstrukcyjnej L1:

| Składnik | Opis | Waga |
|---|---|---|
| `L_adv` | `MSE(D(x_fake, c), 1.0)` – generator chce, by D uznał obraz za prawdziwy | 1.0 |
| `L_L1` | Masked L1 między obrazem wygenerowanym a referencyjnym | `λ_L1 = 200.0` |

``` python
L_G = L_adv + λ_L1 · L_masked_L1
= MSE(D(x_fake, c), 1.0) + 200.0 · L_masked_L1
```


### Ważona strata L1 (Masked L1)

Standardowa strata L1 traktuje wszystkie piksele równo, co przy czarnym tle
jest problematyczne – sieć mogłaby zignorować kulę i „zaoszczędzić" stratę
na tle (co też miało miesjce - sieć wpadała w takie lokalne minimum). Rozwiązaniem jest maska pierwszoplanowa wyznaczana na podstawie
jasności pikseli obrazu referencyjnego:

```python
mask = (real_01.abs().mean(dim=1, keepdim=True) > 0.05).float()
weights = 1.0 + (fg_weight - 1.0) * mask   # fg_weight = 50.0
L_masked_L1 = (weights * |x_fake - x_real|).mean()
```

Piksele jaśniejsze niż próg 0.05 (kula, oświetlenie) otrzymują wagę **50×**
większą niż tło. Dzięki temu model skupia się na poprawnym odwzorowaniu
kształtu i oświetlenia kuli, a nie na minimalizowaniu błędu na czarnym tle.

---

### Proces treningu

### Środowisko

| Parametr       | Wartość                    |
|----------------|----------------------------|
| Platforma      | Kaggle Notebook            |
| GPU            | **NVIDIA Tesla T4**        |
| Framework      | PyTorch (CUDA)             |
| Czas treningu  | ~58.7 min (3 520 s)        |
| Seed           | 42                         |

### Hiperparametry treningu

| Hiperparametr       | Wartość              |
|---------------------|----------------------|
| Liczba epok         | **300**              |
| Batch size          | **64**               |
| `lr_G`              | **2e-4**             |
| `lr_D`              | **3e-5**             |
| Betas (Adam)        | (0.5, 0.999)         |
| `lambda_l1`         | **200.0**            |
| `save_every`        | **3**       |
| Scheduler (G i D)   | CosineAnnealingLR, `T_max=300`, `eta_min=1e-5` |
| `num_workers`       | 0                    |
| `pin_memory`        | True (CUDA)          |

### Pętla treningowa

W każdej iteracji:

1. **Trening dyskryminatora:**
   - Generacja `fake_imgs = G(z, c)` z `torch.no_grad()`
   - Obliczenie `L_D` na podstawie `D(real, c)` i `D(fake, c)`
   - Krok optymalizatora dla D

2. **Trening generatora:**
   - Nowe losowanie `z`, generacja `fake_imgs = G(z, c)`
   - Obliczenie `L_G = L_adv + λ_L1 * L_masked_L1`
   - Krok optymalizatora dla G

3. Po każdej epoce: krok schedulera LR (`sched_G.step()`, `sched_D.step()`).

### Wybór najlepszego modelu

Co `save_every=3` epoki obliczana jest walidacyjna strata generatora. Checkpoint `G_best.pth` / `D_best.pth` zapisywany jest za każdym razem, gdy `val_G < best_val_G`.

**Najlepsza walidacyjna strata generatora: 30.7383** (epoka 240).

### Przebieg treningu

![GanLoss](img/loss_curve.png)

Strata treningowa generatora systematycznie spada przez cały trening (155.9 → 11.9),
co jest głównie zasługą poprawy składnika L1 (0.776 → 0.057). Strata adversarialna
stabilizuje się w przedziale 0.50–0.57 po ok. 100 epokach, co świadczy o utrzymaniu
równowagi między generatorem a dyskryminatorem.

Mimo nieuzyskania poprawy wartośći funkcji straty na zbiorze testowym po około 100 epokach, zdecydowano się skorzystać z modelu końcowego (z 300 epoki). Decyzja została podjęta na podstawie oceny wizualnej wyników, jakie uzyskiwały sieci w poszczególnych checkpointach.

---

### Ewaluacja

### Generacja obrazów testowych

Modelem `G_best.pth` wygenerowano 600 obrazów dla wszystkich próbek zbioru testowego. Wektor szumu ustawiony na **zero** (`z = 0`) w trybie ewaluacji - gwarantuje deterministyczne wyjście.

![GanTest](img/output.png)


### Metryki jakości

Obliczenia wykonano przy użyciu bibliotek: `flip_evaluator`, `lpips` (AlexNet v0.1), `skimage.metrics.structural_similarity`, `scipy.spatial.distance.directed_hausdorff` (krawędzie Canny z OpenCV).

| Metoda            | FLIP↓   | LPIPS↓  | SSIM↑   | Hausdorff↓ |
|-------------------|---------|---------|---------|------------|
| neural_renderer_gan   | **0.0125** | **0.1303** | **0.9650** | **19.63** |

### Interpretacja metryk

- **FLIP (0.0125):** Bardzo niska wartość (~1.25% mapy błędów) – błędy percepcyjne są minimalne. FLIP dobrze wykrywa lokalne różnice kolorystyczne i krawędziowe widoczne dla ludzkiego oka.
- **LPIPS (0.1303):** Umiarkowana wartość dystansu percepcyjnego (AlexNet). Sieć poprawnie odwzorowuje strukturę oświetlenia, lecz pewne subtelne różnice w rozbłyskach są jeszcze widoczne.
- **SSIM (0.9650):** Wysoka wartość podobieństwa strukturalnego – model odtwarza kształt, jasność i kontrast kuli.
- **Hausdorff (19.63 px):** Odległość Hausdorffa liczona na obrazach krawędziowych (Canny). Wartość 19.63 px wskazuje, że w nielicznych przypadkach krawędzie kuli lub rozbłysków mogą być lekko przesunięte (szczególnie przy ekstremalnych pozycjach obiektu).

> **Wnioski dot. metryk:** SSIM i FLIP dobrze oddają ogólną jakość obrazu. LPIPS jest bardziej czuły na różnice teksturalne (rozbłyski). Hausdorff może być zawyżony przez krawędzie szumowe lub drobne przesunięcia kuli – niekoniecznie odzwierciedla subiektywną jakość renderingu.



### Podsumowanie

Model osiągnął bardzo dobre wyniki jakościowe (SSIM=0.965, FLIP=0.0125), skutecznie aproksymując model oświetlenia Phonga dla scen z jedną kulą i punktowym źródłem światła. Mimo dobrych wyników metryk widać, że wygenerowanym obiektom brakuje czasem jedności strukturalnej i odpowiedniego położenia. Może to świadczyć o tym, że oceniane obiekty stanowią zbyt małą część sceny i sieć nie jest w stanie optymalnie wyjść z lokalnego minimum - generowania czarnego tła.
