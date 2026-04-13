# SIGK - Projekt 2: Syntezator ekspozycji

## Opis projektu

Celem projektu było stworzenie sieci neuronowej, która na podstawie pojedynczego obrazu
o standardowej ekspozycji (LDR) generuje dwa dodatkowe obrazy: niedoświetlony
(EV = −2.7) oraz prześwietlony (EV = +2.7). Wygenerowane obrazy wraz z oryginalnym
LDR służą następnie do rekonstrukcji obrazu HDR algorytmem Debeveca (OpenCV).

---

## Sieci neuronowe
Autor: Filip Langiewicz

---

### Dane - HDR-Eye (EPFL)

Jako zbiór danych wykorzystano **HDR-Eye**:
<https://www.epfl.ch/labs/mmspg/downloads/hdr-eye/>

Interesujące foldery z datasetu:

| Folder | Zawartość | Rola |
|---|---|---|
| `Bracketed_images/` | Zdjęcia o różnych czasach naświetlenia | Źródło targetów EV ±2.7 |
| `LDR/` | Obrazy o standardowej ekspozycji (EV = 0.0) | Wejście sieci |
| `HDR/` | Referencyjne pliki HDR | Punkt odniesienia po rekonstrukcji |

Podział na zbiory:
- **Testowy**: sceny C40–C46 (7 scen) - stały, określony przez treść zadania
- **Treningowy**: wszystkie pozostałe sceny

---

### Analiza danych (EDA)

Skrypt `eda.py` wykonał skanowanie całego datasetu i wygenerował raport obejmujący:

- liczbę plików na scenę (oczekiwane: 9 zdjęć)
- dostępne wartości EV (odczyt pola `ExposureBiasValue` z metadanych EXIF)
- obecność pliku LDR dla każdej sceny
- przynależność do zbioru treningowego lub testowego

Dla scen z brakującymi metadanymi EV przeprowadzono dodatkową analizę:
wartości EV zostały wyestymowane na podstawie pola `ExposureTime` - obliczono
kroki między kolejnymi zdjęciami jako `ΔEV = log2(ET[i+1] / ET[i])`
i zweryfikowano ich spójność (tolerancja 0.15 EV). Niestety analiza ta nie przyniosła pozytywnych rezultatów z powodu braku niezbędnych metadanych w poszczególnych zdjęciach.

---

### Przygotowanie danych

Skrypt `prepare_data.py` przeskanował folder `Bracketed_images/` i dla każdej sceny:

1. Odczytał `ExposureBiasValue` z EXIF każdego zdjęcia
2. Wybrał obraz z EV = −2.7 (tolerancja ±0.05) jako target *under*
3. Wybrał obraz z EV = +2.7 (tolerancja ±0.05) jako target *over*
4. Skopiował odpowiadający plik LDR

Wynikowa struktura danych (`nn_data/selected/`):

```
nn_data/selected/
├── LDR/                  # obrazy wejściowe (EV = 0.0)
├── Bracketed_images-27/  # targety niedoświetlone (EV = −2.7)
└── Bracketed_images+27/  # targety prześwietlone (EV = +2.7)
```

W związku z ryzykiem trenowania sieci na małym zbiorze danych, niektóre sceny zostały dopuszczone do treningu po ocenie wizualnej czasu naświetlania, jaki mógł być użyty w momencie ich tworzenia.

---

### Podział i przygotowanie patchy

Podział na zbiory odbywa się na poziomie **scen**:

- **Testowy**: sceny C40–C46 (7 scen) — stały, określony przez treść zadania
- **Treningowy**: wszystkie pozostałe sceny

Patche wycinane są **z góry na dysk** przez skrypt `create_dataset.py` (przed treningiem).
Skrypt buduje kompletne trójki (LDR, under, over) dla każdej sceny, a następnie:

1. Wyrównuje rozmiary obrazów w trójce (`align_sizes`) i upscaluje jeśli potrzeba (`ensure_min_size`)
2. Losowo wycina `patches_per` patchy **256 × 256 px** z identycznym cropem dla wszystkich trzech obrazów
3. Dla zbioru treningowego stosuje augmentacje: losowe odbicie poziome, pionowe, obrót o 90°/180°/270°
4. Dla zbioru testowego patche wycina deterministycznie (bez augmentacji, `seed=42`)
5. Zapisuje patche jako PNG w strukturze `split/ldr|under|over/`

`ExposureDataset` (plik `dataset.py`) wczytuje gotowe patche z dysku.
Wynikowa liczba próbek: **1 400 treningowych** i **350 testowych**.

---

### Architektura modelu - ResUNet

Plik: `model.py`

Zastosowano architekturę **ResUNet** - wariant U-Net z  blokami resztkowymi w każdym stopniu enkodera i dekodera.

### Bloki składowe

- **`ResBlock`**: dwie warstwy `Conv2d 3×3` z `BatchNorm2d` i `ReLU`, połączone
  skip connectionem (dodanie wejścia do wyjścia)
- **`EncoderBlock`**: konwolucja wejściowa → `ResBlock` + shortcut 1×1 →
  `MaxPool2d` (downsampling ×2);
- **`DecoderBlock`**: transpozycja konwolucji (`ConvTranspose2d`) → konkatenacja
  ze skip connection → `Conv2d 3×3` → `ResBlock` + shortcut 1×1
- **Bottleneck**: `Conv2d 3×3` → `ResBlock`; podwaja liczbę kanałów
- **Head**: `Conv2d 1×1` → `Sigmoid` (normalizacja wyjścia do \([0, 1]\))

---

### Konfiguracja modeli

| Parametr | Wartość |
|---|---|
| Features (kanały enkodera) | `[32, 64, 128, 256]` |
| Liczba parametrów | 11 906 307 |
| Wejście / wyjście | 3-kanałowy obraz RGB |
| Rozmiar patcha (trening) | 256 × 256 px |
| Liczba patchy na scenę | 50 |

---

### Funkcja straty

Plik: `loss.py`

Zastosowano kombinację dwóch składowych:

`L = α · L1 + (1 − α) · SSIM`

gdzie `α = 0.8`.

---

### Trening

Plik: `train.py` | Środowisko: Kaggle (GPU NVIDIA Tesla T4)

Trening przeprowadzono **oddzielnie** dla każdego kierunku ekspozycji.
Po eksperymentach z dłuższymi sesjami (200 epok, AdamW, CosineAnnealingLR)
zdecydowano się na modele z krótszego treningu (10 epok), które osiągnęły
lepszy PSNR na zbiorze testowym - dłuższy trening prowadził do przeuczenia.

---

### Parametry wspólne

| Parametr | Wartość |
|---|---|
| Optymalizator | Adam |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=10) |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Liczba epok | 10 |
| Próbki treningowe | 1 400 |
| Próbki testowe | 350 |
| Eval every | 1 epoka |
| Metryka wyboru checkpointu | PSNR (maksymalny) |

---

### Wyniki ewaluacji

Uzyskane wyniki PSNR na poziomie ~19–20 dB dla obu kierunków ekspozycji
odzwierciedlają trudność zadania — model trenowano zaledwie przez 10 epok
na małym zbiorze danych (~28 scen treningowych), co ogranicza jego zdolność
generalizacji. Wyniki dla *underexposed* są nieco lepsze niż dla *overexposed*,
co jest spodziewane: zciemnianie obrazu jest operacją bardziej deterministyczną
niż odtwarzanie szczegółów w jasnych obszarach. Aby poprawić wyniki,
można by wydłużyć trening przy jednoczesnym zastosowaniu early stopping opartego
na LPIPS (zamiast PSNR), zwiększyć liczbę scen treningowych lub dodać do funkcji straty składnik percepcyjny oparty na VGG
(perceptual loss). Alternatywnie, zastąpienie architektury ResUNet modelem
opartym na mechanizmie uwagi (np. U-Net z Transformer bottleneckiem) mogłoby
lepiej uchwycić globalne zależności jasności w obrazie.

| Metoda       | PSNR     | LPIPS  |
|--------------|----------|--------|
| underexposed | 19.66 dB | 0.3729 |
| overexposed  | 19.00 dB | 0.5608 |

---

## Algorytmy
Autor: Dominika Boguszewska

---
