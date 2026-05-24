# SIGK - Projekt 5: Stick Animation

## Opis projektu

Celem projektu było stworzenie rozwiązania opartego o sieci dyfuzyjne, które generuje animację stickmana na podstawie polecenia tekstowego określającego typ ruchu. Obsługiwane są dwie klasy animacji: `walk` oraz `jump`.

Model generuje sekwencję ruchu w postaci tensora o wymiarach `[48, 15, 3]`, co odpowiada 48 klatkom animacji, 15 punktom kluczowym szkieletu i 3 współrzędnym przestrzennym dla każdego punktu. Wygenerowane sekwencje są następnie wizualizowane jako animacje GIF przedstawiające połączony szkielet stickmana.


---

## Model dyfuzyjny
Autor: Filip Langiewicz

### Przygotowanie zbioru danych

Dane wejściowe zostały przygotowane w postaci sekwencji ruchu zapisanych jako pliki `.npy`, rozdzielonych na dwie klasy: `walk` oraz `jump`. W preprocessingu każda sekwencja była przeskalowywana czasowo do stałej długości 48 klatek, co zapewniało zgodność z docelowym formatem wejścia i wyjścia modelu.

Następnie każda sekwencja była centrowana przestrzennie względem średniej pozycji wybranych punktów centralnych ciała, co ograniczało wpływ absolutnego położenia postaci i ułatwiało modelowi naukę samej dynamiki ruchu. Dodatkowo dane treningowe były rozszerzane przez augmentację obejmującą losowy obrót wokół osi pionowej oraz opcjonalne odbicie lustrzane szkieletu.

Podział danych został wykonany automatycznie z zachowaniem stratyfikacji względem podtypów ruchu. Ostatecznie przygotowany zbiór treningowy został zapisany do pliku `train.npz`, a zbiór testowy do pliku `test.npz`; osobno zapisano również statystyki normalizacji (`norm_stats.npy`) wykorzystywane później podczas treningu i generacji próbek.

W trakcie przygotowania danych zastosowano także różną liczbę augmentacji dla obu klas, aby zwiększyć różnorodność sekwencji i lepiej zbalansować zbiór treningowy. Dla klasy `walk` generowano 7 wariantów augmentacyjnych, natomiast dla klasy `jump` 13.

### Reprezentacja danych

Każda próbka ruchu ma postać tensora `[48, 15, 3]`, gdzie:
- `48` oznacza liczbę klatek animacji,
- `15` oznacza liczbę punktów kluczowych szkieletu,
- `3` oznacza współrzędne przestrzenne `x, y, z`.

Tak przygotowana reprezentacja odpowiada uproszczonemu modelowi ludzkiego szkieletu i pozwala bezpośrednio modelować trajektorie najważniejszych stawów w czasie. Dzięki temu model nie generuje gotowego obrazu, lecz strukturę ruchu, którą można później wizualizować w formie animacji stickmana.

### Architektura modelu

Zastosowany model generatywny opiera się na architekturze `MotionDenoiser`, której zadaniem jest przewidywanie szumu dodanego do sekwencji ruchu na zadanym kroku procesu dyfuzji. Model przyjmuje na wejściu tensor reprezentujący zaszumioną animację, indeks kroku czasowego dyfuzji oraz etykietę klasy ruchu.

Architektura składa się z kilku głównych elementów:
- warstwy wejściowej projekcji liniowej, która mapuje pozycje 15 stawów z każdej klatki do przestrzeni ukrytej modelu,
- sinusoidalnego embeddingu czasu, przetwarzanego następnie przez sieć MLP,
- embeddingu klasy ruchu, umożliwiającego warunkowanie modelu na `walk` lub `jump`,
- enkodera transformerowego przetwarzającego całą sekwencję klatek,
- warstwy wyjściowej mapującej reprezentację ukrytą z powrotem do formatu `[48, 15, 3]`.

W odróżnieniu od klasycznych modeli generujących pojedyncze obrazy, tutaj model operuje bezpośrednio na sekwencji ruchu. Transformer pozwala uchwycić zależności czasowe pomiędzy klatkami, co ma kluczowe znaczenie przy generowaniu realistycznych animacji.

### Proces dyfuzji

Zastosowano model typu Gaussian Diffusion, który realizuje standardowy schemat forward-reverse. W procesie forward do sekwencji ruchu stopniowo dodawany jest szum Gaussa, natomiast w procesie reverse model uczy się ten szum usuwać i odzyskiwać poprawną strukturę animacji.

W implementacji wykorzystano harmonogram wariancji oparty na liniowo rosnących współczynnikach `beta`, z liczbą kroków dyfuzji równą 1000. Dzięki temu generacja przebiega iteracyjnie od czystego szumu do końcowej sekwencji ruchu.

Model wykorzystuje również mechanizm classifier-free guidance. W trakcie treningu część etykiet klasowych jest losowo ukrywana, a podczas generacji możliwe jest sterowanie siłą warunkowania przez parametr `guidance_scale`, co pozwala uzyskać bardziej zgodne z klasą i stabilniejsze próbki.

### Trening

W implementacji treningu wykorzystano następujące parametry:
- liczba klatek: `48`,
- liczba stawów: `15`,
- wymiar modelu: `384`,
- liczba głów attention: `6`,
- liczba warstw transformera: `6`,
- dropout: `0.1`,
- liczba kroków dyfuzji: `1000`,
- batch size: `32`,
- learning rate: `1e-4`,
- optimizer: `AdamW`,
- weight decay: `1e-4`,
- scheduler: `CosineAnnealingLR`,
- gradient clipping: `1.0`,
- dodatkowa waga składnika velocity loss: `0.1`.

Model był okresowo zapisywany w postaci checkpointów, a dodatkowo co określoną liczbę epok generowane były jakościowe próbki ruchu dla obu klas. Pozwalało to śledzić nie tylko przebieg funkcji straty, ale również faktyczną jakość animacji podczas uczenia.

### Funkcja straty

Podstawowym składnikiem funkcji straty była średniokwadratowa różnica pomiędzy prawdziwym szumem a szumem przewidzianym przez model. Oprócz tego zastosowano dodatkowy składnik oparty na różnicy prędkości pomiędzy kolejnymi klatkami, co miało wspierać płynność i spójność ruchu w czasie.

Łączna strata miała więc charakter mieszany: model uczył się zarówno poprawnej rekonstrukcji szumu, jak i zachowania odpowiedniej dynamiki trajektorii stawów. Taki zabieg był szczególnie istotny dla danych sekwencyjnych, gdzie jakość ruchu zależy nie tylko od pozycji, ale też od zmian pomiędzy klatkami.

Można to zapisać opisowo jako:

`total_loss = noise_loss + 0.1 * velocity_loss`

gdzie:
- `noise_loss` to podstawowy błąd rekonstrukcji szumu,
- `velocity_loss` to dodatkowy składnik wymuszający większą spójność ruchu pomiędzy kolejnymi klatkami,
- `0.1` to waga ograniczająca wpływ drugiego składnika na całkowitą stratę.

Taka postać funkcji straty sprawia, że model nie ogranicza się jedynie do odtwarzania statycznych pozycji, ale uczy się również naturalnej dynamiki ruchu stickmana.

### Przebieg treningu

W pierwszej fazie treningu model uczony był przez 5000 epok. Z zapisanych wyników widać systematyczny spadek straty treningowej od wartości około `1.4979` na początku do znacznie niższych wartości w dalszej części uczenia, przy jednoczesnym okresowym zapisie próbek jakościowych. Niestety trening nie ukończył się poprawnie, z powodu ograniczeń pamięciowych urządzenia.

Na podstawie oceny wizualnej wygenerowanych animacji do dalszej pracy wybrano checkpoint z epoki `3850` z pierwszej fazy treningu. Następnie przeprowadzono drugą fazę dotrenowywania modelu, z której finalnie wykorzystano checkpoint z epoki `4200`, ponieważ osiągał najniższą stratę i jednocześnie dawał dobre jakościowo animacje.

W trakcie treningu generowane były również przykładowe sekwencje dla obu klas i zapisywane do plików pośrednich, co ułatwiało ocenę jakości modelu bez potrzeby przeprowadzania pełnej ewaluacji numerycznej.

### Wyniki jakościowe

Najważniejszym elementem oceny na obecnym etapie była analiza wizualna wygenerowanych animacji.

Wygenerowane animacje pokazują, że model nauczył się tworzyć uporządkowane sekwencje punktów kluczowych odpowiadających ruchowi stickmana. Szczególnie istotne było zachowanie ogólnej struktury ciała oraz ciągłości ruchu pomiędzy kolejnymi klatkami animacji.

Poniżej znajduje się przykładowa siatka wygenerowanych animacji dla klasy `jump`:

<div style="display:grid; grid-template-columns:repeat(4, 300px); gap:12px;">
  <img src="modeling/results/jump/jump_s1.gif" width="300">
  <img src="modeling/results/jump/jump_s2.gif" width="300">
  <img src="modeling/results/jump/jump_s3.gif" width="300">
  <img src="modeling/results/jump/jump_s4.gif" width="300">
  <img src="modeling/results/jump/jump_s5.gif" width="300">
  <img src="modeling/results/jump/jump_s6.gif" width="300">
  <img src="modeling/results/jump/jump_s7.gif" width="300">
  <img src="modeling/results/jump/jump_s8.gif" width="300">
  <img src="modeling/results/jump/jump_s9.gif" width="300">
  <img src="modeling/results/jump/jump_s10.gif" width="300">
  <img src="modeling/results/jump/jump_s11.gif" width="300">
  <img src="modeling/results/jump/jump_s12.gif" width="300">
</div>


Poniżej znajduje się przykładowa siatka wygenerowanych animacji dla klasy `walk`:


<div style="display:grid; grid-template-columns:repeat(4, 300px); gap:12px;">
  <img src="modeling/results/walk/walk_s1.gif" width="300">
  <img src="modeling/results/walk/walk_s2.gif" width="300">
  <img src="modeling/results/walk/walk_s3.gif" width="300">
  <img src="modeling/results/walk/walk_s4.gif" width="300">
  <img src="modeling/results/walk/walk_s5.gif" width="300">
  <img src="modeling/results/walk/walk_s6.gif" width="300">
  <img src="modeling/results/walk/walk_s7.gif" width="300">
  <img src="modeling/results/walk/walk_s8.gif" width="300">
  <img src="modeling/results/walk/walk_s9.gif" width="300">
  <img src="modeling/results/walk/walk_s10.gif" width="300">
  <img src="modeling/results/walk/walk_s11.gif" width="300">
  <img src="modeling/results/walk/walk_s12.gif" width="300">
</div>


---

## Ewaluacja

Autor: Dominika Boguszewska

### Uzyskane wartości metryk

W celu oceny jakości generowanych animacji wykorzystano trzy metryki: Frechet Motion Distance (FMD), Mean Per Joint Position Error (MPJPE) oraz wariancję między wygenerowanymi próbkami (Var). Metryki te pozwalają ocenić zarówno zgodność wygenerowanego ruchu z danymi rzeczywistymi, jak i różnorodność generowanych animacji.

| **Ruch** |  **FMD** | **MPJPE** | **Var** |
|:--------:|---------:|----------:|--------:|
|  *walk*  |  35.0393 |    2.7084 |  9.5103 |
|  *jump*  | 149.2054 |    2.1211 |  3.3395 |

### Analiza wyników

Dla animacji typu walk uzyskano znacznie niższą wartość metryki FMD niż dla ruchu jump. Oznacza to, że generowane sekwencje chodu są bardziej zbliżone do rzeczywistych danych treningowych pod względem ogólnej dynamiki i rozkładu ruchu. Ruch chodzenia jest bardziej regularny i powtarzalny, dzięki czemu model łatwiej uczy się jego charakterystyki. W przypadku animacji jump wartość FMD jest wyraźnie wyższa, co wskazuje, że model miał większe trudności z poprawnym odwzorowaniem tego typu ruchu. Skok jest ruchem bardziej dynamicznym i mniej przewidywalnym niż chód, dlatego generowanie realistycznych trajektorii pozycji stawów jest trudniejsze.

Metryka MPJPE osiągnęła podobne wartości dla obu ruchów. Dla ruchu jump uzyskano nawet nieco niższy błąd niż dla walk, co sugeruje, że średnie położenie stawów było odwzorowane poprawnie. Jednocześnie wysoki FMD dla skoku pokazuje, że mimo poprawnych pozycji pojedynczych stawów, cała dynamika ruchu mogła odbiegać od rzeczywistych sekwencji.

Analizując wariancję (Var), można zauważyć, że dla ruchu walk jest ona większa niż dla jump. Oznacza to, że model generował bardziej zróżnicowane animacje chodu, co można uznać za pozytywny rezultat świadczący o kreatywności modelu i unikaniu generowania identycznych próbek. Niższa wariancja dla skoku może wskazywać, że model częściej generował podobne sekwencje ruchu, prawdopodobnie ze względu na trudność tego zadania.

---

## Podsumowanie

Przeprowadzona ewaluacja pokazuje, że model lepiej radzi sobie z generowaniem ruchu typu walk niż jump. Animacje chodu są bardziej realistyczne i bardziej zróżnicowane, co potwierdzają niższe wartości FMD oraz wyższa wariancja. Generowanie ruchu skoku okazało się bardziej wymagające, szczególnie pod względem zachowania realistycznej dynamiki ruchu. Mimo to uzyskane wartości MPJPE wskazują, że model poprawnie odwzorowuje pozycje stawów i potrafi generować spójne animacje szkieletowe.

