# Transformer-Based-Anomaly-Detection
Transformer Based Anomaly Detection for RTG image
---
System do **detekcji anomalii** na zdjęciach RTG pojazdów.  
Wejściem jest pojedynczy obraz RTG (BMP), wyjściem – **heatmapa anomalii** nałożona na ten obraz.

Cały pipeline (tiling → feature extraction → anomaly scoring → stitching → overlay) jest zintegrowany w jednym skrypcie uruchamianym z linii komend.
---

## 1. Cel systemu

- Zdjęcia RTG całych pojazdów są duże i pełne detali.
- Obrazy „czyste” i „brudne” są **globalnie bardzo podobne** (co wyszło w EDA).
- Różnice są **lokalne i subtelne** – małe skrytki, niestandardowe elementy, dopakowane fragmenty.

System nie rozpoznaje konkretnych typów kontrabandy.  
Zamiast tego uczy się, jak wygląda **normalny samochód**, i oznacza na heatmapie te miejsca, w których obraz **odstaje od wzorca normalności**.

---

## 2. Pipeline – high-level

Dla jednego obrazu wejściowego:

1. **Wczytanie obrazu RTG** (BMP).
2. **Tiling** – podział obrazu na kafelki (np. 512×512 px z overlapem).
3. **Preprocessing kafelków** pod Vision Transformer (RGB, resize, normalizacja).
4. **Ekstrakcja cech ViT** dla każdego kafelka na poziomie patchy.
5. **Projekcja PCA + porównanie do memory banku** (PatchCore-style anomaly detection).
6. **Heatmapa per kafelek** – dystanse patchy upsamplowane do 512×512.
7. **Stitching** – złożenie heatmap kafelków w jedną mapę w układzie współrzędnych oryginalnego obrazu.
8. **Normalizacja + prógowanie** – wycięcie „szumu”, podbicie tylko najwyższych anomalii.
9. **Overlay** – nałożenie heatmapy na wejściowy RTG i zapis wyniku jako obraz.

---

## 3. Wymagania

### 3.1. Środowisko

- Python 3.10+
- Rekomendowana karta GPU z CUDA (działa też na CPU, ale wolniej).

### 3.2. Biblioteki

Minimalny zestaw (dopasuj do swojego `requirements.txt`):

- `torch`, `torchvision`, `timm`
- `numpy`, `scikit-learn`
- `pillow`
- `opencv-python`
- `matplotlib` (dla colormap)

### 3.3. Artefakty modelu

Skrypt zakłada istnienie pre-wytrenowanych artefaktów (np. w katalogu `model_artifacts/`):

- `pca.pkl` – PCA dopasowane do patchy z obrazów czystych,
- `memory_bank.npy` – memory bank patchy „normalnych” po PCA,
- konfiguracja modelu (np. parametry ViT, PCA_DIM, MEMORY_BANK_SIZE, K).

---

## 4. Użycie

Przykładowe wywołanie (dopasuj nazwę pliku do swojego skryptu):

```bash
python your_script.py \
  --input path/to/input_image.bmp \
  --output path/to/output_heatmap.bmp \
  --device cuda \
  --threshold-percentile 90

