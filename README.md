# Anomaly Detection

System do wykrywania lokalnych anomalii (np. broni, obcych obiektów) na zdjęciach rentgenowskich samochodów.  
Program wykorzystuje:

- **ViT (DINO)** jako ekstraktor cech,
- **PatchCore** pamięć normalnych patchy (clear),
- **rekonstrukcję heatmapy** z poziomu tilów do pełnego obrazu.

Wejście: jeden obraz `.bmp` z X-rayem auta  
Wyjście: obraz z nałożoną heatmapą anomalii.

---

## Wymagania

- Python **3.10+**
- Karta graficzna z obsługą CUDA (opcjonalnie, ale mocno przyspiesza inferencję)
- Zainstalowane pakiety:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # lub CPU-only wg instrukcji PyTorch
  pip install timm scikit-learn numpy pillow opencv-python matplotlib tqdm
---

## Użycie

Aby skorzystać ze skryptu, upewnij się, że znajdujesz się w katalogu głównym repozytorium ("Transformer-Based-Anomaly-Detection-HackNation").
Następnie, należy użyć komendy:
```bash
python run.py --image <image-path.bmp> --out <output-path.bmp>
```
gdzie:
- <image-path.bmp> to relatywna ścieżka do obrazu,
- <output-path.bmp> to relatywna ścieżka do pliku z końcową heatmapą

---

### Autorzy

Sylwia Rybak,
Quoc Hoang Vu Van,
Mateusz Szymkowiak,
Antoni Poszkuta,
Bartosz Czyż
