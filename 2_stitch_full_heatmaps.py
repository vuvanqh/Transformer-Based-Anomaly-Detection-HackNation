from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm

PROJECT_ROOT = Path(".")

CAR_ROOT = PROJECT_ROOT / "car_data"
CSV_PATH = CAR_ROOT / "tiles_mapping.csv"

OUT_ROOT   = PROJECT_ROOT / "patchcore_outputs"
HEAT_DIR   = OUT_ROOT / "tile_heatmaps"
OUT_STITCH = OUT_ROOT / "stitched_full_images"
OUT_STITCH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df["tile_w"] = df["x1"] - df["x0"]
df["tile_h"] = df["y1"] - df["y0"]

print("Sample mapping rows:")
print(df.head())

for image_rel_path, group in df.groupby("image_rel_path"):
    group = group.copy()

    full_w = int(group["x1"].max())
    full_h = int(group["y1"].max())

    full_heat = np.zeros((full_h, full_w), dtype=np.float32)
    count     = np.zeros_like(full_heat, dtype=np.float32)

    for _, row in group.iterrows():
        tile_id = row["tile_id"]
        x0      = int(row["x0"])
        y0      = int(row["y0"])
        tw      = int(row["tile_w"])
        th      = int(row["tile_h"])

        heat_path = HEAT_DIR / f"heat_{tile_id}.npy"
        if not heat_path.exists():
            print(f"[WARN] Heatmap not found for tile {tile_id}: {heat_path}")
            continue

        tile_heat = np.load(heat_path)  # RAW distances, shape ~ (th, tw)

        if tile_heat.shape != (th, tw):
            from PIL import Image as PilImage
            tmp = PilImage.fromarray(tile_heat.astype("float32"))
            tmp = tmp.resize((tw, th), resample=PilImage.BILINEAR)
            tile_heat = np.array(tmp).astype(np.float32)

        full_heat[y0:y0+th, x0:x0+tw] += tile_heat
        count[y0:y0+th, x0:x0+tw]     += 1.0

    mask = count > 0
    full_heat[mask] = full_heat[mask] / count[mask]

    # ---- save numeric stitched heatmap (RAW distances) ----
    image_stem = Path(str(image_rel_path)).stem

# OBCINANIE PERCENTYLI
    if mask.any():
        vals = full_heat[mask]

        # dolna i górna granica do normalizacji
        low  = float(np.percentile(vals, 5.0))
        high = float(np.percentile(vals, 99.5))

        # próg "podejrzanych" – np. górne 10% wartości
        thr  = float(np.percentile(vals, 98.0))
    else:
        low, high, thr = 0.0, 1.0, 0.0

    if high - low < 1e-9:
        heat_norm = np.zeros_like(full_heat, dtype=np.float32)
    else:
        # 1) tłumimy wszystko poniżej progu do "tła"
        suppressed = full_heat.copy()
        suppressed[suppressed < thr] = low

        # 2) przycinamy do [low, high] i normalizujemy 0–1
        clipped   = np.clip(suppressed, low, high)
        heat_norm = (clipped - low) / (high - low)

    cmap = cm.get_cmap("jet")
    heat_rgb = (cmap(heat_norm)[..., :3] * 255).astype("uint8")
    heat_img = Image.fromarray(heat_rgb)

    img_rel = Path(*str(image_rel_path).split("\\"))
    full_img_path = CAR_ROOT / img_rel

    if full_img_path.exists():
        base = Image.open(full_img_path).convert("RGB")
        base = base.resize((full_w, full_h))
        blended = Image.blend(base, heat_img, alpha=0.45)
        blended.save(OUT_STITCH / f"stitched_heat_{image_stem}.bmp")
    else:
        heat_img.save(OUT_STITCH / f"stitched_heat_{image_stem}.bmp")

    print(f"Saved stitched heatmap for {image_rel_path} → {OUT_STITCH / f'stitched_heat_{image_stem}.bmp'}")

print("Done. Stitched full-image RAW heatmaps in:", OUT_STITCH)
