from pathlib import Path
import numpy as np
import pickle
from PIL import Image
import matplotlib.cm as cm
import cv2

# ====== CONFIG ======
PROJECT_ROOT = Path(".")
OUT_ROOT     = PROJECT_ROOT / "patchcore_outputs"

HEAT_DIR = OUT_ROOT / "tile_heatmaps"
HEAT_DIR.mkdir(parents=True, exist_ok=True)

all_patch_maps = np.load(OUT_ROOT / "all_patch_maps.npy", allow_pickle=True)
cont_meta      = pickle.load(open(OUT_ROOT / "cont_patch_meta.pkl", "rb"))

print(f"Tiles in all_patch_maps: {len(all_patch_maps)}")
print(f"Tiles in cont_patch_meta: {len(cont_meta)}")


def upsample_patch_to_tile_raw(patch_scores, gh, gw, tile_w=512, tile_h=512):
    patch_scores = np.asarray(patch_scores, dtype=np.float32)
    expected = gh * gw
    n = patch_scores.shape[0]

    if n == expected + 1:
        patch_scores = patch_scores[1:]
        n = patch_scores.shape[0]

    if n != expected:
        raise ValueError(
            f"Cannot reshape patch_scores of length {n} to grid {gh}x{gw} (expected {expected})"
        )

    # Reshape to patch grid
    mat = patch_scores.reshape(gh, gw).astype(np.float32)

    tile_heat = cv2.resize(mat, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

    return tile_heat


for i, meta in enumerate(cont_meta):
    tile_path = Path(meta["path"])
    tile_stem = tile_path.stem

    gh = int(meta["grid_h"])
    gw = int(meta["grid_w"])

    tile_w = int(meta.get("tile_w", 512))
    tile_h = int(meta.get("tile_h", 512))

    patch_scores = all_patch_maps[i]

    tile_heat = upsample_patch_to_tile_raw(
        patch_scores,
        gh, gw,
        tile_w=tile_w,
        tile_h=tile_h,
    )

    np.save(HEAT_DIR / f"heat_{tile_stem}.npy", tile_heat.astype(np.float32))

    try:
        mn, mx = float(tile_heat.min()), float(tile_heat.max())
        if mx - mn < 1e-9:
            norm = np.zeros_like(tile_heat, dtype=np.float32)
        else:
            norm = (tile_heat - mn) / (mx - mn)

        cmap = cm.get_cmap("jet")
        heat_rgb = (cmap(norm)[..., :3] * 255).astype("uint8")
        overlay = Image.fromarray(heat_rgb)

        img = Image.open(tile_path).convert("RGB").resize((tile_w, tile_h))
        blended = Image.blend(img, overlay, alpha=0.45)
        blended.save(HEAT_DIR / f"heat_{tile_stem}.bmp")
    except FileNotFoundError:
        print(f"[WARN] Tile image not found for overlay: {tile_path}")

print("Done. Per-tile RAW heatmaps saved in:", HEAT_DIR)
