"""
Usage:
    python integrate_heatmap.py --image "label/folder/image.bmp"

Requires:
 - ./car_data/tiles_mapping.csv  (produced by image_tiling.py)
 - ./patchcore_outputs/tile_heatmaps/heat_<tile_id>.npy  (produced by 1_make_tile_heatmaps.py)
 - full image file at ./car_data/<image_rel_path>  (optional; used for overlay)

Output:
 - ./patchcore_outputs/stitched_full_images/stitched_heat_<image_stem>.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm

# Config (adjust if your structure differs)
PROJECT_ROOT = Path(".")
CAR_ROOT     = PROJECT_ROOT / "car_data"
CSV_PATH     = CAR_ROOT / "tiles_mapping.csv"
OUT_ROOT     = PROJECT_ROOT / "patchcore_outputs"
HEAT_DIR     = OUT_ROOT / "tile_heatmaps"
OUT_STITCH   = OUT_ROOT / "stitched_full_images"
OUT_STITCH.mkdir(parents=True, exist_ok=True)


def stitch_for_image(image_rel_path: str, save_overlay=True):
    df = pd.read_csv(CSV_PATH)
    # normalize slashes so incoming path matches CSV
    image_rel_path_norm = str(Path(image_rel_path).as_posix())

    group = df[df["image_rel_path"].astype(str) == image_rel_path_norm]
    if group.empty:
        print(f"[ERROR] No tiles found in mapping for image_rel_path = '{image_rel_path_norm}'")
        return None

    group = group.copy()
    group["tile_w"] = group["x1"] - group["x0"]
    group["tile_h"] = group["y1"] - group["y0"]

    full_w = int(group["x1"].max())
    full_h = int(group["y1"].max())

    full_heat = np.zeros((full_h, full_w), dtype=np.float32)
    count     = np.zeros_like(full_heat, dtype=np.float32)

    for _, row in group.iterrows():
        tile_id = row["tile_id"]
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        tw = int(row["tile_w"])
        th = int(row["tile_h"])

        heat_path = HEAT_DIR / f"heat_{tile_id}.npy"
        if not heat_path.exists():
            print(f"[WARN] Missing heat for tile {tile_id} -> {heat_path}")
            continue

        tile_heat = np.load(heat_path)  # should be roughly (th, tw)

        # If shapes mismatch, resize tile_heat to expected size
        if tile_heat.shape != (th, tw):
            from PIL import Image as PilImage
            tmp = PilImage.fromarray(tile_heat.astype("float32"))
            tmp = tmp.resize((tw, th), resample=PilImage.BILINEAR)
            tile_heat = np.array(tmp).astype(np.float32)

        full_heat[y0:y0+th, x0:x0+tw] += tile_heat
        count[y0:y0+th, x0:x0+tw]     += 1.0

    mask = count > 0
    if mask.any():
        full_heat[mask] = full_heat[mask] / count[mask]
    else:
        print("[ERROR] No tile heatmaps were found — nothing to stitch.")
        return None

    # percentile based normalization and suppression (same logic as 2_stitch_full_heatmaps.py)
    vals = full_heat[mask]
    low  = float(np.percentile(vals, 5.0))
    high = float(np.percentile(vals, 99.5))
    thr  = float(np.percentile(vals, 98.0))

    if high - low < 1e-9:
        heat_norm = np.zeros_like(full_heat, dtype=np.float32)
    else:
        suppressed = full_heat.copy()
        suppressed[suppressed < thr] = low
        clipped = np.clip(suppressed, low, high)
        heat_norm = (clipped - low) / (high - low)

    cmap = cm.get_cmap("jet")
    heat_rgb = (cmap(heat_norm)[..., :3] * 255).astype("uint8")
    heat_img = Image.fromarray(heat_rgb)

    # try to overlay on original full image if available
    img_rel = Path(*str(image_rel_path_norm).split("/"))
    full_img_path = CAR_ROOT / img_rel

    image_stem = Path(str(image_rel_path_norm)).stem
    out_path = OUT_STITCH / f"stitched_heat_{image_stem}.png"

    if save_overlay and full_img_path.exists():
        base = Image.open(full_img_path).convert("RGB")
        base = base.resize((full_w, full_h))
        blended = Image.blend(base, heat_img, alpha=0.45)
        blended.save(out_path)
    else:
        heat_img.save(out_path)

    print(f"Saved stitched heatmap → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True,
                        help="image_rel_path as in tiles_mapping.csv (e.g. 'label/sub/img.bmp')")
    parser.add_argument("--no-overlay", action="store_true", help="save heatmap only (no overlay on original)")
    args = parser.parse_args()

    stitch_for_image(args.image, save_overlay=not args.no_overlay)


if __name__ == "__main__":
    main()