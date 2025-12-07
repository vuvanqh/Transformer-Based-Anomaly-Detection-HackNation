import os
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(r"./car_data")
TILES_DIR = Path(r"./car_data_tiles")

TILE_SIZE = 512
STRIDE = 384

def load_bmp(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("L")

def get_tile_offsets(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def compute_orig_coords(crop_box, x, y, tile_size):
    crop_x0, crop_y0, _, _ = crop_box
    X0 = crop_x0 + x
    Y0 = crop_y0 + y
    X1 = X0 + tile_size
    Y1 = Y0 + tile_size
    return X0, Y0, X1, Y1


bmp_paths = sorted(ROOT_DIR.rglob("*.bmp"))
print(f"Found {len(bmp_paths)} BMP files")

all_rows = []

for img_path in tqdm(bmp_paths, desc="Processing BMPs"):

    rel = img_path.relative_to(ROOT_DIR)
    label = rel.parts[0]
    subfolder = rel.parts[1] if len(rel.parts) > 2 else ""

    orig_img = load_bmp(img_path)

    crop_box = (0, 0, orig_img.width, orig_img.height)
    w, h = orig_img.size

    xs = get_tile_offsets(w, TILE_SIZE, STRIDE)
    ys = get_tile_offsets(h, TILE_SIZE, STRIDE)

    tiles_dir = TILES_DIR / rel.parent / img_path.stem
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            tile = orig_img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

            X0, Y0, X1, Y1 = compute_orig_coords(crop_box, x, y, TILE_SIZE)

            tile_id = f"{label}_{img_path.stem}_x{i:03d}_y{j:03d}"
            tile_path = tiles_dir / f"{tile_id}.bmp"
            tile.save(tile_path)

            all_rows.append({
                "tile_id": tile_id,
                "label": label,
                "subfolder": subfolder,
                "image_rel_path": str(rel),
                "image_name": img_path.name,
                "x0": int(X0),
                "y0": int(Y0),
                "x1": int(X1),
                "y1": int(Y1),
            })


df = pd.DataFrame(all_rows)
output_csv = ROOT_DIR / "tiles_mapping.csv"
df.to_csv(output_csv, index=False)

print(f"Done. Mapping saved to: {output_csv}")
