from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

TILES_ROOT      = Path("./car_data_tiles")
CSV_PATH        = Path("./car_data/tiles_mapping.csv")
OUT_TILES_ROOT  = Path("./car_data_tiles_preprocessed")

def preprocess_tile_for_vit(tile_pil: Image.Image) -> Image.Image:
    tile_rgb = tile_pil.convert("RGB")
    return tile_rgb


def main():
    df = pd.read_csv(CSV_PATH)
    print("Rows in mapping:", len(df))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing tiles for ViT"):
        image_rel_path = row["image_rel_path"]
        tile_id = row["tile_id"]

        img_rel_path = Path(image_rel_path)
        img_parent = img_rel_path.parent
        img_stem = img_rel_path.stem

        tile_rel_path = img_parent / img_stem / f"{tile_id}.bmp"
        src_path = TILES_ROOT / tile_rel_path

        if not src_path.exists():
            print(f"[WARN] Tile file not found, skipping: {src_path}")
            continue

        tile_pil = Image.open(src_path)  # no .convert("L") here, we let preprocess decide

        out_img = preprocess_tile_for_vit(tile_pil)

        out_full_path = OUT_TILES_ROOT / tile_rel_path
        out_full_path.parent.mkdir(parents=True, exist_ok=True)

        out_img.save(out_full_path)  # save as RGB BMP

    print(f"Done. Preprocessed tiles for ViT saved under: {OUT_TILES_ROOT}")


if __name__ == "__main__":
    main()