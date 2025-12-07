import math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import pickle
import timm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 512
STRIDE = 384
INPUT_SIZE = 224
TILE_BATCH = 32
K = 1

transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# load model + pca + memory bank
def load_vit():
    print("Loading ViT...")
    vit = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    vit.eval().to(DEVICE)
    return vit

def load_pca_and_bank():
    pca = pickle.load(open("patchcore_outputs/pca.pkl","rb"))
    memory_bank = np.load("patchcore_outputs/memory_bank.npy").astype(np.float32)
    nn = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(memory_bank)
    return pca, nn

# tiling
def get_tile_offsets(length):
    if length <= TILE_SIZE:
        return [0]
    starts = list(range(0, length - TILE_SIZE + 1, STRIDE))
    last_start = length - TILE_SIZE
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts

def tile_image(img):
    w, h = img.size
    xs = get_tile_offsets(w)
    ys = get_tile_offsets(h)

    tiles = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            tiles.append((tile, x, y, x+TILE_SIZE, y+TILE_SIZE))
    return tiles, w, h

# batched tiling
def compute_tiles_heat_fast(vit, pca, nn, tiles, batch_size=TILE_BATCH):
    n_tiles = len(tiles)
    tile_tensors = []
    meta = []

    for tile_pil, x0, y0, x1, y1 in tiles:
        t = transform(tile_pil.convert("RGB"))
        tile_tensors.append(t)
        meta.append((x0,y0,x1,y1))

    all_patch_scores = []

    vit.eval()
    with torch.no_grad():
        for i in range(0, n_tiles, batch_size):
            batch = torch.stack(tile_tensors[i:i+batch_size]).to(DEVICE)
            if DEVICE.type == "cuda":
                with autocast():
                    feats = vit.forward_features(batch)
            else:
                feats = vit.forward_features(batch)

            if isinstance(feats, dict):
                feats_tensor = feats.get("x", next(iter(feats.values())))
            else:
                feats_tensor = feats

            if feats_tensor.ndim == 2:
                feats_tensor = feats_tensor.unsqueeze(1)

            feats_np = feats_tensor.cpu().numpy().astype(np.float32)

            B, N_tokens, D = feats_np.shape

            for b in range(B):
                tokens_np = feats_np[b]
                num_tokens = tokens_np.shape[0]

                grid_side = int(round(math.sqrt(num_tokens)))
                expected  = grid_side * grid_side
                if num_tokens == expected + 1:
                    patch_tokens = tokens_np[1:]
                else:
                    patch_tokens = tokens_np

                arr_pca = pca.transform(patch_tokens)
                Dists, _ = nn.kneighbors(arr_pca, n_neighbors=K, return_distance=True)
                patch_scores = (Dists**2).mean(axis=1).astype(np.float32)
                all_patch_scores.append(patch_scores)

    # convert patch scores => tile heatmaps
    tiles_info = []
    for ps, (x0,y0,x1,y1) in zip(all_patch_scores, meta):
        gh = gw = int(round(math.sqrt(len(ps))))
        mat = ps.reshape(gh, gw).astype(np.float32)
        small = Image.fromarray(mat)
        up = small.resize((x1-x0, y1-y0), resample=Image.BILINEAR)
        tile_heat = np.array(up).astype(np.float32)
        tiles_info.append((tile_heat, x0, y0, x1, y1))
    return tiles_info


# Image reconstruction
def stitch_tiles(tiles_info, full_w, full_h):
    full_heat = np.zeros((full_h, full_w), dtype=np.float32)
    count     = np.zeros_like(full_heat, dtype=np.float32)

    for tile_heat, x0,y0,x1,y1 in tiles_info:
        h,w = tile_heat.shape
        full_heat[y0:y1, x0:x1] += tile_heat
        count[y0:y1, x0:x1]     += 1

    mask = count > 0
    full_heat[mask] /= count[mask]
    return full_heat


# heatmap
def make_heatmap(full_heat, base_img):
    vals = full_heat[full_heat > 0]
    low  = float(np.percentile(vals, 5))
    high = float(np.percentile(vals, 99.5))
    thr  = float(np.percentile(vals, 90))

    suppressed = full_heat.copy()
    suppressed[suppressed < thr] = low

    clipped = np.clip(suppressed, low, high)
    heat_norm = (clipped - low) / (high - low + 1e-9)

    cmap = cm.get_cmap("jet")
    heat_rgb = (cmap(heat_norm)[..., :3] * 255).astype("uint8")
    heat_img = Image.fromarray(heat_rgb)

    base_img = base_img.convert("RGB").resize(heat_img.size)
    blended = Image.blend(base_img, heat_img, alpha=0.45)
    return blended


# MAIN testing function
def run_single_image(path_in, path_out):
    img = Image.open(path_in).convert("L")

    tiles, full_w, full_h = tile_image(img)

    vit = load_vit()
    pca, nn = load_pca_and_bank()

    tiles_info = compute_tiles_heat_fast(vit, pca, nn, tiles)

    full_heat = stitch_tiles(tiles_info, full_w, full_h)

    overlay = make_heatmap(full_heat, img)

    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path_out)
    print(f"Saved heatmap: {path_out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="heatmap_output.png")
    args = parser.parse_args()

    run_single_image(args.image, args.out)
