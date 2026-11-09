"""Pixel-wise land-cover segmentation (OpenEarthMap-trained UperNet-Swin).

OpenEarthMap: 0.25-0.5 m aerial imagery from 44 countries, 8 classes. Used for
surface classes (roads, trees, water, …) and as a building prior.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from .models import DEVICE, _lock

LANDCOVER_ID = os.environ.get("GEOAI_LANDCOVER", "kishan4444/upernet-swin-openearthmap-road-segmentation")
TRAIN_GSD = 0.4          # m/px, middle of OpenEarthMap's 0.25-0.5 m range
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# user label -> OpenEarthMap class name
ALIASES = {
    "road": "road", "roads": "road", "street": "road", "streets": "road",
    "tree": "tree_vegetation", "trees": "tree_vegetation", "forest": "tree_vegetation",
    "vegetation": "tree_vegetation", "tree canopy": "tree_vegetation",
    "building": "building", "buildings": "building", "house": "building", "houses": "building",
    "water": "water", "river": "water", "lake": "water",
    "grass": "rangeland", "rangeland": "rangeland",
    "bare soil": "bareland", "bareland": "bareland", "sand": "bareland",
    "farmland": "agriculture_land", "field": "agriculture_land", "agriculture": "agriculture_land",
    "pavement": "developed_space", "developed space": "developed_space", "parking lot": "developed_space",
}


def landcover_class(label: str) -> str | None:
    return ALIASES.get(label.strip().lower())


@lru_cache(maxsize=1)
def load_landcover():
    from transformers import AutoModelForSemanticSegmentation
    m = AutoModelForSemanticSegmentation.from_pretrained(LANDCOVER_ID).to(DEVICE).eval()
    names = {int(k): v for k, v in m.config.id2label.items()}
    return m, names


@torch.inference_mode()
def predict_probs(rgb: np.ndarray, tta: bool = True) -> np.ndarray:
    """Softmax probabilities CxHxW (float32) for one RGB tile."""
    model, _ = load_landcover()
    x = torch.from_numpy(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).to(DEVICE)
    H, W = rgb.shape[:2]
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    acc = None
    with _lock:
        for fl in flips:
            xi = torch.flip(x, fl) if fl else x
            lg = model(pixel_values=xi).logits
            lg = F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
            p = torch.softmax(lg, 1)
            p = torch.flip(p, fl) if fl else p
            acc = p if acc is None else acc + p
    return (acc / len(flips))[0].cpu().numpy()


def segment_raster(img, tile: int = 1024, overlap: int = 256, tta: bool = True, progress=None):
    """Sliding-window land-cover map over the whole raster.

    Image is resampled so that ground resolution matches the model's training data.
    Each tile writes only its centre (minus overlap/2 margins), which removes seam artefacts.
    Returns (class_map uint8 HxW at native resolution, prob_max uint8 0-255, names dict).
    """
    _, names = load_landcover()
    gsd = img.gsd_meters
    scale = gsd / TRAIN_GSD                      # >1: upsample, <1: downsample
    scale = float(np.clip(scale, 0.25, 2.0))
    src_tile = int(round(tile / scale))          # native pixels per inference tile
    src_ov = int(round(overlap / scale))
    W, H = img.ds.width, img.ds.height
    cls_map = np.full((H, W), 255, np.uint8)   # 255 = no data
    conf = np.zeros((H, W), np.uint8)
    step = max(1, src_tile - src_ov)
    xs = list(range(0, max(W - src_ov, 1), step)) or [0]
    ys = list(range(0, max(H - src_ov, 1), step)) or [0]
    total, done = len(xs) * len(ys), 0
    half = src_ov // 2
    for y in ys:
        for x in xs:
            w, h = min(src_tile, W - x), min(src_tile, H - y)
            rgb, valid = img.read_rgb(Window(x, y, w, h))
            done += 1
            if valid.mean() < 0.02:
                continue
            if abs(scale - 1) > 0.05:
                t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
                nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
                t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
                rgb_in = t[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
            else:
                rgb_in = rgb
            probs = predict_probs(rgb_in, tta)
            if probs.shape[1:] != (h, w):
                probs = F.interpolate(torch.from_numpy(probs)[None], size=(h, w), mode="bilinear",
                                      align_corners=False)[0].numpy()
            c = probs.argmax(0).astype(np.uint8)
            pm = (probs.max(0) * 255).astype(np.uint8)
            # central region of this tile (full extent at raster edges)
            x0 = half if x > 0 else 0; y0 = half if y > 0 else 0
            x1 = w - half if x + w < W else w; y1 = h - half if y + h < H else h
            sl = (slice(y + y0, y + y1), slice(x + x0, x + x1))
            v = valid[y0:y1, x0:x1]
            cls_map[sl] = np.where(v, c[y0:y1, x0:x1], 255)
            conf[sl] = np.where(v, pm[y0:y1, x0:x1], 0)
            if progress:
                progress(done / total, f"Land-cover tile {done}/{total}")
    return cls_map, conf, names
