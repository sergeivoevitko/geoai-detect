"""Tile2Net: road, sidewalk and crosswalk segmentation from aerial imagery.

Upstream: https://github.com/VIDA-NYU/tile2net (BSD-3-Clause), network code vendored in
third_party/tile2net. Model: HRNetV2-W48 + OCR with multi-scale attention, trained on
aerial orthophotos of US cities (zoom 19/20 web-mercator tiles, ~0.15-0.3 m/px).

Weights (satellite_2021.pth, 578 MB) are downloaded once from Tile2Net's figshare record
and checked against the published SHA-256 before they are loaded.
"""
from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from .models import DEVICE, _lock

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
WEIGHTS_URL = "https://ndownloader.figshare.com/files/67765602"   # DOI 10.6084/m9.figshare.33315570.v1
WEIGHTS_SHA256 = "745f8c099e98f112a152aedba493f61fb6d80c1761e5866f936eb5f361c7ab4d"
WEIGHTS_NAME = "satellite_2021.pth"
CLASSES = {0: "sidewalk", 1: "road", 2: "crosswalk"}             # 3 = background
TILE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _cache_dir() -> str:
    d = os.environ.get("GEOAI_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "geoai")
    os.makedirs(d, exist_ok=True)
    return d


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def weights_path(progress=None) -> str:
    """Local checkpoint path; downloads and verifies it on first use."""
    local = os.environ.get("GEOAI_TILE2NET_CKPT")
    if local:
        return local
    path = os.path.join(_cache_dir(), WEIGHTS_NAME)
    if os.path.exists(path):
        return path
    import requests
    tmp = path + ".part"
    with requests.get(WEIGHTS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total, f"Downloading road/sidewalk model ({done >> 20} of {total >> 20} MB)")
    if _sha256(tmp) != WEIGHTS_SHA256:
        os.remove(tmp)
        raise RuntimeError("Tile2Net weights failed the checksum test; please try again")
    os.replace(tmp, path)
    return path


def available() -> bool:
    return os.path.exists(os.environ.get("GEOAI_TILE2NET_CKPT") or os.path.join(_cache_dir(), WEIGHTS_NAME))


@lru_cache(maxsize=1)
def load_model(path: str):
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    import logging
    logging.getLogger("tile2net").setLevel(logging.WARNING)
    from tile2net.ocrnet import HRNet_Mscale   # vendored
    import numpy.core.multiarray as ncm
    net = HRNet_Mscale(num_classes=4, criterion=None)
    # the checkpoint also stores a few numpy scalars (training metrics); allow exactly those
    # types and nothing else, so the file still cannot run code when it is loaded
    allowed = [(ncm.scalar, "numpy.core.multiarray.scalar"), np.dtype,
               type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(allowed):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in ck["state_dict"].items()}
    net.load_state_dict(sd, strict=True)
    return net.to(DEVICE).eval()


def pick_gsd(gsd_m: float, setting="auto") -> float:
    """Model resolution: 0.15 m when the imagery is that sharp (finds small streets and
    crosswalks), otherwise 0.3 m."""
    if setting not in (None, "auto"):
        return float(setting)
    return 0.15 if gsd_m <= 0.22 else 0.3


def _predict(net, rgb: np.ndarray) -> np.ndarray:
    """rgb HxWx3 uint8 (H, W <= TILE) -> class probabilities 4xHxW (flip-averaged)."""
    h, w = rgb.shape[:2]
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    x = np.pad(x, ((0, TILE - h), (0, TILE - w), (0, 0)), mode="reflect")
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(DEVICE)
    with torch.inference_mode():
        p = net({"images": t})["pred"] + net({"images": t.flip(3)})["pred"].flip(3)
        p = torch.softmax(p.float(), 1)[0, :, :h, :w]
    return p.cpu().numpy()


def segment(img, gsd="auto", progress=None) -> dict[str, np.ndarray]:
    """Road / sidewalk / crosswalk masks (bool, native raster grid) for a GeoImage.

    The image is resampled to the model resolution and run in 1024 px tiles; each tile
    keeps only its centre (minus half the overlap), so there are no seams."""
    progress = progress or (lambda p, m: None)
    path = weights_path(lambda p, m: progress(0.05 * p, m))
    net = load_model(path)
    target = pick_gsd(img.gsd_meters, gsd)
    scale = img.gsd_meters / target                    # >1: upsample, <1: downsample
    W, H = img.ds.width, img.ds.height
    src = max(64, int(round(TILE / scale)))            # native pixels per tile
    ov = max(16, int(round(192 / scale)))
    step = max(1, src - ov)
    xs = list(range(0, max(W - ov, 1), step)) or [0]
    ys = list(range(0, max(H - ov, 1), step)) or [0]
    lab = np.full((H, W), 3, np.uint8)
    half = ov // 2
    total = len(xs) * len(ys)
    for i, (y, x) in enumerate((y, x) for y in ys for x in xs):
        progress(0.05 + 0.95 * i / total, f"Roads and sidewalks: tile {i + 1} of {total}")
        w, h = min(src, W - x), min(src, H - y)
        rgb, valid = img.read_rgb(Window(x, y, w, h))
        if valid.mean() < 0.02:
            continue
        nh, nw = min(TILE, max(32, int(round(h * scale)))), min(TILE, max(32, int(round(w * scale))))
        if (nh, nw) != (h, w):
            t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
            rgb_in = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)[0] \
                .permute(1, 2, 0).clamp(0, 255).byte().numpy()
        else:
            rgb_in = rgb
        with _lock:
            p = _predict(net, rgb_in)
        if p.shape[1:] != (h, w):
            p = F.interpolate(torch.from_numpy(p)[None], size=(h, w), mode="bilinear",
                              align_corners=False)[0].numpy()
        c = p.argmax(0).astype(np.uint8)
        c[~valid] = 3
        # keep the tile centre; image borders are kept in full
        r0 = 0 if y == 0 else half; c0 = 0 if x == 0 else half
        r1 = h if y + h >= H else h - half; c1 = w if x + w >= W else w - half
        lab[y + r0:y + r1, x + c0:x + c1] = c[r0:r1, c0:c1]
    progress(1.0, "Roads and sidewalks done")
    return {name: lab == k for k, name in CLASSES.items()}
