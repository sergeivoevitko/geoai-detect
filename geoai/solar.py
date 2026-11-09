"""Rooftop solar panel detection for 5-30 cm imagery.

Open-vocabulary detectors tend to box the whole roof at these scales. Solar water-heater
collectors (very common on Israeli roofs, ~1x2 m) and PV modules are instead found as
mid-dark, colour-neutral, rectangular patches on building roofs. Roof shadows are darker and
strongly blue, so they are excluded by colour; tanks, AC units and clutter by shape.
"""
from __future__ import annotations

import numpy as np
import rasterio.features
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape


def detect_on_roofs(img, roofs, min_area=0.8, max_area=2000.0, progress=None) -> list[dict]:
    """roofs: building polygons in the image CRS. Returns [{"geometry", "score"}]."""
    from skimage.color import rgb2lab
    out = []
    gsd = img.gsd
    for i, roof in enumerate(roofs):
        win = from_bounds(*roof.buffer(1.0).bounds, transform=img.ds.transform).round_offsets().round_lengths()
        if win.width < 6 or win.height < 6:
            continue
        rgb, valid = img.read_rgb(win)
        wt = img.ds.window_transform(win)
        inroof = rasterio.features.rasterize([(roof.buffer(-gsd), 1)], out_shape=rgb.shape[:2],
                                             transform=wt, dtype=np.uint8).astype(bool) & valid
        if inroof.sum() < 30:
            continue
        lab = rgb2lab(rgb).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        Lr = np.median(L[inroof])
        # Measured on Israeli orthophotos: collectors/PV are mid-dark and colour-neutral
        # (L* 35-47, |a*|,|b*| < ~4); roof shadows are darker and strongly blue (b* ~ -19).
        cand = (inroof & (L >= 28) & (L < min(54.0, Lr - 10)) &
                (np.abs(A) < 5) & (B > -8) & (B < 6))
        cand = ndimage.binary_opening(cand, iterations=1)
        cand = ndimage.binary_closing(cand, iterations=1)
        lb, n = ndimage.label(cand)
        if not n:
            continue
        for geom, v in rasterio.features.shapes(lb.astype(np.int32), mask=cand, transform=wt):
            g = shape(geom).buffer(0)
            a = g.area
            if not (min_area <= a <= max_area):
                continue
            rect = g.minimum_rotated_rectangle
            fill = a / rect.area if rect.area else 0
            if fill < 0.58:
                continue
            # long thin strips along roof edges are parapet shadows, not panels
            xs, ys = rect.exterior.coords.xy
            e1 = np.hypot(xs[1] - xs[0], ys[1] - ys[0]); e2 = np.hypot(xs[2] - xs[1], ys[2] - ys[1])
            short, long_ = min(e1, e2), max(e1, e2)
            if short < 0.5 or long_ / max(short, 1e-6) > 6:
                continue
            blob = lb == v
            contrast = float(Lr - L[blob].mean())
            score = float(np.clip(0.4 + contrast / 60 + (fill - 0.65), 0, 1))
            out.append({"geometry": rect if fill > 0.8 else g.simplify(gsd), "score": round(score, 3)})
        if progress and i % 20 == 0:
            progress((i + 1) / len(roofs), f"Solar panels: roof {i + 1}/{len(roofs)}")
    return out


# ------------------------------------------------------------------ learned detector
YOLO_REPO = "finloop/yolov8s-seg-solar-panels"   # MIT weights, trained on ~30 cm US rooftop PV
YOLO_TARGET_GSD = 0.2                            # m/px the image is resampled to (tested: 0.15 m -> x0.75)


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401  (AGPL-3.0 library; fine for personal use)
        return True
    except Exception:
        return False


def detect_yolo(img, roofs, conf: float = 0.2, tile: int = 1024, overlap: int = 128, progress=None) -> list[dict]:
    """Tiled YOLOv8-seg solar panel detection. Keeps panels on roofs, or large arrays (>100 m2)."""
    from functools import lru_cache
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from rasterio.windows import Window
    from shapely.ops import unary_union
    from ultralytics import YOLO

    model = YOLO(hf_hub_download(YOLO_REPO, "best.pt"))
    scale = min(1.0, img.gsd_meters / YOLO_TARGET_GSD)
    roof_zone = unary_union([r.buffer(1.0) for r in roofs]) if roofs else None
    W, H = img.ds.width, img.ds.height
    step = tile - overlap
    tiles = [(x, y) for y in range(0, max(H - overlap, 1), step) for x in range(0, max(W - overlap, 1), step)]
    out = []
    for i, (x, y) in enumerate(tiles):
        win = Window(x, y, min(tile, W - x), min(tile, H - y))
        rgb, valid = img.read_rgb(win)
        if valid.mean() < 0.05:
            continue
        im = Image.fromarray(rgb)
        im2 = im.resize((max(32, int(im.width * scale)), max(32, int(im.height * scale))), Image.BILINEAR)
        r = model.predict(np.asarray(im2)[:, :, ::-1], imgsz=1024, conf=conf, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            continue
        wt = img.ds.window_transform(win)
        for mk, sc in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            m = np.asarray(Image.fromarray((mk * 255).astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]))) > 127
            m &= valid
            for geom, v in rasterio.features.shapes(m.astype(np.uint8), mask=m, transform=wt):
                g = shape(geom).buffer(0)
                if g.area < 0.6:
                    continue
                if roof_zone is not None and not g.intersects(roof_zone) and g.area < 100:
                    continue
                out.append({"geometry": g, "score": float(sc)})
        if progress:
            progress((i + 1) / len(tiles), f"Solar panels (AI model): tile {i + 1}/{len(tiles)}")
    return out


def merge(rule_dets: list[dict], yolo_dets: list[dict], gsd: float) -> list[dict]:
    """Union of both detectors; overlapping detections become one object (max score)."""
    from .vectorize import merge_instances
    items = [dict(d, src="rules") for d in rule_dets] + [dict(d, src="ai") for d in yolo_dets]
    merged = merge_instances([{"geometry": d["geometry"], "score": d["score"]} for d in items],
                             iou_thr=0.1, contain_thr=0.4)
    out = []
    for m in merged:
        g = m["geometry"].simplify(gsd)
        rect = g.minimum_rotated_rectangle
        if rect.area and g.area / rect.area > 0.85:
            g = rect
        out.append({"geometry": g, "score": m["score"]})
    return out
