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
