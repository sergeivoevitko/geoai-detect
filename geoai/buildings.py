"""Split very large building footprints that are really several touching buildings.

Large complexes (hospitals, campuses, attached blocks) often come out of the land-cover model
as one blob. Neighbouring buildings are usually separated on the photo by a roof edge: a change
in roof height (a shadow line), material or colour. We over-segment the blob with a watershed
seeded at the "cores" of the shape, then merge back any two parts whose shared border shows no
strong edge in the image. Only splits along real edges survive.
"""
from __future__ import annotations

import numpy as np
import rasterio.features
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape


def split_large(img, geoms_scores: list[dict], min_area: float = 350.0, min_part: float = 40.0) -> list[dict]:
    from skimage.feature import peak_local_max
    from skimage.filters import sobel
    from skimage.segmentation import watershed
    from skimage.color import rgb2gray

    gsd = img.gsd
    out = []
    for it in geoms_scores:
        g = it["geometry"]
        if g.area < min_area:
            out.append(it); continue
        win = from_bounds(*g.buffer(2).bounds, transform=img.ds.transform).round_offsets().round_lengths()
        rgb, valid = img.read_rgb(win)
        wt = img.ds.window_transform(win)
        blob = rasterio.features.rasterize([(g, 1)], out_shape=rgb.shape[:2], transform=wt,
                                           dtype=np.uint8).astype(bool) & valid
        if blob.sum() < 100:
            out.append(it); continue
        gray = ndimage.gaussian_filter(rgb2gray(rgb), 1.0)
        grad = sobel(gray)
        edt = ndimage.distance_transform_edt(blob)
        # seeds 1: narrow necks -- shrinking the blob by ~1 m separates buildings joined by a thin
        # strip (a shared wall line, a gap the segmentation closed)
        core = ndimage.binary_erosion(blob, iterations=max(1, int(1.0 / gsd)))
        lab_c, n_c = ndimage.label(core)
        if n_c:
            sizes = ndimage.sum(core, lab_c, range(1, n_c + 1)) * gsd * gsd
            big = [k + 1 for k, a in enumerate(sizes) if a >= min_part * 0.5]
        else:
            big = []
        markers = np.zeros(blob.shape, np.int32)
        if len(big) >= 2:
            for j, k in enumerate(big, start=1):
                markers[lab_c == k] = j
        else:
            # seeds 2: separate "cores" of the shape (one per house-sized part)
            md = max(3, int(5.0 / gsd))
            peaks = peak_local_max(edt, min_distance=md, threshold_abs=2.5 / gsd, labels=blob.astype(int))
            if len(peaks) < 2:
                out.append(it); continue
            for k, (r, c) in enumerate(peaks, start=1):
                markers[r, c] = k
        # flood the image gradient: borders settle on the strongest edges between the seeds
        labels = watershed(grad, markers, mask=blob)
        # merge neighbours whose shared border has no real edge
        thr = float(np.percentile(grad[blob], 80))
        changed = True
        while changed:
            changed = False
            ids = [k for k in np.unique(labels) if k > 0]
            for a in ids:
                ma = labels == a
                if not ma.any():
                    continue
                ring = ndimage.binary_dilation(ma, iterations=1) & ~ma & blob
                for b in np.unique(labels[ring]):
                    if b <= 0 or b == a:
                        continue
                    border = ring & (labels == b)
                    border = border | (ndimage.binary_dilation(labels == b, iterations=1) & ma)
                    small = min(ma.sum(), (labels == b).sum()) * gsd * gsd < min_part
                    if small or grad[border].mean() < thr:
                        labels[labels == b] = a
                        changed = True
                        break
                if changed:
                    break
        parts = [k for k in np.unique(labels) if k > 0]
        if len(parts) < 2:
            out.append(it); continue
        for k in parts:
            m = labels == k
            for geom, v in rasterio.features.shapes(m.astype(np.uint8), mask=m, transform=wt):
                p = shape(geom).buffer(0)
                if p.area >= min_part:
                    out.append({"geometry": p, "score": it["score"]})
    return out


def split_with_sam(img, items: list[dict], min_area: float = 350.0, spacing_m: float = 5.0,
                   min_roof: float = 25.0, progress=None) -> list[dict]:
    """Dense blocks: replace each large merged footprint by the individual roofs SAM finds in it.

    A grid of single clicks is placed inside the footprint; each click returns the roof surface it
    lands on. Good roofs (confident, mostly inside the footprint, house-sized) are accepted largest-
    score first without overlapping each other. If SAM cannot explain most of the footprint, the
    original footprint is kept, so this never loses a building."""
    from rasterio.windows import from_bounds
    from . import models
    gsd = img.gsd
    out = []
    big = [it for it in items if it["geometry"].area >= min_area]
    done = 0
    for it in items:
        g = it["geometry"]
        if g.area < min_area:
            out.append(it); continue
        win = from_bounds(*g.buffer(8).bounds, transform=img.ds.transform).round_offsets().round_lengths()
        rgb, valid = img.read_rgb(win)
        wt = img.ds.window_transform(win)
        blob = rasterio.features.rasterize([(g, 1)], out_shape=rgb.shape[:2], transform=wt,
                                           dtype=np.uint8).astype(bool) & valid
        inner = ndimage.binary_erosion(blob, iterations=max(1, int(1.5 / gsd)))
        step = max(2, int(spacing_m / gsd))
        rr, cc = np.mgrid[step // 2:blob.shape[0]:step, step // 2:blob.shape[1]:step]
        pts = [(c, r) for r, c in zip(rr.ravel(), cc.ravel()) if inner[r, c]]
        if len(pts) < 2:
            out.append(it); continue
        masks, scores = models.sam2_points_batch(rgb, ("roofs", id(img), win.col_off, win.row_off,
                                                       win.width, win.height), pts)
        cand = []
        for m, sc in zip(masks, scores):
            a = m.sum() * gsd * gsd
            if sc < 0.5 or a < min_roof or a > g.area * 1.1:
                continue
            inside = (m & blob).sum() / max(1, m.sum())
            if inside < 0.8:
                continue
            cand.append((sc * min(1.0, a / 80.0), m & blob))
        cand.sort(key=lambda t: -t[0])
        taken = np.zeros(blob.shape, bool)
        parts = []
        for _, m in cand:
            new = m & ~taken
            if new.sum() < 0.7 * m.sum() or new.sum() * gsd * gsd < min_roof:
                continue
            taken |= new
            parts.append(new)
        if len(parts) < 2 or taken.sum() < 0.5 * blob.sum():
            out.append(it)
        else:
            # unclaimed remainder: keep building-sized, wide pieces (missed roofs), drop alley strips
            rest = ndimage.binary_opening(blob & ~ndimage.binary_dilation(taken, iterations=1),
                                          iterations=max(1, int(1.0 / gsd)))
            lab_r, n_r = ndimage.label(rest)
            for k in range(1, n_r + 1):
                piece = lab_r == k
                if (piece.sum() * gsd * gsd >= 60 and
                        ndimage.distance_transform_edt(piece).max() * gsd >= 3.0):
                    parts.append(piece)
            for m in parts:
                m = ndimage.binary_opening(m, iterations=1)
                for geom, v in rasterio.features.shapes(m.astype(np.uint8), mask=m, transform=wt):
                    p = shape(geom).buffer(0)
                    if p.area >= min_roof:
                        out.append({"geometry": p, "score": it["score"]})
        done += 1
        if progress:
            progress(done / max(1, len(big)), f"Separating attached buildings {done}/{len(big)}")
    return out
