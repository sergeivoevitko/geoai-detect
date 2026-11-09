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


def split_large(img, geoms_scores: list[dict], min_area: float = 1200.0, min_part: float = 60.0) -> list[dict]:
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
        md = max(3, int(8.0 / gsd))                       # parts at least ~16 m apart
        peaks = peak_local_max(edt, min_distance=md, threshold_abs=4.0 / gsd, labels=blob.astype(int))
        if len(peaks) < 2:
            out.append(it); continue
        markers = np.zeros(blob.shape, np.int32)
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
