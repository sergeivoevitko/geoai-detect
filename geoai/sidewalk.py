"""Split the road surface into carriageway (asphalt driving surface) and sidewalks.

At 10-30 cm/px the two are visibly different materials: asphalt is dark, grey and
uniform; Israeli sidewalks are lighter concrete/interlocking pavers, usually separated by a
curb. The asphalt appearance is learned per image from the pixels right under the
centerlines (robust median / MAD in Lab colour), so it adapts to each street and photo.
"""
from __future__ import annotations

import numpy as np
import rasterio.features
from scipy import ndimage


def split(rgb: np.ndarray, surface: np.ndarray, centerlines, transform, px_m: float,
          building: np.ndarray, tree: np.ndarray, shadow: np.ndarray,
          road_c: np.ndarray, dev: np.ndarray, sidewalk_max_m: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (carriageway, sidewalk) boolean masks on the given grid.

    The land-cover model separates "road" (asphalt carriageway) from "developed space"
    (sidewalks, plazas, parking). Inside the road surface, everything that is not developed
    space is carriageway (incl. cars and the parts hidden by trees/shadows); developed-space
    strips along the carriageway edge are sidewalks."""
    from .centerline import fill_small_holes
    H, W = surface.shape
    if not surface.any() or not centerlines:
        return surface, np.zeros_like(surface)
    seed = rasterio.features.rasterize([(ln.buffer(max(px_m, 0.8)), 1) for ln in centerlines],
                                       out_shape=(H, W), transform=transform, dtype=np.uint8).astype(bool)
    seed &= surface
    occl = (tree | shadow) & ~building
    carr = surface & (road_c | occl | ~dev)
    carr = ndimage.binary_closing(carr, iterations=max(1, int(0.8 / px_m)))
    carr = fill_small_holes(carr, int(25 / (px_m * px_m)))          # parked/driving cars, markings
    carr = ndimage.binary_opening(carr, iterations=max(1, int(0.8 / px_m)))
    lb, n = ndimage.label(carr)
    hit = np.unique(lb[seed]); hit = hit[hit > 0]
    carr = np.isin(lb, hit) | seed

    dist_out = ndimage.distance_transform_edt(~carr) * px_m
    side = dev & ~carr & ~building & (dist_out <= sidewalk_max_m)
    side = ndimage.binary_opening(side, iterations=1)
    lb, n = ndimage.label(side)
    if n:
        touch = np.unique(lb[ndimage.binary_dilation(carr, iterations=2) & side])
        side = np.isin(lb, touch[touch > 0])
    return carr, side
