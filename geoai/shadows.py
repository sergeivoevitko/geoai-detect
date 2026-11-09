"""Shadow detection and compensation for aerial RGB.

Building shadows in city imagery make roads dark and bluish; models trained mostly on
sunlit scenes then miss them (or call them water). We detect shadow pixels (dark AND
bluish, which separates them from dark-but-neutral asphalt) and brighten them with a
per-channel gain that matches the sunlit part of the same tile, feathered at the edges.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def shadow_mask(rgb: np.ndarray, min_px: int = 1500) -> np.ndarray:
    f = rgb.astype(np.float32)
    valid = f.mean(2) > 2
    if valid.sum() < 100:
        return np.zeros(valid.shape, bool)
    # region-scale brightness and colour: cast shadows are large, smooth, dark and cool
    Lr = ndimage.gaussian_filter(f.mean(2), 4)
    R = ndimage.gaussian_filter(f[..., 0], 4); B = ndimage.gaussian_filter(f[..., 2], 4)
    med = float(np.median(Lr[valid]))
    m = (Lr < 0.72 * med) & (B + 6 >= R) & valid
    m = ndimage.binary_closing(m, iterations=4)
    m = ndimage.binary_opening(m, iterations=3)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, np.nonzero(sizes >= min_px)[0] + 1)
    return m


def compensate(rgb: np.ndarray, mask: np.ndarray | None = None, max_gain: float = 3.5) -> np.ndarray:
    m = shadow_mask(rgb) if mask is None else mask
    frac = m.mean()
    if frac < 0.003 or frac > 0.85:
        return rgb
    f = rgb.astype(np.float32)
    lit = ~ndimage.binary_dilation(m, iterations=6) & (f.mean(2) > 2)
    if lit.sum() < 100:
        return rgb
    gain = np.clip(f[lit].mean(0) / np.maximum(f[m].mean(0), 1.0), 1.0, max_gain)   # per channel
    soft = ndimage.gaussian_filter(m.astype(np.float32), 3)[..., None]
    out = f * (1.0 + soft * (gain[None, None, :] - 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)
