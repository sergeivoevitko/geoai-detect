"""Raster I/O: open georeferenced imagery, read RGB windows, generate tiles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine


@dataclass
class Tile:
    window: Window          # pixel window in the source raster
    transform: Affine       # affine transform of this window (pixel -> source CRS)
    image: np.ndarray       # HxWx3 uint8 RGB
    valid: np.ndarray       # HxW bool, True where the image has data
    touches: tuple          # (left, top, right, bottom): True if the side is an interior seam


class GeoImage:
    """Thin wrapper around a rasterio dataset that always yields 8-bit RGB."""

    def __init__(self, path: str, bands: tuple[int, int, int] | None = None):
        self.path = path
        self.ds = rasterio.open(path)
        if self.ds.crs is None:
            raise ValueError(
                f"{path} has no coordinate reference system. A georeferenced image "
                "(GeoTIFF with CRS, or an image with a .prj/.aux.xml/world file) is required."
            )
        self.bands = bands or self._default_bands()
        self._stretch = None if self.ds.dtypes[0] == "uint8" else self._compute_stretch()

    # ------------------------------------------------------------------ info
    def _default_bands(self) -> tuple[int, int, int]:
        n = self.ds.count
        if n >= 3:
            return (1, 2, 3)
        return (1, 1, 1)

    @property
    def crs(self):
        return self.ds.crs

    @property
    def gsd(self) -> float:
        """Ground sample distance in CRS units (meters for projected CRS)."""
        return float(abs(self.ds.res[0]))

    @property
    def gsd_meters(self) -> float:
        if self.ds.crs.is_geographic:
            # approximate: degrees -> meters at image center latitude
            lat = (self.ds.bounds.top + self.ds.bounds.bottom) / 2
            return self.gsd * 111_320 * np.cos(np.radians(lat))
        return self.gsd * (self.ds.crs.linear_units_factor[1] if self.ds.crs.linear_units_factor else 1)

    def info(self) -> dict:
        return {
            "path": self.path,
            "width": self.ds.width,
            "height": self.ds.height,
            "bands": self.ds.count,
            "dtype": self.ds.dtypes[0],
            "crs": self.ds.crs.to_string(),
            "gsd_m": round(self.gsd_meters, 4),
            "bounds": list(self.ds.bounds),
        }

    # --------------------------------------------------------------- reading
    def _compute_stretch(self):
        """Per-band 2-98 percentile stretch, computed on a decimated read."""
        f = max(1, max(self.ds.width, self.ds.height) // 2048)
        arr = self.ds.read(
            list(self.bands),
            out_shape=(3, max(1, self.ds.height // f), max(1, self.ds.width // f)),
            masked=True,
        )
        lo, hi = [], []
        for b in arr:
            vals = b.compressed() if np.ma.isMaskedArray(b) else b.ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                lo.append(0.0); hi.append(1.0); continue
            lo.append(float(np.percentile(vals, 2)))
            hi.append(float(np.percentile(vals, 98)))
        return np.array(lo), np.array(hi)

    def read_rgb(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        arr = self.ds.read(list(self.bands), window=window, boundless=True, masked=True)
        valid = ~np.ma.getmaskarray(arr).any(axis=0)
        data = np.ma.filled(arr, 0).astype(np.float32)
        if self._stretch is not None:
            lo, hi = self._stretch
            data = (data - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None] * 255.0
        rgb = np.clip(data, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        # all-black or all-white pixels in uint8 imagery are usually nodata collars
        if self.ds.nodata is None and self.ds.dtypes[0] == "uint8":
            valid &= ~((rgb == 0).all(axis=2))
        return rgb, valid

    def tiles(self, size: int = 1024, overlap: int = 128) -> Iterator[Tile]:
        W, H = self.ds.width, self.ds.height
        step = max(1, size - overlap)
        xs = list(range(0, max(W - overlap, 1), step)) or [0]
        ys = list(range(0, max(H - overlap, 1), step)) or [0]
        for y in ys:
            for x in xs:
                w = min(size, W - x)
                h = min(size, H - y)
                win = Window(x, y, w, h)
                rgb, valid = self.read_rgb(win)
                if valid.mean() < 0.02:
                    continue
                touches = (x > 0, y > 0, x + w < W, y + h < H)
                yield Tile(win, self.ds.window_transform(win), rgb, valid, touches)

    def tile_count(self, size: int = 1024, overlap: int = 128) -> int:
        step = max(1, size - overlap)
        nx = len(range(0, max(self.ds.width - overlap, 1), step)) or 1
        ny = len(range(0, max(self.ds.height - overlap, 1), step)) or 1
        return nx * ny

    def close(self):
        self.ds.close()
