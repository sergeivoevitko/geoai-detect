#!/usr/bin/env python
"""Command-line detection: GeoTIFF in, one Shapefile per class out.

Examples
  python cli.py image.tif --classes building road tree --out results/
  python cli.py image.tif --classes building --engine sam3 --threshold 0.4
  python cli.py --xyz "https://server/{z}/{x}/{y}.jpg" --bbox -122.26 37.87 -122.25 37.875 --gsd 0.3 \
                --classes building --out results/
"""
import argparse
import os
import sys
import time

from geoai.export import export_shapefiles
from geoai.pipeline import DetectOptions, detect_image
from geoai.raster import GeoImage
from geoai import sources


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="georeferenced raster (GeoTIFF, JP2, VRT…)")
    ap.add_argument("--classes", nargs="+", default=["building", "road", "tree"])
    ap.add_argument("--engine", default="hybrid", choices=["hybrid", "sam3", "gdino_sam2"])
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--min-area", type=float, default=4.0, help="minimum polygon area in m²")
    ap.add_argument("--no-regularize", action="store_true", help="keep raw building outlines")
    ap.add_argument("--fast", action="store_true", help="skip test-time augmentation (4x faster, slightly less accurate)")
    ap.add_argument("--out", default="outputs/cli")
    ap.add_argument("--xyz", help="XYZ tile URL template instead of a file")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="WGS84 bbox for --xyz")
    ap.add_argument("--gsd", type=float, default=0.3, help="target m/px for --xyz")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    path = a.image
    if a.xyz:
        if not a.bbox:
            ap.error("--xyz needs --bbox")
        z = sources.zoom_for_gsd(a.gsd, (a.bbox[1] + a.bbox[3]) / 2)
        path = os.path.join(a.out, "service.tif")
        print(f"Downloading tiles at zoom {z}…")
        sources.fetch_xyz(a.xyz, tuple(a.bbox), z, path)
    if not path:
        ap.error("give an image path or --xyz")

    from geoai.models import sam3_available
    sam3_ok = sam3_available()
    engine = a.engine

    img = GeoImage(path)
    print(f"{path}: {img.ds.width}x{img.ds.height}px, CRS {img.crs}, {img.gsd_meters:.3f} m/px, engine={engine}")
    opt = DetectOptions(classes=a.classes, engine=engine, sam3_ok=sam3_ok, tta=not a.fast, threshold=a.threshold, tile_size=a.tile_size,
                        overlap=max(128, a.tile_size // 6), min_area=a.min_area, regularize=not a.no_regularize)
    t0 = time.time()
    feats = detect_image(img, opt, lambda p, m: print(f"\r[{p*100:5.1f}%] {m}".ljust(60), end="", flush=True))
    print()
    prefix = os.path.splitext(os.path.basename(path))[0] + "_"
    for shp in export_shapefiles(feats, img.crs, a.out, prefix):
        print("wrote", shp)
    print(f"{len(feats)} objects in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
