"""Write detections to Shapefiles (one per class) in the source image's CRS."""
from __future__ import annotations

import os
import re
import zipfile

import geopandas as gpd


def safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", label.strip().lower()).strip("_") or "objects"


def to_geodataframe(features: list[dict], crs) -> gpd.GeoDataFrame:
    rows = []
    for i, f in enumerate(features, 1):
        g = f["geometry"]
        rows.append({
            "id": i,
            "class": f["label"][:254],
            "score": round(float(f.get("score", 1.0)), 4),
            "source": f.get("source", "auto")[:20],
            "area_m2": round(float(g.area), 2),
            "perim_m": round(float(g.length), 2),
            "geometry": g,
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if gdf.crs is not None and gdf.crs.is_geographic and len(gdf):
        # area/perimeter in m^2 / m using an equal-area-ish local projection
        proj = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = proj.area.round(2)
        gdf["perim_m"] = proj.length.round(2)
    return gdf


def export_shapefiles(features: list[dict], crs, out_dir: str, prefix: str = "") -> list[str]:
    """Write one .shp per class. Returns list of written .shp paths."""
    os.makedirs(out_dir, exist_ok=True)
    by_class: dict[str, list[dict]] = {}
    for f in features:
        by_class.setdefault(f["label"], []).append(f)
    written = []
    for label, feats in by_class.items():
        gdf = to_geodataframe(feats, crs)
        if gdf.empty:
            continue
        path = os.path.join(out_dir, f"{prefix}{safe_name(label)}.shp")
        gdf.to_file(path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")
        written.append(path)
    return written


def zip_shapefiles(shp_paths: list[str], zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for shp in shp_paths:
            base = os.path.splitext(shp)[0]
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                if os.path.exists(base + ext):
                    z.write(base + ext, os.path.basename(base + ext))
    return zip_path
