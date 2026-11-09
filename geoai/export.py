"""Write detections to Shapefiles (one per class) in the source image's CRS."""
from __future__ import annotations

import os
import re
import zipfile

import geopandas as gpd
import numpy as np


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
            **{k[:10]: v for k, v in (f.get("attrs") or {}).items()},   # shapefile: 10-char field names
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


def _layer_name(label: str) -> str:
    n = safe_name(label)
    return n if n[0].isalpha() else "c_" + n


def write_layers(features: list[dict], target_crs: str, out_dir: str, name: str,
                 formats=("gdb",)) -> list[str]:
    """Write all features, one layer per class, reprojected to target_crs.

    features: dicts with geometry, label, crs (source CRS of that feature), score, source, attrs.
    formats: "gdb" -> <name>.gdb (File Geodatabase, one feature class per class)
             "shp" -> <name>_shp/<class>.shp
    Returns written paths."""
    import pandas as pd
    from pyproj import CRS
    from shapely.ops import transform as stransform
    from pyproj import Transformer

    os.makedirs(out_dir, exist_ok=True)
    tcrs = CRS.from_user_input(target_crs)
    by_class: dict[str, list[dict]] = {}
    for f in features:
        by_class.setdefault(f["label"], []).append(f)
    frames = {}
    for label, feats in by_class.items():
        parts = []
        groups: dict = {}
        for f in feats:
            groups.setdefault(f.get("crs"), []).append(f)
        for crs, group in groups.items():
            gdf = to_geodataframe(group, crs)
            if crs and CRS.from_user_input(crs) != tcrs:
                gdf = gdf.to_crs(tcrs)
            parts.append(gdf)
        gdf = pd.concat(parts, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=tcrs)
        gdf["id"] = np.arange(1, len(gdf) + 1, dtype="int32")
        frames[_layer_name(label)] = gdf
    written = []
    if "gdb" in formats:
        gdb = os.path.join(out_dir, name + ".gdb")
        if os.path.exists(gdb):
            import shutil
            shutil.rmtree(gdb)
        for lname, gdf in frames.items():
            gdf.to_file(gdb, layer=lname, driver="OpenFileGDB", engine="pyogrio")
        written.append(gdb)
    if "shp" in formats:
        sdir = os.path.join(out_dir, name + "_shp")
        os.makedirs(sdir, exist_ok=True)
        for lname, gdf in frames.items():
            p = os.path.join(sdir, lname + ".shp")
            gdf.to_file(p, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")
        written.append(sdir)
    return written
