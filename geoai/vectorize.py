"""Mask -> georeferenced polygons, cross-tile merging and geometry cleanup."""
from __future__ import annotations

import numpy as np
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from scipy import ndimage

# Classes treated as continuous surfaces (dissolved into connected regions)
# rather than countable instances.
STUFF_CLASSES = {"road", "roads", "tree", "trees", "forest", "vegetation", "grass",
                 "water", "river", "field", "farmland", "parking lot", "sidewalk",
                 "impervious surface", "bare soil", "sand"}


def is_stuff(label: str) -> bool:
    return label.strip().lower() in STUFF_CLASSES


def clean_mask(mask: np.ndarray, min_pixels: int = 16, close_iter: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    if close_iter:
        m = ndimage.binary_closing(m, iterations=close_iter)
        m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(m) if m.sum() < 50_000 else m
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.isin(lab, np.nonzero(sizes >= min_pixels)[0] + 1)
        m = keep
    return m


def mask_to_polygons(mask: np.ndarray, transform: Affine) -> list[Polygon]:
    polys = []
    for geom, val in rasterio.features.shapes(mask.astype(np.uint8), mask=mask.astype(bool),
                                              transform=transform, connectivity=8):
        if val == 1:
            p = shape(geom)
            if not p.is_valid:
                p = p.buffer(0)
            polys.extend(_explode(p))
    return polys


def _explode(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        return [p for p in g.geoms if isinstance(p, Polygon)]
    return []


def regularize_building(p: Polygon, tol: float) -> Polygon:
    """Make building footprints look like buildings: straight edges, right angles
    when the shape is close to a rectangle."""
    mrr = p.minimum_rotated_rectangle
    if mrr.area > 0 and p.area / mrr.area > 0.85:
        return mrr
    s = p.simplify(tol, preserve_topology=True)
    return s if s.is_valid and not s.is_empty else p


def merge_instances(items: list[dict], iou_thr: float = 0.3, contain_thr: float = 0.6) -> list[dict]:
    """Merge duplicate detections from overlapping tiles.

    items: [{"geometry": Polygon, "score": float}]
    Two detections are the same object if IoU > iou_thr or one is mostly inside the other.
    Duplicates are unioned (so objects cut by a tile edge are re-assembled), keeping the max score.
    """
    if not items:
        return []
    geoms = [it["geometry"] for it in items]
    tree = STRtree(geoms)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            inter = g.intersection(geoms[j]).area
            if inter == 0:
                continue
            union = g.area + geoms[j].area - inter
            if inter / union > iou_thr or inter / min(g.area, geoms[j].area) > contain_thr:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    out = []
    for idx in groups.values():
        if len(idx) == 1:
            out.append(items[idx[0]])
            continue
        u = unary_union([geoms[k] for k in idx])
        parts = _explode(u)
        if not parts:
            continue
        g = max(parts, key=lambda q: q.area)
        out.append({"geometry": g, "score": max(items[k]["score"] for k in idx)})
    return out


def dissolve_stuff(items: list[dict]) -> list[dict]:
    """Union all polygons of a surface class and split into connected regions."""
    if not items:
        return []
    u = unary_union([it["geometry"].buffer(0) for it in items])
    scores = [it["score"] for it in items]
    tree = STRtree([it["geometry"] for it in items])
    out = []
    for p in _explode(u):
        hits = tree.query(p, predicate="intersects")
        s = float(np.mean([scores[int(h)] for h in hits])) if len(hits) else float(np.mean(scores))
        out.append({"geometry": p, "score": s})
    return out


def postprocess(label: str, items: list[dict], gsd: float, min_area: float,
                simplify_px: float = 1.0, regularize: bool = True) -> list[dict]:
    if is_stuff(label):
        merged = dissolve_stuff(items)
    else:
        merged = merge_instances(items)
    tol = simplify_px * gsd
    building = label.lower() in {"building", "buildings", "house", "houses", "roof"}
    if building:
        # sheds, cars and noise below this size are almost never real buildings
        min_area = max(min_area, 15.0)
    out = []
    for it in merged:
        g = it["geometry"]
        if building:
            # segmentation masks bleed ~1 px past roof edges (measured on NAIP ground truth:
            # +4 pts area IoU, +8 pts object F1)
            g = g.buffer(-gsd, join_style=2)
            if g.is_empty:
                continue
            g = max(_explode(g), key=lambda q: q.area)
        if g.area < min_area:
            continue
        if regularize and building:
            g = regularize_building(g, tol * 1.5)
        else:
            g = g.simplify(tol, preserve_topology=True)
        if not g.is_valid:
            g = g.buffer(0)
        for p in _explode(g):
            if p.area >= min_area:
                out.append({"geometry": p, "score": it["score"]})
    return out
