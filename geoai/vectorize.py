"""Mask -> georeferenced polygons, cross-tile merging and geometry cleanup."""
from __future__ import annotations

import math

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
                 "water", "river", "field", "farmland", "parking lot", "parking", "sidewalk", "sidewalks",
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


def _main_angle(p: Polygon) -> float:
    c = list(p.minimum_rotated_rectangle.exterior.coords)
    a, b = max(((c[i], c[i + 1]) for i in range(4)), key=lambda e: math.dist(*e))
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _chaikin_poly(p: Polygon, n: int = 2) -> Polygon:
    pts = np.asarray(p.exterior.coords[:-1])
    for _ in range(n):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, 0); r = 0.25 * pts + 0.75 * np.roll(pts, -1, 0)
        pts = np.empty((len(q) * 2, 2)); pts[0::2] = q; pts[1::2] = r
    return Polygon(pts).buffer(0)


def _largest(g):
    if g.geom_type == "MultiPolygon":
        return max(g.geoms, key=lambda x: x.area)
    return g


def regularize_building(p: Polygon, tol: float) -> Polygon:
    """Make building footprints look like buildings.

    * nearly rectangular -> the rectangle
    * otherwise walls are squared to the building's own main direction: the outline is redrawn on
      a 25 cm grid aligned with the building and jogs smaller than ~2 m are removed, so walls are
      straight and corners are right angles (kept only when it still matches the shape, IoU >= 0.85)
    * round buildings (towers, domes) keep a smooth curve
    * anything else: small notches/spikes removed, edges straightened
    """
    p = _largest(p.buffer(0))
    mrr = p.minimum_rotated_rectangle
    if mrr.area > 0 and p.area / mrr.area > 0.85:
        return mrr
    x0, y0, x1, y1 = p.bounds
    if max(x1 - x0, y1 - y0) < 1e-3:                    # geographic coordinates: keep it simple
        s = p.simplify(tol, preserve_topology=True)
        return s if s.is_valid and not s.is_empty else p
    p = Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area >= 25]).buffer(0)
    p = _largest(p)
    if 4 * math.pi * p.area / p.length ** 2 > 0.75:     # round
        return _chaikin_poly(p.simplify(0.5))
    from rasterio import features
    from rasterio.transform import from_origin
    from scipy import ndimage
    from shapely import affinity
    from shapely.geometry import shape
    step, pad = 0.25, 2
    ang, c = _main_angle(p), p.centroid
    q = affinity.rotate(p, -ang, origin=c)
    bx0, by0, bx1, by1 = q.bounds
    W, H = int((bx1 - bx0) / step) + 2 * pad, int((by1 - by0) / step) + 2 * pad
    if W * H < 50_000_000:
        tr = from_origin(bx0 - pad * step, by1 + pad * step, step, step)
        m = features.rasterize([(q, 1)], out_shape=(H, W), transform=tr).astype(bool)
        k = 4                                            # 2 m jogs at 25 cm
        m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=k), iterations=k)
        shp = [shape(g) for g, _ in features.shapes(m.astype(np.uint8), mask=m, transform=tr)]
        if shp:
            r = max(shp, key=lambda g: g.area)
            r = Polygon(r.exterior, [h for h in r.interiors if Polygon(h).area >= 25]).simplify(0.01)
            r = affinity.rotate(r, ang, origin=c)
            if r.is_valid and r.intersection(p).area / max(r.union(p).area, 1e-9) >= 0.85:
                return r
    s = p.buffer(1.0, join_style=2).buffer(-1.0, join_style=2).buffer(-0.8, join_style=2).buffer(0.8, join_style=2)
    s = _largest(s) if not s.is_empty else p
    s = s.simplify(max(tol, 0.6), preserve_topology=True)
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


def safe_linemerge(geom):
    """shapely.linemerge that also accepts a single LineString / empty geometry."""
    from shapely.ops import linemerge as _lm
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    m = _lm(geom)
    return list(m.geoms) if hasattr(m, "geoms") else [m]
