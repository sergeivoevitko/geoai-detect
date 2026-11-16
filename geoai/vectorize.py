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


def _axial(a: float) -> float:
    return a % 180.0


def _adiff(a: float, b: float) -> float:
    d = abs(_axial(a) - _axial(b))
    return min(d, 180.0 - d)


def _aligned_share(p: Polygon, main: float, tol_deg: float = 15.0) -> float:
    """Share of the outline (after light simplification) running parallel or perpendicular to main."""
    c = list(p.simplify(0.5).exterior.coords)
    tot = ok = 0.0
    for a, b in zip(c[:-1], c[1:]):
        L = math.dist(a, b)
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        tot += L
        if min(_adiff(ang, main), _adiff(ang, main + 90)) <= tol_deg:
            ok += L
    return ok / tot if tot else 0.0


def snap_outline(p: Polygon, main: float | None = None, tol: float = 0.75, angle_tol: float = 15.0,
                 min_edge: float = 0.0) -> Polygon:
    """Straight walls: simplify, snap each edge to the main direction or its perpendicular when it
    is within angle_tol, keep other edges' own direction, merge consecutive edges that share a
    direction, and rebuild the corners where neighbouring walls meet (like ArcGIS Regularize
    Building Footprint). Returns the simplified outline if the result is not a valid polygon."""
    base = p.simplify(tol, preserve_topology=True)
    if base.is_empty or base.geom_type != "Polygon":
        return p
    if main is None:
        main = _main_angle(p)
    c = list(base.exterior.coords)[:-1]
    if len(c) < 4:
        return base
    # edges as (mid point, angle, length)
    edges = []
    for a, b in zip(c, c[1:] + c[:1]):
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        for target in (main, main + 90):
            if _adiff(ang, target) <= angle_tol:
                ang = _axial(target); break
        else:
            ang = _axial(ang)
        edges.append([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), ang, L])
    # merge consecutive edges with the same direction (length-weighted line through both)
    merged = True
    while merged and len(edges) > 3:
        merged = False
        for i in range(len(edges)):
            j = (i + 1) % len(edges)
            if _adiff(edges[i][1], edges[j][1]) < 1.0 or edges[j][2] < min_edge:
                (m1, a1, l1), (m2, a2, l2) = edges[i], edges[j]
                w = l1 + l2
                ang = a1 if l1 >= l2 else a2
                edges[i] = [((m1[0] * l1 + m2[0] * l2) / w, (m1[1] * l1 + m2[1] * l2) / w), ang, w]
                del edges[j]
                merged = True
                break
    if len(edges) < 3:
        return base
    pts = []
    for i in range(len(edges)):
        (m1, a1, _), (m2, a2, _) = edges[i - 1], edges[i]
        t1, t2 = math.radians(a1), math.radians(a2)
        d1, d2 = (math.cos(t1), math.sin(t1)), (math.cos(t2), math.sin(t2))
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < math.sin(math.radians(10)):      # near-parallel neighbours: step between them
            pts.append(m1 if not pts else pts[-1])
            pts.append(m2)
            continue
        t = ((m2[0] - m1[0]) * d2[1] - (m2[1] - m1[1]) * d2[0]) / den
        pts.append((m1[0] + d1[0] * t, m1[1] + d1[1] * t))
    r = Polygon(pts)
    if not r.is_valid:
        r = r.buffer(0)
        r = _largest(r) if not r.is_empty else r
    if r.is_empty or r.geom_type != "Polygon" or \
            r.intersection(p).area / max(r.union(p).area, 1e-9) < 0.8:
        return base
    return r


def _is_round(p: Polygon) -> bool:
    """A really round outline: compact, and every boundary point about the same distance from the
    centre (an irregular compact blob is not round)."""
    sm = p.simplify(0.3)
    if 4 * math.pi * sm.area / max(sm.length, 1e-9) ** 2 < 0.8:
        return False
    c = p.centroid
    ring = p.exterior
    d = np.array([c.distance(ring.interpolate(t, normalized=True)) for t in np.linspace(0, 1, 72, endpoint=False)])
    return float(d.std() / max(d.mean(), 1e-9)) < 0.07


def regularize_building(p: Polygon, tol: float) -> Polygon:
    """Make building footprints look like buildings.

    * nearly rectangular -> the rectangle
    * round buildings (towers, domes) keep a smooth curve
    * everything else -> wall snapping: walls near the main direction or its perpendicular become
      exactly parallel/perpendicular (right-angle corners), other walls stay straight in their own
      direction, short jogs are merged and corners are rebuilt where neighbouring walls meet
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
    if _is_round(p):                                     # round towers, domes
        return _chaikin_poly(p.simplify(0.5))
    ang = _main_angle(p)
    # light clean-up first: small notches and spikes (not for small or thin buildings it would eat)
    q0 = p.buffer(0.8, join_style=2).buffer(-0.8, join_style=2).buffer(-0.6, join_style=2).buffer(0.6, join_style=2)
    q0 = _largest(q0) if not q0.is_empty else p
    if q0.area < 0.85 * p.area:
        q0 = p
    r = snap_outline(q0, ang, tol=max(tol, 0.75), min_edge=1.0)
    if not r.is_valid or r.is_empty or r.intersection(p).area / max(r.union(p).area, 1e-9) < 0.75:
        r = snap_outline(p, ang, tol=0.4, min_edge=0.5)
    return r if r.is_valid and not r.is_empty else p


def regularize_panel(p: Polygon) -> Polygon:
    """Solar panels / collectors: a rectangle when the shape is close to one, otherwise straight
    edges snapped to the panel row direction."""
    p = _largest(p.buffer(0))
    if p.is_empty:
        return p
    mrr = p.minimum_rotated_rectangle
    if mrr.area > 0 and p.area / mrr.area > 0.7:
        return mrr
    x0, y0, x1, y1 = p.bounds
    if max(x1 - x0, y1 - y0) < 1e-3:
        return p.simplify(1e-6, preserve_topology=True)
    q = p.buffer(0.3, join_style=2).buffer(-0.3, join_style=2)
    q = _largest(q) if not q.is_empty else p
    r = snap_outline(q, _main_angle(q), tol=0.3, angle_tol=20.0, min_edge=0.4)
    return r if r.is_valid and not r.is_empty else q


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
