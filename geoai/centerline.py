"""Centerlines from the road surface (medial axis), instead of the network model's shortcuts.

Tracing the middle of the detected road surface gives
  * a proper ring around roundabout islands (the island is a hole in the surface),
  * one centerline per carriageway on divided roads (the median is a hole),
  * lines that follow curves instead of cutting corners.
Short side branches ("spurs") that skeletons grow at junctions and bumps are pruned.
"""
from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_small_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    holes = ndimage.binary_fill_holes(mask) & ~mask
    lab, n = ndimage.label(holes)
    if not n:
        return mask
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    small = np.isin(lab, np.nonzero(sizes <= max_px)[0] + 1)
    return mask | small


def _chaikin(coords, iters=2):
    pts = np.asarray(coords, float)
    if len(pts) < 3:
        return pts
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2)); mid[0::2] = q; mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def surface_centerlines(surface: np.ndarray, transform: Affine, px_m: float,
                        min_spur_m: float = 6.0, small_hole_m2: float = 60.0, dedupe_m: float = 4.0) -> list[LineString]:
    from skimage.morphology import skeletonize

    m = fill_small_holes(surface, int(small_hole_m2 / (px_m * px_m)))     # cars, markings, buses
    m = _fill_thin_holes(m, max_halfwidth_px=max(1, int(0.8 / px_m)))     # lane markings, kerbs
    # smooth the outline: ragged edges are what grows skeleton side branches
    m = ndimage.gaussian_filter(m.astype(np.float32), sigma=max(1.0, 1.0 / px_m)) > 0.5
    m = ndimage.binary_opening(m, iterations=1)
    if not m.any():
        return []
    halfw = ndimage.distance_transform_edt(m) * px_m
    sk = skeletonize(m)
    H, W = sk.shape
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    deg = np.where(sk, nb, 0)
    nodes = sk & (deg != 2)
    nlab, _ = ndimage.label(nodes, structure=np.ones((3, 3)))

    # ---- trace edges between node clusters
    visited = np.zeros_like(sk)
    edges = []      # (node_a, node_b, [(r,c),...])

    def walk(r0, c0, r, c):
        """Follow degree-2 skeleton pixels from node pixel (r0,c0) via (r,c) to the next node."""
        path = [(r0, c0)]
        seen = {(r0, c0)}
        prev, cur = (r0, c0), (r, c)
        while True:
            path.append(cur); seen.add(cur)
            if nlab[cur]:
                return path, int(nlab[cur])
            visited[cur] = True
            nxt = None
            for dr, dc in _N8:
                n = (cur[0] + dr, cur[1] + dc)
                if not (0 <= n[0] < H and 0 <= n[1] < W) or not sk[n] or n == prev or n in seen:
                    continue
                if nlab[n] and (nlab[n] != nlab[r0, c0] or len(path) > 3):
                    nxt = n
                    break
                if not nlab[n] and not visited[n] and nxt is None:
                    nxt = n
            if nxt is None:
                return path, 0
            prev, cur = cur, nxt

    for r0, c0 in zip(*np.nonzero(nodes)):
        a = nlab[r0, c0]
        for dr, dc in _N8:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W) or not sk[r, c] or nlab[r, c] or visited[r, c]:
                continue
            path, b = walk(r0, c0, r, c)
            edges.append([a, b, path])
    # isolated loops (e.g. a roundabout ring with no arms inside the image)
    rest = sk & ~visited & ~nodes
    lab_r, nr = ndimage.label(rest, structure=np.ones((3, 3)))
    for k in range(1, nr + 1):
        rc = np.column_stack(np.nonzero(lab_r == k))
        if len(rc) > 8:
            # order points around their centroid
            ang = np.arctan2(rc[:, 0] - rc[:, 0].mean(), rc[:, 1] - rc[:, 1].mean())
            pts = [tuple(p) for p in rc[np.argsort(ang)]]
            edges.append([-k, -k, pts + pts[:1]])

    # ---- prune spurs: short branches that end in a dead end
    def length(p):
        a = np.asarray(p, float)
        return float(np.sum(np.hypot(*np.diff(a, axis=0).T))) * px_m if len(a) > 1 else 0.0

    for _ in range(4):
        degc = {}
        for a, b, _p in edges:
            degc[a] = degc.get(a, 0) + 1; degc[b] = degc.get(b, 0) + 1
        keep = []
        changed = False
        for e in edges:
            a, b, p = e
            dead = (b == 0) or degc.get(a, 0) == 1 or degc.get(b, 0) == 1
            hub = p[0] if degc.get(a, 0) > 1 else p[-1]
            thr = max(min_spur_m, 2.0 * float(halfw[hub]))
            border = any(x[0] <= 1 or x[1] <= 1 or x[0] >= H - 2 or x[1] >= W - 2 for x in (p[0], p[-1]))
            if dead and a != b and not border and length(p) < thr and (degc.get(a, 0) > 1 or degc.get(b, 0) > 1):
                changed = True
                continue
            keep.append(e)
        edges = keep
        if not changed:
            break

    # ---- to map coordinates, smooth, merge through degree-2 joints
    lines = []
    for _a, _b, p in edges:
        if len(p) < 2:
            continue
        xy = [transform * (c + 0.5, r + 0.5) for r, c in p]
        sm = _chaikin(LineString(xy).simplify(px_m * 1.5).coords)
        if len(sm) >= 2:
            lines.append(LineString(sm))
    if not lines:
        return []
    from .vectorize import safe_linemerge
    out = safe_linemerge(unary_union(lines))
    out = [g for g in out if g.geom_type == "LineString" and g.length > px_m * 3]
    return _dedupe_parallel(out, dedupe_m)


def _fill_thin_holes(mask, max_halfwidth_px: int):
    """Fill enclosed holes that are thin everywhere (painted markings, kerbs), keep wide ones (medians)."""
    holes = ndimage.binary_fill_holes(mask) & ~mask
    lab, n = ndimage.label(holes)
    if not n:
        return mask
    thick = ndimage.maximum(ndimage.distance_transform_edt(holes), lab, range(1, n + 1))
    thin = np.isin(lab, np.nonzero(np.asarray(thick) <= max_halfwidth_px)[0] + 1)
    return mask | thin


def _dedupe_parallel(lines, tol: float):
    """Drop a line that runs within `tol` metres of a longer line along most of its length."""
    if len(lines) < 2:
        return lines
    keep = []
    for ln in sorted(lines, key=lambda g: -g.length):
        if keep:
            near = unary_union([k for k in keep if k.distance(ln) < tol])
            if not near.is_empty and ln.intersection(near.buffer(tol)).length > 0.8 * ln.length:
                continue
        keep.append(ln)
    return keep
