"""Road extraction: SAM-Road network + land-cover surface, adaptive width, junctions,
roundabouts and a per-segment agreement score.

Outputs
  * road surface mask (native resolution) -> road polygons
  * centerline features with attributes:
        width_m   measured road width
        kind      "road" | "roundabout"
        agree     agreement score 0-1 between the independent models
        review    1 if the segment should be checked by a human / AI verifier
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import rasterio.features
from rasterio.enums import Resampling
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point, box
from shapely.ops import polygonize, unary_union

WORK_GSD = 0.4   # m/px working grid for geometry (matches the land-cover model's resolution)


def _work_factor(gsd_m: float) -> int:
    return max(1, int(round(WORK_GSD / gsd_m)))


def _sample_line(ln: LineString, step: float) -> list[Point]:
    n = max(2, int(ln.length / step) + 1)
    return [ln.interpolate(i / (n - 1), normalized=True) for i in range(n)]


def extract_roads(img, cls_map: np.ndarray, idx_of: dict, opt, progress) -> tuple[list[dict], np.ndarray]:
    from . import samroad

    gsd = img.gsd_meters
    # ---------------------------------------------------------------- 1. network
    s = gsd / samroad.TRAIN_GSD
    W1, H1 = max(1, int(round(img.ds.width * s))), max(1, int(round(img.ds.height * s)))
    arr = img.ds.read(list(img.bands), out_shape=(3, H1, W1), resampling=Resampling.average, masked=True)
    data = np.ma.filled(arr, 0).astype(np.float32)
    if img._stretch is not None:
        lo, hi = img._stretch
        data = (data - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None] * 255
    rgb1 = np.clip(data, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    road_prob, nodes, edges, _ = samroad.infer(rgb1, progress=lambda p, m: progress(0.7 * p, m))
    t1 = img.ds.transform * Affine.scale(img.ds.width / W1, img.ds.height / H1)
    lines = [ln for ln in samroad.graph_to_lines(nodes, edges, t1) if ln.length > 0]
    H, W = cls_map.shape
    if not lines:
        return [], np.zeros((H, W), bool)
    progress(0.72, "Roads: bridging gaps, measuring width and junctions")

    # ------------------------------------------------- 2. working grid (~0.4 m)
    f = _work_factor(gsd)
    cw = cls_map[::f, ::f]
    Hw, Ww = cw.shape
    tw = img.ds.transform * Affine.scale(f, f)
    wg = gsd * f                                        # metres per working pixel
    road_c = cw == idx_of["road"]
    paved = road_c | (cw == idx_of["developed_space"]) if "developed_space" in idx_of else road_c.copy()
    # occluders: tree canopy and cast shadows (shadows are often mislabelled as water)
    from rasterio.windows import Window
    from .shadows import shadow_mask
    rgbw, _ = img.read_rgb(Window(0, 0, img.ds.width, img.ds.height)) if f == 1 else (None, None)
    if rgbw is None:
        arrw = img.ds.read(list(img.bands), out_shape=(3, Hw, Ww), resampling=Resampling.average, masked=True)
        dw = np.ma.filled(arrw, 0).astype(np.float32)
        if img._stretch is not None:
            lo, hi = img._stretch
            dw = (dw - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None] * 255
        rgbw = np.clip(dw, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    shadow = shadow_mask(rgbw[:Hw, :Ww], min_px=max(50, int(40 / (wg * wg))))
    building = cw == idx_of.get("building", 254)
    tree = cw == idx_of.get("tree_vegetation", 254)
    occluder = (tree | shadow) & ~building

    # bridge gaps the network model leaves at big junctions (shadows, crosswalks, wide asphalt)
    footprint = box(*img.ds.bounds).buffer(-3.0)
    bridges = _bridge_gaps(lines, paved, tw, footprint, opt.bridge_max_gap, occluder)
    lines = _node_network(lines + bridges)
    bridge_union = unary_union(bridges).buffer(0.5) if bridges else None
    junctions = _junctions(lines)

    center = rasterio.features.rasterize([(ln, 1) for ln in lines], out_shape=(Hw, Ww), transform=tw,
                                         all_touched=True, dtype=np.uint8).astype(bool)
    dist = ndimage.distance_transform_edt(~center).astype(np.float32) * wg   # metres to nearest centerline

    surface = ((road_c & (dist <= opt.road_wide_half_width)) |
               (paved & (dist <= opt.road_max_half_width)) |
               (dist <= opt.road_min_half_width))
    # junction disks: fill the corners of large intersections with paved pixels
    if junctions:
        jd = rasterio.features.rasterize([(p.buffer(opt.junction_radius), 1) for p in junctions],
                                         out_shape=(Hw, Ww), transform=tw, dtype=np.uint8).astype(bool)
        surface |= paved & jd
    # grow through connected "road" pixels: fills big intersections and wide junction areas
    # where the network model leaves gaps (shadows, crosswalks, open asphalt)
    steps = int(opt.road_grow_max / wg)
    if steps > 0:
        # Grow through road-like pixels (road class, shadows on the ground, pavement next to road).
        # A grown blob is kept only if it BRIDGES two or more separate parts of the road surface
        # (a junction between road arms, a shadow lying across a street). Blobs touching the road on
        # one side only (building shadows, forecourts) are dropped except for a thin edge strip.
        prior = surface.copy()
        grow_ok = (road_c | (shadow & ~building & ~tree) |
                   (paved & ndimage.binary_dilation(road_c, iterations=max(1, int(3 / wg)))))
        grown = ndimage.binary_dilation(prior, iterations=steps, mask=prior | grow_ok)
        added = grown & ~prior
        if added.any():
            lab_a, na = ndimage.label(added)
            ring = ndimage.binary_dilation(added, iterations=2) & prior
            lab_r, _ = ndimage.label(ring)
            keep = np.zeros(na + 1, bool)
            for k, sl in enumerate(ndimage.find_objects(lab_a), start=1):
                if sl is None:
                    continue
                pad = tuple(slice(max(0, x.start - 4), x.stop + 4) for x in sl)
                blob = lab_a[pad] == k
                touch = ndimage.binary_dilation(blob, iterations=3) & (lab_r[pad] > 0)
                keep[k] = len(np.unique(lab_r[pad][touch])) >= 2
            surface = prior | keep[lab_a] | (added & (dist <= opt.road_max_half_width))
    surface = ndimage.binary_closing(surface, iterations=2)
    lab, n = ndimage.label(surface)
    hit = np.unique(lab[center]); hit = hit[hit > 0]
    surface = np.isin(lab, hit)

    # -------------------------------------------------------- 3. roundabouts
    inside = ndimage.distance_transform_edt(surface).astype(np.float32) * wg   # half-width proxy
    ring_ids = set()
    islands = []
    faces = list(polygonize(unary_union(lines)))
    for face in faces:
        a = face.area
        if not (30 <= a <= 6000):
            continue
        compact = 4 * np.pi * a / (face.length ** 2)
        if compact < 0.7:
            continue
        # half width along the ring
        hw = _median_at(inside, tw, _sample_line(LineString(face.exterior.coords), 3.0))
        island = face.buffer(-max(hw, 1.5))
        if island.is_empty or island.area < 5:
            continue
        islands.append(island)
        for i, ln in enumerate(lines):
            if ln.within(face.buffer(1.0)) or face.exterior.buffer(1.0).contains(ln):
                ring_ids.add(i)
    if islands:
        im = rasterio.features.rasterize([(g, 1) for g in islands], out_shape=(Hw, Ww), transform=tw,
                                         dtype=np.uint8).astype(bool)
        surface &= ~im
        inside = ndimage.distance_transform_edt(surface).astype(np.float32) * wg

    # ------------------------------------------- 4. per-segment width + agreement
    progress(0.85, "Roads: scoring segments")
    inv1 = ~t1
    feats = []
    occl_corr = []
    for i, ln in enumerate(lines):
        pts = _sample_line(ln, 2.0)
        hws = _values_at(inside, tw, pts)
        # occlusion (trees, shadows) only ever narrows the visible road, so use an upper
        # percentile of the half-widths as the segment's true width
        width = float(2 * np.percentile(hws, 75)) if len(hws) else 0.0
        occl = float(np.mean(_values_at(occluder.astype(np.float32), tw, pts))) if pts else 0.0
        # SAM-Road road probability along the line (1 m grid)
        sr = []
        for p in pts:
            c, r = inv1 * (p.x, p.y)
            r, c = int(r), int(c)
            if 0 <= r < road_prob.shape[0] and 0 <= c < road_prob.shape[1]:
                sr.append(float(road_prob[max(0, r - 1):r + 2, max(0, c - 1):c + 2].max()))
        p_sr = float(np.clip(np.mean(sr) / 0.5, 0, 1)) if sr else 0.0
        # land-cover agreement: is the ground under the centerline paved?
        p_lc = float(np.mean(_values_at(paved.astype(np.float32), tw, pts))) if pts else 0.0
        cv = float(np.std(hws) / (np.mean(hws) + 1e-6)) if len(hws) > 2 else 0.0
        s_w = float(np.clip(1 - cv, 0, 1))
        agree = 0.4 * p_sr + 0.4 * p_lc + 0.2 * s_w
        # hidden road is inferred, not seen: lower agreement is expected there
        p_lc_eff = max(p_lc, min(1.0, p_lc + occl))
        agree = 0.4 * p_sr + 0.4 * p_lc_eff + 0.2 * s_w
        review = int(agree < opt.review_threshold or p_lc_eff < 0.4)
        feats.append({
            "geometry": ln.simplify(gsd * 2),
            "score": round(agree, 3),
            "attrs": {"width_m": round(width, 1),
                      "kind": "roundabout" if i in ring_ids else
                              ("gap_fill" if bridge_union is not None and ln.within(bridge_union) else "road"),
                      "agree": round(agree, 3), "p_samroad": round(p_sr, 3), "p_landcov": round(p_lc, 3),
                      "occluded": round(occl, 3), "review": review},
        })
        if occl > 0.05 and width > 0:
            occl_corr.append((ln, min(max(width / 2, opt.road_min_half_width), opt.road_wide_half_width)))

    # continue each road at its measured width under trees / shadows ("as if the tree were not there")
    if occl_corr and not os.environ.get("GEOAI_NO_OCCL"):
        cm = rasterio.features.rasterize([(ln.buffer(hw, cap_style=2), 1) for ln, hw in occl_corr],
                                         out_shape=(Hw, Ww), transform=tw, dtype=np.uint8).astype(bool)
        surface |= cm & occluder
        if islands:
            surface &= ~im

    # ----------------------------------------------- 5. back to native resolution
    if f > 1:
        surface = np.repeat(np.repeat(surface, f, 0), f, 1)[:H, :W]
        if surface.shape != (H, W):
            pad = np.zeros((H, W), bool); pad[:surface.shape[0], :surface.shape[1]] = surface; surface = pad
    progress(1.0, "Roads done")
    return feats, surface


def _end_dirs(ln: LineString, look: float = 8.0):
    """(endpoint, outward unit direction) for both ends of a line."""
    out = []
    L = ln.length
    for at_start in (True, False):
        p = ln.interpolate(0 if at_start else L)
        q = ln.interpolate(min(look, L) if at_start else max(0.0, L - look))
        v = np.array([p.x - q.x, p.y - q.y]); n = np.linalg.norm(v)
        if n > 0:
            out.append((p, v / n))
    return out


def _bridge_gaps(lines, paved, transform, footprint, max_gap: float, occluder=None) -> list[LineString]:
    """Join dangling ends that face each other (or face another road) across a paved gap."""
    ends = []
    for ln in lines:
        for p, d in _end_dirs(ln):
            if footprint.contains(p):            # ends at the image border are not gaps
                ends.append((p, d, ln))
    if not ends:
        return []
    network = unary_union(lines)
    out, used = [], set()
    for i, (p, d, ln) in enumerate(ends):
        if i in used:
            continue
        best = None
        # 1) another dangling end, roughly facing us
        for j, (q, e, ln2) in enumerate(ends):
            if j == i or j in used or ln2 is ln:
                continue
            v = np.array([q.x - p.x, q.y - p.y]); dist = np.linalg.norm(v)
            if dist == 0 or dist > max_gap:
                continue
            v /= dist
            if v @ d > 0.94 and -(v @ e) > 0.94:   # both within ~20 degrees of the gap
                if best is None or dist < best[0]:
                    best = (dist, LineString([p, q]), j)
        # 2) otherwise: straight ahead onto the rest of the network (T-junction)
        if best is None:
            t_gap = max_gap * 0.4
            ray = LineString([p, Point(p.x + d[0] * t_gap, p.y + d[1] * t_gap)])
            hit = ray.intersection(network.difference(ln.buffer(0.5)))
            if not hit.is_empty:
                pts = [hit] if hit.geom_type == "Point" else [g for g in getattr(hit, "geoms", []) if g.geom_type == "Point"]
                if hit.geom_type in ("LineString", "MultiLineString"):
                    pts = [Point(c) for g in getattr(hit, "geoms", [hit]) for c in g.coords]
                if pts:
                    q = min(pts, key=p.distance)
                    if p.distance(q) > 1.0:
                        best = (p.distance(q), LineString([p, q]), None)
        if best is None:
            continue
        seg = best[1]
        pts = _sample_line(seg, 1.0)
        vals = _values_at(paved.astype(np.float32), transform, pts)
        ok = len(vals) and vals.mean() >= 0.6
        if not ok and occluder is not None and len(vals):
            # hidden under trees / in shadow: accept if the gap is paved-or-covered throughout
            cov = _values_at((paved | occluder).astype(np.float32), transform, pts)
            ok = cov.mean() >= 0.85
        if ok:
            out.append(seg)
            used.add(i)
            if best[2] is not None:
                used.add(best[2])
    return out


def _node_network(lines) -> list[LineString]:
    """Split lines where they cross and merge through degree-2 vertices."""
    from shapely.ops import linemerge
    noded = unary_union(lines)
    merged = linemerge(noded)
    geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return [g for g in geoms if g.geom_type == "LineString" and g.length > 0.5]


def _junctions(lines) -> list[Point]:
    cnt = Counter()
    for ln in lines:
        for c in (ln.coords[0], ln.coords[-1]):
            cnt[(round(c[0], 1), round(c[1], 1))] += 1
    return [Point(k) for k, v in cnt.items() if v >= 3]


def _values_at(arr: np.ndarray, transform: Affine, pts) -> np.ndarray:
    inv = ~transform
    out = []
    for p in pts:
        c, r = inv * (p.x, p.y)
        r, c = int(r), int(c)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            out.append(arr[r, c])
    return np.array(out, np.float32)


def _median_at(arr, transform, pts) -> float:
    v = _values_at(arr, transform, pts)
    return float(np.median(v)) if len(v) else 0.0
