"""Road, sidewalk, crosswalk, parking and centerline layers from several models.

Pipeline (each step is conservative: a piece is only added when two sources agree):

  1. fusion       Tile2Net road is the base. SAM 3 "road" pieces fill gaps only where they connect
                  road on two sides AND a SAM-Road centerline runs through them (bridge shadows,
                  big junctions, a lane through a square). SAM-Road centerline segments outside the
                  visible road become road when they connect to the network and are not in a
                  parking lot, on a building or on a plaza (streets missed at coarse resolution,
                  road under tree canopy -> source "occluded").
  2. network      centerlines traced from the fused road, stubs pruned, gaps <= 25 m closed, one
                  node per junction, grade-separated crossings (bridges) left unconnected, straight
                  where straight, true circles for roundabouts; width / lanes / class / source.
  3. curb line    road and sidewalk share one edge (no overlap).
  4. sidewalks    gaps filled on street sides that already have >= 40 % detected sidewalk.
  5. islands      planted / unpaved islands left out of the road, paved ones join the sidewalk.
  6. smoothing    pixel steps removed: straight edges, rounded corners.

All distances are in the raster CRS units (metres for projected CRSs).
"""
from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, shape
from shapely.ops import linemerge, nearest_points, substring, unary_union

from .centerline import surface_centerlines

EMPTY = GeometryCollection()


# ------------------------------------------------------------------ helpers
def polys(g) -> list:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and not p.is_empty]


def lines_in(g) -> list:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    out = []
    for p in getattr(g, "geoms", []):
        out += lines_in(p)
    return out


class Grid:
    """A coarser working grid over the raster (res in CRS units per pixel)."""

    def __init__(self, img, res: float):
        ds = img.ds
        self.f = max(1, int(round(res / img.gsd)))
        self.H, self.W = max(1, ds.height // self.f), max(1, ds.width // self.f)
        self.tr = ds.transform * rasterio.Affine.scale(ds.width / self.W, ds.height / self.H)
        self.inv = ~self.tr
        self.px = abs(self.tr.a)

    def burn(self, geom) -> np.ndarray:
        if geom is None or geom.is_empty:
            return np.zeros((self.H, self.W), bool)
        return features.rasterize([(geom, 1)], out_shape=(self.H, self.W), transform=self.tr).astype(bool)

    def down(self, mask: np.ndarray) -> np.ndarray:
        """native-resolution bool mask -> this grid (majority)."""
        H, W = self.H * self.f, self.W * self.f
        m = mask[:H, :W]
        if m.shape != (H, W):
            m = np.pad(m, ((0, H - m.shape[0]), (0, W - m.shape[1])))
        return m.reshape(self.H, self.f, self.W, self.f).mean(axis=(1, 3)) >= 0.5

    def vec(self, mask: np.ndarray, min_area: float = 2.0):
        if not mask.any():
            return EMPTY
        g = [shape(s).buffer(0) for s, _ in features.shapes(mask.astype(np.uint8), mask=mask, transform=self.tr)]
        return unary_union([p for p in g if p.area >= min_area])

    def sample(self, field: np.ndarray, ln, step: float = 2.0) -> np.ndarray:
        v = []
        for t in np.arange(0, ln.length + 1e-6, step):
            p = ln.interpolate(t)
            c, r = self.inv * (p.x, p.y)
            if 0 <= int(r) < self.H and 0 <= int(c) < self.W:
                v.append(field[int(r), int(c)])
        return np.array(v)

    def at(self, field: np.ndarray, x: float, y: float, default=0.0):
        c, r = self.inv * (x, y)
        return field[int(r), int(c)] if 0 <= int(r) < self.H and 0 <= int(c) < self.W else default


def read_rgb(img, grid: Grid) -> np.ndarray:
    a = img.ds.read(list(img.bands), out_shape=(3, grid.H, grid.W), resampling=Resampling.average, masked=True)
    d = np.ma.filled(a, 0).astype(np.float32)
    if img._stretch is not None:
        lo, hi = img._stretch
        d = (d - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None] * 255
    return np.clip(d, 0, 255)


def chaikin_ring(coords, n=2):
    pts = np.asarray(coords[:-1])
    if len(pts) < 3:
        return np.asarray(coords)
    for _ in range(n):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, 0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, 0)
        pts = np.empty((len(q) * 2, 2)); pts[0::2] = q; pts[1::2] = r
    return np.vstack([pts, pts[:1]])


def chaikin_line(coords, n=2):
    pts = np.asarray(coords)
    for _ in range(n):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]; r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((len(q) * 2, 2)); mid[0::2] = q; mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def smooth_area(g, close=1.0, open_=1.0, simplify=0.75, min_hole=20.0, min_area=4.0):
    """Remove pixel steps: close small notches, cut spikes, straighten, round corners."""
    if g is None or g.is_empty:
        return EMPTY
    g = g.buffer(close, join_style=1).buffer(-(close + open_), join_style=1).buffer(open_, join_style=1)
    g = g.simplify(simplify, preserve_topology=True)
    out = []
    for p in polys(g):
        if p.area < min_area:
            continue
        holes = [chaikin_ring(h.coords) for h in p.interiors if Polygon(h).area >= min_hole]
        q = Polygon(chaikin_ring(p.exterior.coords), holes).buffer(0)
        out += polys(q)
    return unary_union(out) if out else EMPTY


# ------------------------------------------------------------------ 1. fusion
def fuse(base, t_side, s_road, s_park, bld, cl_old, width_field, grid: Grid, green):
    """Returns (fused road, list of (piece, source)) ."""
    added = []
    # -- SAM 3 gap fills
    if not s_road.is_empty:
        cand = s_road.difference(base.buffer(0.3)).difference(t_side).difference(bld.buffer(0.5))
        cand = cand.difference(s_park.buffer(1.0)).buffer(-1.0).buffer(1.0)
        for p in polys(cand):
            if p.area < 25:
                continue
            c = p.exterior.buffer(1.0).intersection(base).buffer(0.5)
            if len([q for q in polys(c) if q.area > 2]) < 2:          # must bridge a gap
                continue
            ln = p.intersection(cl_old)
            r = p.minimum_rotated_rectangle
            sides = sorted(LineString(r.exterior.coords[i:i + 2]).length for i in range(4))
            if ln.is_empty or ln.length < min(8.0, 0.5 * sides[-1]):  # a road runs through it
                continue
            added.append((p, "detected"))
    fused = unary_union([base] + [p for p, _ in added]).buffer(0.5).buffer(-0.5)

    # -- SAM-Road centerline fill
    near = fused.buffer(3.0)
    for ln in lines_in(cl_old.difference(fused.buffer(1.0))):
        if ln.length < 10:
            continue
        ends = [Point(ln.coords[0]), Point(ln.coords[-1])]
        n_conn = sum(near.contains(e) for e in ends)
        if n_conn == 0 or (n_conn == 1 and ln.length < 30):
            continue
        if ln.intersection(s_park.buffer(1.0)).length > 0.3 * ln.length:
            continue
        if ln.intersection(bld).length > 0.2 * ln.length:
            continue
        v = grid.sample(width_field, ln)
        v = v[v > 0]
        hw = float(np.clip(np.median(v), 2.5, 7.0)) if len(v) >= 3 else 3.0    # >= 5 m wide
        strip = ln.buffer(hw, cap_style=2, join_style=1)
        if strip.intersection(t_side).area > 0.3 * strip.area:                 # plaza
            continue
        g = grid.sample(green.astype(np.float32), ln, 1.0)
        src = "occluded" if len(g) and g.mean() >= 0.5 else "inferred"
        piece = strip.difference(bld.buffer(0.3)).difference(s_park)
        if not piece.is_empty:
            added.append((piece, src))
    if added:
        fused = unary_union([fused] + [p for p, _ in added]).buffer(0.5).buffer(-0.5)
    return fused, added


# ------------------------------------------------------------------ 2. network
def _key(p, q=0.01):
    return (round(p[0] / q) * q, round(p[1] / q) * q)


def _graph(lines):
    g = nx.MultiGraph()
    for ln in lines:
        if ln.length < 0.3:
            continue
        g.add_edge(_key(ln.coords[0]), _key(ln.coords[-1]), geom=ln)
    return g


def _edges(g):
    return [d["geom"] for _, _, d in g.edges(data=True)]


def _node(lines):
    """Split lines where they cross or touch (planar noding)."""
    return [l for l in lines_in(unary_union(lines)) if l.length > 0.3]


def _prune(lines, min_len=15.0, rounds=3):
    for _ in range(rounds):
        g = _graph(lines)
        drop = set()
        for u, v, d in g.edges(data=True):
            ln, du, dv = d["geom"], g.degree(u), g.degree(v)
            if ln.length < min_len and (du == 1) != (dv == 1) and max(du, dv) >= 3:
                drop.add(id(ln))
            elif ln.length < 6 and du == 1 and dv == 1:
                drop.add(id(ln))
        if not drop:
            break
        lines = [l for l in _edges(g) if id(l) not in drop]
    return lines


def _close_gaps(lines, park, bld, max_gap=25.0):
    g = _graph(lines)
    net = unary_union(lines)
    links = []
    for node in [n for n in g.nodes if g.degree(n) == 1]:
        (_, _, d), = list(g.edges(node, data=True))
        ln = d["geom"]
        at_start = _key(ln.coords[0]) == node
        a = ln.interpolate(0 if at_start else ln.length)
        b = ln.interpolate(min(10, ln.length) if at_start else max(0, ln.length - 10))
        v = np.array([a.x - b.x, a.y - b.y]); n = np.linalg.norm(v)
        if n == 0:
            continue
        v /= n
        others = net.difference(ln.buffer(0.5))
        for t in np.arange(2, max_gap + 1, 1.0):
            c = Point(a.x + v[0] * t, a.y + v[1] * t)
            hit = c.buffer(max(1.5, t * math.tan(math.radians(35)))).intersection(others)
            if not hit.is_empty:
                link = LineString([a, nearest_points(c, hit)[1]])
                if link.length <= max_gap and link.intersection(park).length <= 0.5 * link.length \
                        and link.intersection(bld).length <= 0.2 * link.length:
                    links.append(link)
                break
    return links


def _direction(ln, from_end_key, look=15.0):
    if _key(ln.coords[0]) == from_end_key:
        a, b = ln.interpolate(0), ln.interpolate(min(look, ln.length))
    else:
        a, b = ln.interpolate(ln.length), ln.interpolate(max(0, ln.length - look))
    return math.atan2(b.y - a.y, b.x - a.x)


def _angdiff(a, b):
    d = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    return d


def _junctions(lines, halfwidth, grid: Grid, crosswalks):
    """One node per junction; grade-separated crossings (bridges) stay unconnected.

    A 4-way crossing is treated as a bridge when it is two straight through-roads, at least one
    of them wide (>= 12 m), the arms are long and no crosswalk is near: at-grade crossings of
    major roads in Israeli cities nearly always have crosswalks."""
    g = _graph(lines)
    J = [n for n in g.nodes if g.degree(n) >= 3]
    width = {n: 2 * grid.at(halfwidth, n[0], n[1]) for n in J}
    jg = nx.Graph(); jg.add_nodes_from(J)
    for i, a in enumerate(J):
        for b in J[i + 1:]:
            r = float(np.clip(0.6 * max(width[a], width[b]), 12.0, 30.0))
            if math.dist(a, b) < r:
                jg.add_edge(a, b)
    moves, through, drop, n_sep = {}, [], set(), 0
    for comp in nx.connected_components(jg):
        comp = set(comp)
        c = tuple(np.mean(np.array(list(comp)), axis=0))
        internal, branches = [], []
        for n in comp:
            for u, v, d in g.edges(n, data=True):
                ln = d["geom"]
                other = v if u == n else u
                if other in comp and ln.length < 30:
                    internal.append(ln)
                elif other not in comp or ln.length >= 30:
                    branches.append((n, ln))
        # de-duplicate branches seen from both ends
        seen, br = set(), []
        for n, ln in branches:
            if id(ln) not in seen:
                seen.add(id(ln)); br.append((n, ln))
        sep = False
        if len(br) == 4 and (crosswalks is None or crosswalks.is_empty or crosswalks.distance(Point(c)) > 20):
            ang = [_direction(ln, n) for n, ln in br]
            best = None
            for pairing in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
                dev = [abs(math.pi - _angdiff(ang[i], ang[j])) for i, j in pairing]
                if best is None or sum(dev) < best[0]:
                    best = (sum(dev), pairing, dev)
            _, pairing, dev = best
            if max(dev) < math.radians(30):
                (i1, j1), (i2, j2) = pairing
                axis = lambda i, j: (ang[i] - ang[j]) / 2
                cross = _angdiff(ang[i1], ang[i2])
                cross = min(cross, math.pi - cross)
                long_arms = all(ln.length >= 25 for _, ln in br)
                wide = max(2 * grid.at(halfwidth, *br[k][1].interpolate(0.5, normalized=True).coords[0])
                           for k in range(4)) >= 12
                if cross > math.radians(25) and long_arms and wide:
                    sep = True
                    for i, j in pairing:
                        (ni, li), (nj, lj) = br[i], br[j]
                        ci = list(li.coords) if _key(li.coords[-1]) == ni else list(li.coords)[::-1]
                        cj = list(lj.coords) if _key(lj.coords[0]) == nj else list(lj.coords)[::-1]
                        through.append(LineString(ci[:-1] + [c] + cj[1:]))
                        drop.update((id(li), id(lj)))
                    drop.update(id(l) for l in internal)
                    n_sep += 1
        if not sep:
            if len(comp) > 1:
                for n in comp:
                    moves[n] = c
                drop.update(id(l) for l in internal)
    out = []
    for ln in _edges(g):
        if id(ln) in drop:
            continue
        cs = list(ln.coords)
        a, b = _key(cs[0]), _key(cs[-1])
        if a in moves:
            cs[0] = moves[a]
        if b in moves:
            cs[-1] = moves[b]
        if len(cs) >= 2 and LineString(cs).length > 0.5:
            out.append(LineString(cs))
    return out + through, n_sep


def _merge_chains(lines):
    """Join pieces between junctions into one line (crossing lines are not split)."""
    m = linemerge(MultiLineString(lines)) if lines else EMPTY
    return [l for l in lines_in(m) if l.length > 1]


def _shape_lines(lines):
    """Straight where straight, smooth curves, true circular arcs on roundabouts."""
    def fit_circle(pts):
        x, y = np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones(len(x))]
        (cx, cy, c0), *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
        r = math.sqrt(max(c0 + cx ** 2 + cy ** 2, 0))
        return cx, cy, r, float(np.mean(np.abs(np.hypot(x - cx, y - cy) - r)))

    arcs = []
    for ln in lines:
        if ln.length < 20 or len(ln.coords) < 6:
            continue
        pts = [ln.interpolate(t).coords[0] for t in np.linspace(0, ln.length, 30)]
        cx, cy, r, res = fit_circle(pts)
        if 8 < r < 90 and res < 0.06 * r + 0.5:
            ang = np.unwrap([math.atan2(py - cy, px - cx) for px, py in pts])
            if abs(ang[-1] - ang[0]) > math.radians(70):
                arcs.append((cx, cy, r, ang[0], ang[-1]))
    groups = []
    for cx, cy, r, a0, a1 in arcs:
        for gr in groups:
            (gx, gy), rr = gr["c"], gr["r"]
            if math.hypot(gx - cx, gy - cy) < 0.25 * rr + 3 and abs(rr - r) < 0.25 * rr + 3:
                n = len(gr["arcs"]); gr["c"] = ((gx * n + cx) / (n + 1), (gy * n + cy) / (n + 1))
                gr["r"] = (rr * n + r) / (n + 1); gr["arcs"].append((a0, a1)); break
        else:
            groups.append({"c": (cx, cy), "r": r, "arcs": [(a0, a1)]})
    circles = []
    for gr in groups:
        cover = np.zeros(360, bool)
        for a0, a1 in gr["arcs"]:
            lo, hi = sorted((math.degrees(a0), math.degrees(a1)))
            for a in range(int(lo), int(hi) + 1):
                cover[a % 360] = True
        if cover.sum() >= 270 and 8 < gr["r"] < 60:
            circles.append((Point(*gr["c"]), gr["r"]))

    out = []
    for ln in lines:
        done = False
        for c, r in circles:
            dd = [abs(math.hypot(x - c.x, y - c.y) - r) for x, y in ln.coords]
            if np.mean(dd) < 0.12 * r + 1.0:
                a0 = math.atan2(ln.coords[0][1] - c.y, ln.coords[0][0] - c.x)
                a1 = math.atan2(ln.coords[-1][1] - c.y, ln.coords[-1][0] - c.x)
                mid = ln.interpolate(0.5, normalized=True)
                am = math.atan2(mid.y - c.y, mid.x - c.x)
                span = (a1 - a0) % (2 * math.pi)
                if not (0 <= (am - a0) % (2 * math.pi) <= span):
                    span -= 2 * math.pi
                n = max(4, int(abs(span) * r / 2))
                inner = [(c.x + r * math.cos(a0 + span * t / n), c.y + r * math.sin(a0 + span * t / n))
                         for t in range(1, n)]
                out.append((LineString([ln.coords[0]] + inner + [ln.coords[-1]]), "roundabout"))
                done = True
                break
        if done:
            continue
        chord = LineString([ln.coords[0], ln.coords[-1]])
        dev = max(chord.distance(Point(p)) for p in ln.coords) if chord.length > 0 else 99
        if ln.length > 15 and dev < max(1.0, 0.02 * ln.length):
            out.append((chord, "straight"))
        else:
            out.append((LineString(chaikin_line(ln.simplify(0.75).coords, 2)), "curve"))
    return out


def network(road, park, bld, crosswalks, added, grid: Grid):
    """Centerlines with attributes + the road surface with even edges."""
    m = grid.burn(road)
    halfw = ndimage.distance_transform_edt(m) * grid.px
    lines = surface_centerlines(m, grid.tr, grid.px, min_spur_m=10.0)
    lines = _node(lines)
    lines = _prune(lines)
    links = _close_gaps(lines, park, bld)
    lines = _node(lines + links)
    lines, n_sep = _junctions(lines, halfw, grid, crosswalks)
    lines = _prune(lines)
    lines = _merge_chains(lines)
    shaped = _shape_lines(lines)

    link_zone = unary_union(links).buffer(1.0) if links else EMPTY
    occl_zone = unary_union([p for p, s in added if s == "occluded"]) if added else EMPTY
    inf_zone = unary_union([p for p, s in added if s == "inferred"]) if added else EMPTY
    recs, corridors, inferred = [], [], []
    for ln, kind in shaped:
        hw = grid.sample(halfw, ln)
        seen_frac = float(np.mean(hw > 0)) if len(hw) else 0.0
        hw = hw[hw > 0]
        if len(hw) >= 3:
            w = float(2 * np.median(hw)); cv = float(np.std(hw) / (np.mean(hw) + 1e-6))
        else:
            w, cv = 6.0, 1.0
        w = float(np.clip(w, 4.0, 40.0))
        if not occl_zone.is_empty and ln.intersection(occl_zone).length > 0.5 * ln.length:
            src = "occluded"
        elif (not link_zone.is_empty and ln.within(link_zone)) or seen_frac < 0.5 or \
                (not inf_zone.is_empty and ln.intersection(inf_zone).length > 0.5 * ln.length):
            src = "inferred"
        else:
            src = "detected"
        recs.append({"geometry": ln, "attrs": {
            "width_m": round(w, 1), "lanes": max(1, int(round(w / 3.3))),
            "road_class": "highway" if w >= 18 else ("main" if w >= 10 else "local"),
            "shape": kind, "review": int(src != "detected")}, "source": src})
        if src != "detected":
            inferred.append(ln.buffer(w / 2, cap_style=2, join_style=1))
        if cv < 0.25 and ln.length > 12:
            corridors.append(ln.buffer(w / 2, cap_style=1, join_style=1, resolution=8))
    corr = unary_union(corridors) if corridors else EMPTY
    rest = road.difference(corr.buffer(1.5)).buffer(-1.0).buffer(1.0)
    rest = unary_union([p for p in polys(rest) if p.area > 40])
    surface = unary_union([corr, rest] + inferred)
    surface = surface.buffer(3.0, join_style=1).buffer(-3.0, join_style=1)
    # keep the holes (islands, roundabout centres) the surface had before evening the edges
    holes = unary_union([Polygon(h) for p in polys(road) for h in p.interiors if Polygon(h).area >= 10])
    if not holes.is_empty:
        surface = surface.difference(holes.buffer(-0.3))
    if not bld.is_empty:
        surface = surface.difference(bld)
    net = unary_union([r["geometry"] for r in recs]) if recs else EMPTY
    surface = unary_union([p for p in polys(surface) if p.intersects(net) or p.area > 2000])
    return recs, surface, {"links": len(links), "grade_separated": n_sep}


# ------------------------------------------------------------------ 3-5. curb, sidewalks, islands
def curb(road, side, centerlines):
    """Road and sidewalk share one edge. Wide overlaps on a centerline are road, thin ones sidewalk."""
    ov = road.intersection(side)
    to_side = [p for p in polys(ov) if p.area >= 1 and not (not p.buffer(-1.5).is_empty and p.intersects(centerlines))]
    if to_side:
        road = road.difference(unary_union(to_side).buffer(0.05))
    side = side.difference(road.buffer(0.05))
    return road, side


def complete_sidewalks(road, side, recs, blocked, min_cover=0.40, max_sw=5.0, step=2.0):
    def box_(ln, t0, t1, sgn, depth):
        seg = substring(ln, t0, t1)
        return seg.buffer(sgn * depth, single_sided=True, cap_style=2) if seg.length >= 0.5 else EMPTY

    fills = []
    for r in recs:
        ln, a = r["geometry"], r["attrs"]
        if ln.length < 20 or a["road_class"] == "highway":
            continue
        hw = a["width_m"] / 2
        for sgn in (1, -1):
            band = ln.buffer(sgn * (hw + max_sw), single_sided=True, cap_style=2).difference(road)
            sws = side.intersection(band)
            ts = np.arange(4.0, ln.length - 4.0, step)
            if len(ts) < 4 or sws.is_empty:
                continue
            cov = np.array([box_(ln, t, t + step, sgn, hw + max_sw).intersection(sws).area > 0.5 for t in ts])
            if cov.mean() < min_cover or cov.all():
                continue
            w = float(np.clip(sws.area / max(cov.sum() * step, 1), 1.5, max_sw))
            strip = ln.buffer(sgn * (hw + w), single_sided=True, cap_style=2).difference(road)
            gaps = unary_union([box_(ln, t, t + step, sgn, hw + w + 1) for t, c in zip(ts, cov) if not c])
            f = strip.intersection(gaps).difference(blocked).difference(side).buffer(-0.4).buffer(0.4)
            own = ln.buffer(hw + 1.0, cap_style=2)
            for p in polys(f):
                if p.area < 4:
                    continue
                if road.intersection(p.buffer(4.0)).difference(own).area > 5:   # a median
                    continue
                fills.append(p)
    return fills


def islands(road, side_detected, grid: Grid, green):
    """Holes and painted-over patches inside the road. Paved (Tile2Net saw sidewalk) -> sidewalk;
    anything else (planting, dirt, roundabout centres) is left out of the road."""
    rm = grid.burn(road)
    filled = ndimage.binary_fill_holes(rm)
    holes = filled & ~rm
    painted = rm & ndimage.binary_opening(green, iterations=2)
    inner = ndimage.binary_erosion(filled, iterations=2)
    lab, _ = ndimage.label(holes | painted)
    paved, other = [], []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        blob = lab[sl] == i
        area = blob.sum() * grid.px ** 2
        if not (10 <= area <= 4000) or inner[sl][blob].mean() <= 0.9:
            continue
        ring = ndimage.binary_dilation(blob, iterations=max(1, int(1.5 / grid.px))) & ~blob
        if not ring.any() or rm[sl][ring].mean() < 0.75:
            continue
        if ndimage.binary_erosion(blob, iterations=max(1, int(1.0 / grid.px))).sum() == 0:
            continue
        m = np.zeros((grid.H, grid.W), bool); m[sl] = blob
        p = grid.vec(m, 1.0).buffer(0.3).buffer(-0.3).simplify(0.3)
        g = green[sl][blob].mean()
        sw = p.intersection(side_detected).area / max(p.area, 1e-6)
        (paved if (sw > 0.5 and g <= 0.35) else other).append(p)
    return unary_union(paved) if paved else EMPTY, unary_union(other) if other else EMPTY


def crosswalk_shape(p, road):
    """Detected crosswalks have zigzag edges (the stripes). A crosswalk is a rectangle, so it is
    replaced by its best-fitting rotated rectangle; bent ones (across a curve or a corner) are
    split along their length and each part gets its own rectangle. Clipped to the road."""
    p = p.buffer(0.6, join_style=2).buffer(-0.6, join_style=2)       # close the gaps between stripes
    if p.is_empty or p.area < 3:
        return []
    r = p.minimum_rotated_rectangle
    if p.area / max(r.area, 1e-9) >= 0.55:
        parts = [r]
    else:
        c = list(r.exterior.coords)
        e = [(c[i], c[i + 1]) for i in range(4)]
        long_ = max(e, key=lambda s: math.dist(*s))
        L = math.dist(*long_)
        n = max(2, min(4, int(round(L / 6))))
        ux, uy = (long_[1][0] - long_[0][0]) / L, (long_[1][1] - long_[0][1]) / L
        parts = []
        for k in range(n):
            a0, a1 = L * k / n, L * (k + 1) / n
            ox, oy = long_[0]
            cut = Polygon([(ox + ux * a0 - uy * 1e3, oy + uy * a0 + ux * 1e3), (ox + ux * a1 - uy * 1e3, oy + uy * a1 + ux * 1e3),
                           (ox + ux * a1 + uy * 1e3, oy + uy * a1 - ux * 1e3), (ox + ux * a0 + uy * 1e3, oy + uy * a0 - ux * 1e3)])
            q = p.intersection(cut)
            if q.area >= 1:
                parts.append(q.minimum_rotated_rectangle)
    out = []
    for q in sorted(parts, key=lambda x: -x.area):
        if road is not None and not road.is_empty and q.intersection(road.buffer(0.5)).area < 0.5 * q.area:
            continue
        if out:
            q = q.difference(unary_union(out))
        out += [x for x in polys(q) if x.area >= 3]
    return out


def crosswalks_from_stripes(img, blob, road):
    """Rebuild a detected crosswalk area from its painted stripes.

    White stripes inside the blob are found in the image; stripes with the same direction that lie
    next to each other form one crosswalk, drawn as a rectangle aligned with the stripes. This
    splits blobs that merged two crosswalks and drops paint/asphalt the model spilled onto.
    Returns [] if no stripes are found (e.g. coarse imagery); the caller then falls back."""
    from rasterio.windows import from_bounds
    from shapely import affinity
    gsd = img.gsd
    if gsd > 0.25:                                     # stripes ~0.5 m wide: need sharp imagery
        return []
    x0, y0, x1, y1 = blob.buffer(1.5).bounds
    win = from_bounds(x0, y0, x1, y1, img.ds.transform).round_offsets().round_lengths()
    if win.width < 8 or win.height < 8:
        return []
    rgb, valid = img.read_rgb(win)
    g = rgb.astype(np.float32).mean(2)
    tr = img.ds.window_transform(win)
    inside = features.rasterize([(blob.buffer(0.5), 1)], out_shape=g.shape, transform=tr).astype(bool) & valid
    if inside.sum() < 30:
        return []
    # local contrast: a stripe is brighter than the asphalt ~1.5 m around it, also in shadow
    k = max(5, int(round(3.0 / gsd)) | 1)
    bg = ndimage.median_filter(g, size=k)
    d = g - bg
    spread = np.percentile(d[inside], 95) - np.percentile(d[inside], 50)
    if spread < 8:
        return []
    white = inside & (d > max(6.0, 0.35 * spread))
    white = ndimage.binary_opening(white, iterations=1)
    stripes = []
    for s_, _ in features.shapes(white.astype(np.uint8), mask=white, transform=tr):
        q = shape(s_)
        if not (0.25 <= q.area <= 6.0):
            continue
        r = q.minimum_rotated_rectangle
        c = list(r.exterior.coords)
        e = sorted((math.dist(c[i], c[i + 1]), c[i], c[i + 1]) for i in range(4))
        short, long_ = e[0][0], e[-1][0]
        if short < 0.15 or long_ / short < 2.2 or not (1.2 <= long_ <= 6.0):
            continue
        a = math.atan2(e[-1][2][1] - e[-1][1][1], e[-1][2][0] - e[-1][1][0]) % math.pi
        stripes.append((q, a))
    if len(stripes) < 3:
        return []
    # group: similar direction and close together
    n = len(stripes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            da = abs(stripes[i][1] - stripes[j][1]); da = min(da, math.pi - da)
            if da < math.radians(15) and stripes[i][0].distance(stripes[j][0]) < 2.5:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    o = blob.centroid
    body = blob.buffer(0.6, join_style=2).buffer(-0.6, join_style=2)
    rects = []                                           # (angle, rotated-frame box)
    for idx in groups.values():
        if len(idx) < 4:                                  # a zebra crossing has many stripes
            continue
        angs = np.array([stripes[i][1] for i in idx])
        a = math.atan2(np.sin(2 * angs).mean(), np.cos(2 * angs).mean()) / 2   # mean axial angle
        u = affinity.rotate(unary_union([stripes[i][0] for i in idx]), -a, origin=o, use_radians=True)
        rb = affinity.rotate(body, -a, origin=o, use_radians=True)
        ux0, uy0, ux1, uy1 = u.bounds                  # stripes lie along x in this frame
        band = Polygon([(ux0 - 0.3, uy0 - 50), (ux1 + 0.3, uy0 - 50), (ux1 + 0.3, uy1 + 50), (ux0 - 0.3, uy1 + 50)])
        # the crossing length comes from the model's area along this band (stripes can be worn)
        part = [q for q in polys(rb.intersection(band)) if q.intersects(u.buffer(0.5))]
        ext = unary_union(part + [u])
        _, y0, _, y1 = ext.bounds
        rects.append((a, (ux0 - 0.2, y0, ux1 + 0.2, y1)))
    # merge boxes of the same crosswalk (same direction, overlapping)
    out_boxes = []
    for a, b in rects:
        for k, (a2, b2) in enumerate(out_boxes):
            da = abs(a - a2); da = min(da, math.pi - da)
            if da < math.radians(15) and not (b[2] < b2[0] or b2[2] < b[0] or b[3] < b2[1] - 1.0 or b2[3] < b[1] - 1.0):
                out_boxes[k] = (a2, (min(b[0], b2[0]), min(b[1], b2[1]), max(b[2], b2[2]), max(b[3], b2[3])))
                break
        else:
            out_boxes.append((a, b))
    rects_ = [affinity.rotate(Polygon([(x0_, y0_), (x1_, y0_), (x1_, y1_), (x0_, y1_)]), a, origin=o, use_radians=True)
              for a, (x0_, y0_, x1_, y1_) in out_boxes]
    rects_.sort(key=lambda r: -r.area)
    out = []
    for r in rects_:
        if r.area < 3:
            continue
        if road is not None and not road.is_empty and r.intersection(road.buffer(0.5)).area < 0.5 * r.area:
            continue                                      # mostly off the road
        if any(r.intersection(k).area > 0.25 * r.area for k in out):
            continue                                      # part of a bigger crosswalk
        out.append(r.difference(unary_union(out)) if out else r)
    return [p for r in out for p in polys(r) if p.area >= 3]


def smooth_crosswalks(g):
    """Keep the detected crosswalk shape; remove only the zigzag the stripes leave on its edge:
    close the gaps between stripes, straighten the edges, soften the corners slightly."""
    g = g.buffer(0.6, join_style=2).buffer(-0.6, join_style=2)          # fill between stripes
    g = g.buffer(-0.4, join_style=2).buffer(0.4, join_style=2)          # cut small spikes
    g = g.simplify(0.5, preserve_topology=True)                         # straight edges
    out = []
    for p in polys(g):
        if p.area < 3:
            continue
        q = Polygon(chaikin_ring(p.exterior.coords, 1)).buffer(0)       # one light corner pass
        out += polys(q)
    return unary_union(out) if out else EMPTY


# ------------------------------------------------------------------ main
def build(img, t2n: dict, samroad_lines: list, legacy_road: np.ndarray, building: np.ndarray,
          sam3_road=None, sam3_parking=None, legacy_parking: np.ndarray | None = None,
          sidewalks=True, crosswalks=True, progress=None) -> tuple[list[dict], dict]:
    """t2n: native-resolution bool masks {"road", "sidewalk", "crosswalk"}; samroad_lines: SAM-Road
    centerlines (LineStrings); legacy_road / building / legacy_parking: native bool masks.
    Returns (features, stats). Features: label, geometry, score, source, attrs."""
    progress = progress or (lambda p, m: None)
    grid = Grid(img, 0.3)
    rgb = read_rgb(img, grid)
    green = ndimage.binary_opening(2 * rgb[1] - rgb[0] - rgb[2] > 15, iterations=1)

    progress(0.05, "Roads: combining models")
    t_road = grid.vec(grid.down(t2n["road"]))
    t_side = grid.vec(grid.down(t2n["sidewalk"]))
    t_cross = grid.vec(grid.down(t2n["crosswalk"]))
    bld = grid.vec(grid.down(building), 10.0)
    base = unary_union([t_road, t_cross])
    cl_old = unary_union(samroad_lines) if samroad_lines else EMPTY
    s_road = unary_union(sam3_road) if sam3_road else EMPTY
    s_park = unary_union(sam3_parking) if sam3_parking else EMPTY
    width_field = ndimage.distance_transform_edt(grid.down(legacy_road)) * grid.px
    fused, added = fuse(base, t_side, s_road, s_park, bld, cl_old, width_field, grid, green)

    progress(0.35, "Roads: cleaning the network")
    park_ref = s_park if not s_park.is_empty else (grid.vec(grid.down(legacy_parking), 20.0)
                                                  if legacy_parking is not None else EMPTY)
    recs, surface, nstats = network(fused, park_ref, bld, t_cross, added, grid)
    cl_union = unary_union([r["geometry"] for r in recs]) if recs else EMPTY

    progress(0.6, "Roads: curbs, sidewalks and islands")
    road, side = curb(surface, t_side, cl_union)
    fills = complete_sidewalks(road, side, recs, unary_union([bld, park_ref, t_cross])) if sidewalks else []
    side = unary_union([side] + fills)
    paved, other = islands(road, t_side, grid, green)
    road = road.difference(paved).difference(other)
    side = unary_union([side, paved])

    progress(0.8, "Roads: smoothing outlines")
    road = smooth_area(road)
    side = smooth_area(side, close=0.5, open_=0.5, simplify=0.5).difference(road)
    parking = park_ref.difference(unary_union([road, side]).buffer(0.5)) if not park_ref.is_empty else EMPTY
    parking = smooth_area(parking.buffer(-1.5).buffer(1.5), min_area=60.0)
    cross = t_cross.intersection(road.buffer(0.5)) if crosswalks else EMPTY
    # crosswalks keep their detected shape; only the zigzag edge (the stripes) is smoothed away
    cross = smooth_crosswalks(cross) if not cross.is_empty else EMPTY

    feats = []
    for p in polys(road):
        feats.append({"label": "road", "geometry": p, "score": 0.9, "source": "auto"})
    if sidewalks:
        fill_u = unary_union(fills) if fills else EMPTY
        for p in polys(side):
            if p.area < 2:
                continue
            src = "inferred" if not fill_u.is_empty and p.intersection(fill_u).area > 0.5 * p.area else "auto"
            feats.append({"label": "sidewalk", "geometry": p, "score": 0.8, "source": src})
    for p in polys(cross):
        if p.area >= 3:
            feats.append({"label": "crosswalk", "geometry": p, "score": 0.8, "source": "auto"})
    for p in polys(parking):
        feats.append({"label": "parking", "geometry": p, "score": 0.7, "source": "auto"})
    for r in recs:
        feats.append({"label": "road_centerline", "geometry": r["geometry"],
                      "score": 0.9 if r["source"] == "detected" else 0.6,
                      "source": r["source"], "attrs": r["attrs"]})
    stats = {"tile2net_ha": round(base.area / 1e4, 2), "road_ha": round(road.area / 1e4, 2),
             "added_pieces": len(added), "sidewalk_fills": len(fills), **nstats}
    progress(1.0, "Roads done")
    return feats, stats
