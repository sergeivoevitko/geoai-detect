"""Road network cleanup prototype.

A1 one node per junction   A2 one line per segment, prune stubs   A3 straight/curved/circle geometry
A4 close gaps between dead ends and nearby roads   A5 attributes
B1 road surface along inferred links   B2 even road edges from centerline + width
B3 drop road pieces not connected to the network

usage: net.py image.tif road.gpkg centerline.gpkg out_prefix [parking.gpkg] [building.shp]
"""
import math
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, nearest_points, polygonize, unary_union

img, roadp, clp, out = sys.argv[1:5]
parkp = sys.argv[5] if len(sys.argv) > 5 else None
bldp = sys.argv[6] if len(sys.argv) > 6 else None
d = rasterio.open(img); crs = d.crs


def rd(p):
    try:
        return unary_union(list(gpd.read_file(p).to_crs(crs).geometry))
    except Exception:
        return Polygon()


road = rd(roadp)
lines0 = [g for g in gpd.read_file(clp).to_crs(crs).geometry if g is not None and g.length > 0]
park = rd(parkp) if parkp else Polygon()
bld = rd(bldp) if bldp else Polygon()

# road half-width field (0.5 m grid)
f = max(1, int(round(0.5 / abs(d.res[0])))); H, W = d.height // f, d.width // f
tr = d.transform * rasterio.Affine.scale(f); inv = ~tr; px = abs(tr.a)


def halfwidth_field(poly):
    m = features.rasterize([(poly, 1)], out_shape=(H, W), transform=tr).astype(bool)
    return ndimage.distance_transform_edt(m) * px


dist = halfwidth_field(road)


def sample(field, ln, step=2.0):
    v = []
    for t in np.arange(0, ln.length + 1e-6, step):
        p = ln.interpolate(t); c, r = inv * (p.x, p.y)
        if 0 <= int(r) < H and 0 <= int(c) < W:
            v.append(field[int(r), int(c)])
    return np.array(v)


def ends(ln):
    return Point(ln.coords[0]), Point(ln.coords[-1])


def key(p, q=0.01):
    return (round(p.x / q) * q, round(p.y / q) * q)


def build_graph(lines):
    g = nx.MultiGraph()
    noded = unary_union(lines)
    for ln in getattr(noded, "geoms", [noded]):
        if ln.length < 0.5:
            continue
        a, b = ends(ln)
        g.add_edge(key(a), key(b), geom=ln)
    return g


def lines_of(g):
    return [dat["geom"] for _, _, dat in g.edges(data=True)]


def stats(lines, label):
    g = build_graph(lines)
    L, B, R, T = d.bounds
    dead = sum(1 for n in g.nodes if g.degree(n) == 1 and min(n[0] - L, R - n[0], n[1] - B, T - n[1]) > 15)
    comp = nx.number_connected_components(g)
    print(f"  {label:9s}: {len(lines):4d} lines, {sum(l.length for l in lines)/1000:6.2f} km, "
          f"{comp:3d} separate pieces, {dead:3d} dead ends")


stats(lines0, "before")

# ---------------------------------------------------------------- A2a prune short stubs
def prune(lines, min_len=15.0, rounds=3):
    for _ in range(rounds):
        g = build_graph(lines)
        drop = set()
        for u, v, k, dat in g.edges(keys=True, data=True):
            ln = dat["geom"]
            du, dv = g.degree(u), g.degree(v)
            if ln.length < min_len and (du == 1) != (dv == 1) and max(du, dv) >= 3:
                drop.add(id(ln))                       # stub off a junction
            elif ln.length < 6 and du == 1 and dv == 1:
                drop.add(id(ln))                       # tiny isolated fragment
        if not drop:
            break
        lines = [l for l in lines_of(g) if id(l) not in drop]
    return lines


lines = prune(lines0)

# ---------------------------------------------------------------- A4 close gaps
def end_dir(ln, at_start, look=10.0):
    if at_start:
        a = ln.interpolate(0); b = ln.interpolate(min(look, ln.length))
    else:
        a = ln.interpolate(ln.length); b = ln.interpolate(max(0, ln.length - look))
    v = np.array([a.x - b.x, a.y - b.y]); n = np.linalg.norm(v)
    return (a, v / n) if n > 0 else (a, None)


links = []
g = build_graph(lines)
net_u = unary_union(lines)
for node in [n for n in g.nodes if g.degree(n) == 1]:
    (u, v, dat), = [(u, v, dat) for u, v, dat in g.edges(node, data=True)]
    ln = dat["geom"]
    at_start = key(Point(ln.coords[0])) == node
    p, dvec = end_dir(ln, at_start)
    if dvec is None:
        continue
    # look ahead in a +-35 degree cone up to 25 m for any other line
    best = None
    others = net_u.difference(ln.buffer(0.5))
    for dist_ahead in np.arange(2, 26, 1.0):
        probe = Point(p.x + dvec[0] * dist_ahead, p.y + dvec[1] * dist_ahead).buffer(max(1.5, dist_ahead * math.tan(math.radians(35))))
        hit = probe.intersection(others)
        if not hit.is_empty:
            q = nearest_points(Point(p.x + dvec[0] * dist_ahead, p.y + dvec[1] * dist_ahead), hit)[1]
            best = LineString([p, q]); break
    if best is None or best.length > 25:
        continue
    if not park.is_empty and best.intersection(park).length > 0.5 * best.length:
        continue
    if not bld.is_empty and best.intersection(bld).length > 0.2 * best.length:
        continue
    links.append(best)
lines = lines + links

# ---------------------------------------------------------------- A1 one node per junction
g = build_graph(lines)
junctions = [n for n in g.nodes if g.degree(n) >= 3]
clusters = []
if junctions:
    pts = np.array(junctions)
    jg = nx.Graph(); jg.add_nodes_from(range(len(pts)))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(*(pts[i] - pts[j])) < 12.0:
                jg.add_edge(i, j)
    for comp in nx.connected_components(jg):
        if len(comp) > 1:
            clusters.append([junctions[i] for i in comp])
moves = {}
for cl in clusters:
    c = tuple(np.mean(np.array(cl), axis=0))
    for n in cl:
        moves[n] = c
new_lines = []
for ln in lines_of(g):
    a, b = key(Point(ln.coords[0])), key(Point(ln.coords[-1]))
    if a in moves and b in moves and moves[a] == moves[b] and ln.length < 30:
        continue                                        # internal piece of a collapsed junction
    cs = list(ln.coords)
    if a in moves: cs[0] = moves[a]
    if b in moves: cs[-1] = moves[b]
    if len(cs) >= 2 and LineString(cs).length > 0.5:
        new_lines.append(LineString(cs))
lines = prune(new_lines)

# ---------------------------------------------------------------- A2b merge chains between junctions
merged = linemerge(unary_union(lines))
lines = [l for l in getattr(merged, "geoms", [merged]) if l.length > 1]

# ---------------------------------------------------------------- A3 geometry: circles, straight, smooth
def chaikin(cs, n=2):
    pts = np.asarray(cs)
    for _ in range(n):
        if len(pts) < 3: break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]; r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((len(q) * 2, 2)); mid[0::2] = q; mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def fit_circle(cs):
    x, y = np.asarray(cs)[:, 0], np.asarray(cs)[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]; b = x ** 2 + y ** 2
    (cx, cy, c0), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(c0 + cx ** 2 + cy ** 2, 0))
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, float(np.mean(res))


arcs = []
for ln in lines:
    if ln.length < 20 or len(ln.coords) < 6:
        continue
    pts = [ln.interpolate(t).coords[0] for t in np.linspace(0, ln.length, 30)]
    cx, cy, r, res = fit_circle(pts)
    if 8 < r < 90 and res < 0.06 * r + 0.5:
        ang = np.unwrap([math.atan2(py - cy, px_ - cx) for px_, py in pts])
        if abs(ang[-1] - ang[0]) > math.radians(70):              # a real arc, not a gentle bend
            arcs.append((cx, cy, r, ang[0], ang[-1]))
groups = []                                                         # arcs of the same roundabout
for cx, cy, r, a0, a1 in arcs:
    for gr in groups:
        c, rr = gr["c"], gr["r"]
        if math.hypot(c[0] - cx, c[1] - cy) < 0.25 * rr + 3 and abs(rr - r) < 0.25 * rr + 3:
            n = len(gr["arcs"]); gr["c"] = ((c[0] * n + cx) / (n + 1), (c[1] * n + cy) / (n + 1))
            gr["r"] = (rr * n + r) / (n + 1); gr["arcs"].append((a0, a1)); break
    else:
        groups.append({"c": (cx, cy), "r": r, "arcs": [(a0, a1)]})
circles = []
for gr in groups:
    cover = np.zeros(360, bool)                                     # angular coverage of the arcs
    for a0, a1 in gr["arcs"]:
        lo, hi = sorted((math.degrees(a0), math.degrees(a1)))
        for a in range(int(lo), int(hi) + 1):
            cover[a % 360] = True
    if cover.sum() >= 270 and 8 < gr["r"] < 60:
        circles.append((Point(*gr["c"]), gr["r"]))


def on_circle(ln):
    for c, r in circles:
        dd = [abs(math.hypot(x - c.x, y - c.y) - r) for x, y in ln.coords]
        if np.mean(dd) < 0.12 * r + 1.0:
            return c, r
    return None


geo = []
for ln in lines:
    hit = on_circle(ln)
    if hit:
        c, r = hit
        a0 = math.atan2(ln.coords[0][1] - c.y, ln.coords[0][0] - c.x)
        a1 = math.atan2(ln.coords[-1][1] - c.y, ln.coords[-1][0] - c.x)
        mid = ln.interpolate(0.5, normalized=True); am = math.atan2(mid.y - c.y, mid.x - c.x)
        # choose the arc direction that passes the line's midpoint
        span = (a1 - a0) % (2 * math.pi)
        if not (0 <= (am - a0) % (2 * math.pi) <= span):
            span = span - 2 * math.pi
        if ln.is_ring or ln.length > 1.5 * math.pi * r:
            span = 2 * math.pi
        if span == 2 * math.pi:
            n = max(16, int(abs(span) * r / 2))
            geo.append((LineString([(c.x + r * math.cos(a0 + span * t / n), c.y + r * math.sin(a0 + span * t / n))
                                    for t in range(n + 1)]), "roundabout"))
            continue
        n = max(4, int(abs(span) * r / 2))
        inner = [(c.x + r * math.cos(a0 + span * t / n), c.y + r * math.sin(a0 + span * t / n)) for t in range(1, n)]
        geo.append((LineString([ln.coords[0]] + inner + [ln.coords[-1]]), "roundabout"))   # ends stay on their nodes
        continue
    chord = LineString([ln.coords[0], ln.coords[-1]])
    dev = max(chord.distance(Point(p)) for p in ln.coords) if chord.length > 0 else 99
    if ln.length > 15 and dev < max(1.0, 0.02 * ln.length):
        geo.append((chord, "straight"))
    else:
        geo.append((LineString(chaikin(ln.simplify(0.75).coords, 2)), "curve"))

# ---------------------------------------------------------------- A5 attributes + B1/B2 surfaces
link_u = unary_union(links).buffer(1.0) if links else Polygon()
recs, corridors, inferred = [], [], []
for ln, shape_kind in geo:
    hw = sample(dist, ln)
    hw = hw[hw > 0]
    is_link = (not link_u.is_empty) and ln.within(link_u)
    if len(hw) >= 3:
        width = float(2 * np.median(hw)); cv = float(np.std(hw) / (np.mean(hw) + 1e-6))
    else:
        width, cv = 6.0, 1.0
    width = float(np.clip(width, 4.0, 30.0))
    lanes = max(1, int(round(width / 3.3)))
    rclass = "highway" if width >= 18 else ("main" if width >= 10 else "local")
    seen = float(np.mean(sample(dist, ln) > 0)) if ln.length > 0 else 0.0
    source = "inferred" if (is_link or seen < 0.5) else "detected"
    recs.append({"geometry": ln, "width_m": round(width, 1), "lanes": lanes, "class": rclass,
                 "shape": shape_kind, "source": source, "seen": round(seen, 2)})
    if source == "inferred":
        inferred.append(ln.buffer(width / 2, cap_style=2, join_style=1))
    if cv < 0.25 and ln.length > 12:
        corridors.append(ln.buffer(width / 2, cap_style=1, join_style=1, resolution=8))

# B2: even edges where width is regular, detected shape elsewhere (bus bays, turn lanes, plazas)
corr = unary_union(corridors) if corridors else Polygon()
rest = road.difference(corr.buffer(1.5)).buffer(-1.0).buffer(1.0)
rest = unary_union([p for p in getattr(rest, "geoms", [rest]) if p.area > 40])
surface = unary_union([corr, rest] + inferred)
surface = surface.buffer(3.0, join_style=1).buffer(-3.0, join_style=1)        # rounded curb returns
if not bld.is_empty:
    surface = surface.difference(bld)

# B3: keep only road pieces connected to the network
net_all = unary_union([r["geometry"] for r in recs])
pieces = [p for p in getattr(surface, "geoms", [surface]) if p.intersects(net_all) or p.area > 2000]
dropped = sum(p.area for p in getattr(surface, "geoms", [surface])) - sum(p.area for p in pieces)
surface = unary_union(pieces)
# final outline smoothing (method A)
surface = surface.simplify(0.5, preserve_topology=True)

final_lines = [r["geometry"] for r in recs]
stats(final_lines, "after")
print(f"  links added {len(links)}, junction clusters merged {len(clusters)}, circles fitted {len(circles)}, "
      f"road pieces dropped {dropped:.0f} m2, surface {road.area/1e4:.2f} -> {surface.area/1e4:.2f} ha")
gpd.GeoDataFrame(recs, crs=crs).to_file(f"{out}_centerline.gpkg", driver="GPKG")
gpd.GeoDataFrame(geometry=list(getattr(surface, "geoms", [surface])), crs=crs).to_file(f"{out}_road.gpkg", driver="GPKG")
gpd.GeoDataFrame(geometry=links, crs=crs).to_file(f"{out}_links.gpkg", driver="GPKG")
