"""Road polygon smoothing prototypes.
A: smooth outline (open/close, simplify, Chaikin)   B: rebuild from smoothed centerlines + width, fillet corners
usage: smooth.py image.tif road.gpkg centerline.gpkg out_prefix"""
import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
img, roadp, clp, out = sys.argv[1:5]
d = rasterio.open(img); crs = d.crs
road = unary_union(list(gpd.read_file(roadp).to_crs(crs).geometry))
cl = gpd.read_file(clp).to_crs(crs)

def chaikin_ring(coords, n=2):
    pts = np.asarray(coords[:-1])
    for _ in range(n):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, 0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, 0)
        pts = np.empty((len(q) * 2, 2)); pts[0::2] = q; pts[1::2] = r
    return np.vstack([pts, pts[:1]])
def chaikin_line(coords, n=2):
    pts = np.asarray(coords)
    for _ in range(n):
        if len(pts) < 3: break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]; r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((len(q) * 2, 2)); mid[0::2] = q; mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts
def polys(g): return list(g.geoms) if hasattr(g, "geoms") else [g]
def smooth_poly(p, n=2):
    ext = chaikin_ring(p.exterior.coords, n)
    holes = [chaikin_ring(h.coords, n) for h in p.interiors if Polygon(h).area > 20]
    return Polygon(ext, holes).buffer(0)

# ---------- A: outline smoothing
a = road.buffer(1.0, join_style=1).buffer(-2.0, join_style=1).buffer(1.0, join_style=1)  # close gaps, remove spikes
a = a.simplify(0.75, preserve_topology=True)                                              # straight where straight
A = unary_union([smooth_poly(p) for p in polys(a) if p.area > 20])                         # round the corners

# ---------- B: rebuild from centerlines
f = max(1, int(round(0.5 / abs(d.res[0])))); H, W = d.height // f, d.width // f
tr = d.transform * rasterio.Affine.scale(f); inv = ~tr
m = features.rasterize([(road, 1)], out_shape=(H, W), transform=tr).astype(bool)
dist = ndimage.distance_transform_edt(m) * abs(tr.a)
parts = []
for ln in cl.geometry:
    if ln.length < 5: continue
    s = LineString(chaikin_line(ln.simplify(1.0).coords, 3))
    # measure half-width along the line, per ~25 m piece, robust median
    n = max(1, int(s.length // 25)); 
    for i in range(n):
        seg = __import__("shapely.ops", fromlist=["substring"]).substring(s, i * s.length / n, (i + 1) * s.length / n)
        v = []
        for t in np.linspace(0, seg.length, max(3, int(seg.length // 2))):
            p = seg.interpolate(t); c, r = inv * (p.x, p.y)
            if 0 <= int(r) < H and 0 <= int(c) < W and dist[int(r), int(c)] > 0: v.append(dist[int(r), int(c)])
        hw = float(np.clip(np.median(v), 2.0, 15.0)) if v else 3.0
        parts.append(seg.buffer(hw, cap_style=1, join_style=1, resolution=8))
B = unary_union(parts)
B = B.buffer(3.0, join_style=1, resolution=8).buffer(-3.0, join_style=1, resolution=8)   # curb-like fillets at junctions
# add back big areas the rebuild misses (plazas, wide junctions), smoothed like A
extra = A.difference(B.buffer(0.5)).buffer(-1.5).buffer(1.5)
B = unary_union([B] + [p for p in polys(extra) if p.area > 120])
for name, g in (("A", A), ("B", B)):
    gg = gpd.GeoDataFrame(geometry=polys(g), crs=crs); gg.to_file(f"{out}_{name}.gpkg", driver="GPKG")
    nv = sum(len(p.exterior.coords) for p in polys(g))
    print(f"{name}: {len(gg)} polygons, {g.area/1e4:.2f} ha, {nv} vertices")
nv0 = sum(len(p.exterior.coords) for p in polys(road)); print(f"raw: {road.area/1e4:.2f} ha, {nv0} vertices")
