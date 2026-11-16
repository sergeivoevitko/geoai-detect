"""Rectilinear building regularization prototype.
usage: ortho.py buildings.shp out.gpkg"""
import sys, math, numpy as np, geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage
from shapely import affinity
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
STEP = 0.25        # m grid in the building's own frame
MIN_STEP = 2.0     # m: wall jogs smaller than this are removed
def main_angle(p):
    r = p.minimum_rotated_rectangle; c = list(r.exterior.coords)
    e = max(((c[i], c[i+1]) for i in range(4)), key=lambda s: math.dist(*s))
    return math.degrees(math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]))
def chaikin(p, n=2):
    pts = np.asarray(p.exterior.coords[:-1])
    for _ in range(n):
        q = 0.75*pts + 0.25*np.roll(pts,-1,0); r = 0.25*pts + 0.75*np.roll(pts,-1,0)
        pts = np.empty((len(q)*2,2)); pts[0::2]=q; pts[1::2]=r
    return Polygon(pts).buffer(0)
def iou(a, b): return a.intersection(b).area / max(a.union(b).area, 1e-9)
def regularize(p):
    p = Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area >= 25]).buffer(0)
    if p.geom_type == 'MultiPolygon': p = max(p.geoms, key=lambda x: x.area)
    mrr = p.minimum_rotated_rectangle
    if p.area / mrr.area > 0.85:
        return mrr, "rect"
    ang = main_angle(p); c = p.centroid
    q = affinity.rotate(p, -ang, origin=c)
    x0, y0, x1, y1 = q.bounds; pad = 2
    W = int((x1-x0)/STEP)+2*pad; H = int((y1-y0)/STEP)+2*pad
    tr = from_origin(x0-pad*STEP, y1+pad*STEP, STEP, STEP)
    m = features.rasterize([(q,1)], out_shape=(H,W), transform=tr).astype(bool)
    k = max(1, int(round(MIN_STEP/STEP/2)))
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=k), iterations=k)
    shp = [shape(g) for g, v in features.shapes(m.astype(np.uint8), mask=m, transform=tr)]
    if not shp: return p.simplify(0.6), "straight"
    r = max(shp, key=lambda g: g.area)
    r = Polygon(r.exterior, [h for h in r.interiors if Polygon(h).area >= 25]).simplify(0.01)
    r = affinity.rotate(r, ang, origin=c)
    if iou(r, p) >= 0.85:
        return r, "ortho"
    round_ = 4 * math.pi * p.area / p.length ** 2 > 0.75          # round towers, domes, stadiums
    if round_:
        return chaikin(p.simplify(0.5)), "curved"
    q = p.buffer(1.0, join_style=2).buffer(-1.0, join_style=2)       # close small notches
    q = q.buffer(-0.8, join_style=2).buffer(0.8, join_style=2)       # cut small spikes
    q = q if not q.is_empty else p
    q = max(q.geoms, key=lambda x: x.area) if q.geom_type == "MultiPolygon" else q
    return q.simplify(0.6, preserve_topology=True), "straight"
g = gpd.read_file(sys.argv[1]); out, kinds = [], {}
for p in g.geometry:
    if p is None or p.is_empty: continue
    p = max(p.geoms, key=lambda x: x.area) if p.geom_type == "MultiPolygon" else p
    r, k = regularize(p); out.append(r); kinds[k] = kinds.get(k, 0) + 1
gpd.GeoDataFrame(geometry=out, crs=g.crs).to_file(sys.argv[2], driver="GPKG")
print(len(out), "buildings:", kinds)
