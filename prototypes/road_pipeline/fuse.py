"""Fuse Tile2Net + SAM 3 roads, trace centerlines from the result, clean SAM 3 parking.
usage: fuse.py image.tif t2n_prefix sam3_prefix building.shp out_prefix"""
import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from shapely.ops import unary_union
sys.path.insert(0, "/home/user/geoai-detect")
from geoai.centerline import surface_centerlines
img, t2n, s3, bldp, out = sys.argv[1:6]
legacyp = sys.argv[6] if len(sys.argv) > 6 else None     # previous model's road polygons (width source)
d = rasterio.open(img); crs = d.crs
def rd(p):
    try: return unary_union(list(gpd.read_file(p).to_crs(crs).geometry))
    except Exception: from shapely.geometry import GeometryCollection; return GeometryCollection()
t_road, t_side, t_cross = rd(f"{t2n}_road.shp"), rd(f"{t2n}_sidewalk.shp"), rd(f"{t2n}_crosswalk.shp")
s_road, s_park, bld = rd(f"{s3}_road.shp"), rd(f"{s3}_parking_lot.shp"), rd(bldp)
cl_old = rd(f"{t2n}_road_centerline.shp")
base = unary_union([t_road, t_cross])
cand = s_road.difference(base.buffer(0.3)).difference(t_side).difference(bld.buffer(0.5)).difference(s_park.buffer(1.0))
cand = cand.buffer(-1.0).buffer(1.0)
pieces = [p for p in getattr(cand, "geoms", [cand]) if p.area >= 25]
def contacts(p):
    """number of separate places where the piece touches the Tile2Net road"""
    c = p.exterior.buffer(1.0).intersection(base)
    c = c.buffer(0.5)
    return len([g for g in getattr(c, "geoms", [c]) if g.area > 2])
# gap filling only: a SAM 3 piece must connect road on two separate sides (a gap between road
# pieces, e.g. a bridge shadow), not just sit next to the road (plazas, yards)
def along_centerline(p):
    """a SAM-Road centerline runs through the piece lengthwise (a road, not a railway or a yard)"""
    ln = p.intersection(cl_old)
    L = max(p.minimum_rotated_rectangle.exterior.length / 2 - min(
        LineString(p.minimum_rotated_rectangle.exterior.coords[i:i + 2]).length for i in range(2)), 1.0)
    return ln.length >= min(8.0, 0.5 * L) if not ln.is_empty else False
from shapely.geometry import LineString
keep = [p for p in pieces if contacts(p) >= 2 and along_centerline(p)]
added = unary_union(keep)
fused = unary_union([base, added]).buffer(0.5).buffer(-0.5)
# ---- centerline fill: SAM-Road segments that run outside the visible road (residential streets the
# road model missed at this resolution, or road under trees), connected to the network, not in a
# parking lot, not on buildings. Width from the previous model's road polygons (or the nearby road).
from shapely.geometry import Point
fr = max(1, int(round(0.5 / abs(d.res[0])))); Hr, Wr = d.height // fr, d.width // fr
trr = d.transform * rasterio.Affine.scale(fr); invr = ~trr
from scipy import ndimage
wsrc = rd(legacyp) if legacyp else fused
wm = features.rasterize([(wsrc, 1)], out_shape=(Hr, Wr), transform=trr).astype(bool) if not wsrc.is_empty else np.zeros((Hr, Wr), bool)
wdist = ndimage.distance_transform_edt(wm) * abs(trr.a)
def halfwidth(ln):
    v = []
    for t in np.linspace(0, ln.length, max(3, int(ln.length // 2))):
        p = ln.interpolate(t); c, r = invr * (p.x, p.y)
        if 0 <= int(r) < Hr and 0 <= int(c) < Wr and wdist[int(r), int(c)] > 0: v.append(wdist[int(r), int(c)])
    return float(np.clip(np.median(v), 2.5, 7.0)) if len(v) >= 3 else 3.0
near = fused.buffer(3.0)
seg_fill, n_seg = [], 0
for ln in getattr(cl_old.difference(fused.buffer(1.0)), "geoms", []):
    if ln.length < 10: continue
    ends = [Point(ln.coords[0]), Point(ln.coords[-1])]
    n_conn = sum(near.contains(e) for e in ends)
    if n_conn == 0 or (n_conn == 1 and ln.length < 30): continue             # connected; short stubs must bridge
    if ln.intersection(s_park.buffer(1.0)).length > 0.3 * ln.length: continue   # parking aisles
    if ln.intersection(bld).length > 0.2 * ln.length: continue
    hw = halfwidth(ln)
    strip = ln.buffer(hw, cap_style=2, join_style=1)
    if strip.intersection(t_side).area > 0.3 * strip.area: continue          # plaza / pedestrian area
    seg_fill.append(strip); n_seg += 1
cl_added = unary_union(seg_fill).difference(bld.buffer(0.3)).difference(s_park) if seg_fill else None
if cl_added is not None and not cl_added.is_empty:
    fused = unary_union([fused, cl_added]).buffer(0.5).buffer(-0.5)
    added = unary_union([added, cl_added])
print(f"centerline fill: {n_seg} segments, {(cl_added.area if cl_added is not None else 0)/1e4:.2f} ha")
# centerlines traced from the fused road surface (~0.6 m grid)
f = max(1, int(round(0.6 / abs(d.res[0])))); H, W = d.height // f, d.width // f
tr = d.transform * rasterio.Affine.scale(f)
m = features.rasterize([(fused, 1)], out_shape=(H, W), transform=tr).astype(bool)
lines = surface_centerlines(m, tr, abs(tr.a), min_spur_m=10.0)
# parking: SAM 3 parking lots minus road and sidewalk
park = s_park.difference(unary_union([fused, t_side]).buffer(0.5))
park = [p for p in getattr(park.buffer(-1.5).buffer(1.5), "geoms", [park]) if p.area >= 60]
G = lambda geoms: gpd.GeoDataFrame(geometry=list(geoms), crs=crs)
G([fused]).explode(index_parts=False).to_file(f"{out}_road.gpkg", driver="GPKG")
G([added]).explode(index_parts=False).to_file(f"{out}_added.gpkg", driver="GPKG")
G(lines).to_file(f"{out}_centerline.gpkg", driver="GPKG")
G(park).to_file(f"{out}_parking.gpkg", driver="GPKG")
L = lambda g: sum(x.length for x in g)
print(f"tile2net road {base.area/1e4:.2f} ha | SAM 3 added {len(keep)} pieces {added.area/1e4:.2f} ha | fused {fused.area/1e4:.2f} ha | "
      f"centerlines {len(lines)} ({L(lines)/1000:.2f} km, old {cl_old.length/1000:.2f} km) | parking {len(park)} ({sum(p.area for p in park)/1e4:.2f} ha)")
