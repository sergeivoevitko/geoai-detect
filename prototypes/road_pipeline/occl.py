"""Fill road under tree canopy: SAM-Road centerline pieces outside the visible road, mostly over
green pixels, connected to the road network, widened to the neighbouring road width.
usage: occl.py image.tif road.(gpkg|shp) centerline.(gpkg|shp) out_prefix"""
import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, substring
from scipy import ndimage
img, roadp, clp, out = sys.argv[1:5]
d = rasterio.open(img); crs = d.crs
road = unary_union(list(gpd.read_file(roadp).to_crs(crs).geometry))
cl = gpd.read_file(clp).to_crs(crs)
# greenness (excess green) on a ~0.6 m grid
f = max(1, int(round(0.6 / abs(d.res[0])))); H, W = d.height // f, d.width // f
a = d.read([1, 2, 3], out_shape=(3, H, W)).astype(np.float32)
exg = 2 * a[1] - a[0] - a[2]
green = ndimage.binary_opening(exg > 18, iterations=1)
tr = d.transform * rasterio.Affine.scale(d.width / W, d.height / H)
inv = ~tr
def frac_green(ln, step=1.0):
    pts = [ln.interpolate(t) for t in np.arange(0, ln.length + 1e-6, step)]
    v = []
    for p in pts:
        c, r = inv * (p.x, p.y)
        r, c = int(r), int(c)
        if 0 <= r < H and 0 <= c < W:
            v.append(green[max(0, r-1):r+2, max(0, c-1):c+2].mean() > 0.4)
    return float(np.mean(v)) if v else 0.0
# road half-width from the visible road around a point
rm = features.rasterize([(road, 1)], out_shape=(H, W), transform=tr).astype(bool)
dist = ndimage.distance_transform_edt(rm) * abs(tr.a)
def halfwidth_near(p, r=12.0):
    c, rr = inv * (p.x, p.y); c, rr = int(c), int(rr); k = int(r / abs(tr.a))
    win = dist[max(0, rr-k):rr+k, max(0, c-k):c+k]
    return float(np.clip(np.percentile(win[win > 0], 90), 2.5, 8.0)) if (win > 0).any() else 3.0
outside = cl.geometry.difference(road.buffer(1.0)).explode(index_parts=False)
fills, kept = [], []
near = road.buffer(3.0)
for ln in outside:
    if ln.is_empty or ln.length < 6:
        continue
    ends = [Point(ln.coords[0]), Point(ln.coords[-1])]
    touching = sum(near.contains(e) for e in ends)
    g = frac_green(ln)
    if "-v" in sys.argv: print(f"  len {ln.length:5.0f} m  ends on road {touching}  green {g:.2f}")
    # under canopy and bridging two road pieces, or a short canopy stub from one road
    if g >= 0.6 and (touching == 2 or (touching == 1 and ln.length <= 40)):
        hw = min(halfwidth_near(e) for e in ends if near.contains(e))
        fills.append(ln.buffer(hw, cap_style=2)); kept.append((ln, g))
fill = unary_union(fills).difference(road) if fills else None
gpd.GeoDataFrame(geometry=[k[0] for k in kept], crs=crs).to_file(f"{out}_occl_lines.gpkg", driver="GPKG")
gpd.GeoDataFrame(geometry=[fill] if fill is not None else [], crs=crs).explode(index_parts=False).to_file(f"{out}_occl_fill.gpkg", driver="GPKG")
print(f"centerline pieces outside visible road: {sum(1 for x in outside if not x.is_empty and x.length>=6)} | "
      f"filled under trees: {len(kept)} ({sum(k[0].length for k in kept):.0f} m, {fill.area if fill is not None else 0:.0f} m2)")
