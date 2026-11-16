"""Traffic island prototype.
Candidates: (a) holes already inside the road, (b) non-asphalt patches painted over as road
(green, or concrete clearly brighter than the surrounding asphalt) that are enclosed by road.
Paved islands -> sidewalk layer; vegetated -> removed from road (left out of both layers).
usage: islands.py image.tif road.gpkg sidewalk.gpkg out_prefix"""
import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from scipy import ndimage
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

img, roadp, swp, out = sys.argv[1:5]
t2n_swp = sys.argv[6] if len(sys.argv) > 6 else swp
clp = sys.argv[5] if len(sys.argv) > 5 else None
d = rasterio.open(img)
r = gpd.read_file(roadp); crs = r.crs
road = unary_union(list(r.geometry)); sw = unary_union(list(gpd.read_file(swp).to_crs(crs).geometry))

f = max(1, int(round(0.3 / abs(d.res[0])))); H, W = d.height // f, d.width // f
tr = d.transform * rasterio.Affine.scale(d.width / W, d.height / H); px = abs(tr.a)
a = d.read([1, 2, 3], out_shape=(3, H, W)).astype(np.float32)
exg = 2 * a[1] - a[0] - a[2]
bright = a.mean(0)
green = ndimage.binary_opening(exg > 15, iterations=1)
t2n_sw = unary_union(list(gpd.read_file(t2n_swp).to_crs(crs).geometry))

road_m = features.rasterize([(road, 1)], out_shape=(H, W), transform=tr).astype(bool)
filled = ndimage.binary_fill_holes(road_m)

# asphalt reference: median brightness of the road in a ~40 m neighbourhood
k = max(3, int(40 / px))
asph = ndimage.median_filter(np.where(road_m, bright, np.nan).astype(np.float32), size=1) if False else None
rb = np.where(road_m & ~green, bright, 0).astype(np.float32); rc = (road_m & ~green).astype(np.float32)
loc_mean = ndimage.uniform_filter(rb, k) / np.maximum(ndimage.uniform_filter(rc, k), 1e-3)
concrete = (bright > loc_mean + 28) & ~green                     # clearly lighter than nearby asphalt

holes = filled & ~road_m                                          # (a) existing holes
# junction nodes: painted-over islands are only accepted near a junction (cars are bright too)
from shapely.geometry import Point
from collections import Counter
jn = Polygon()
if clp:
    cnt = Counter()
    for ln in gpd.read_file(clp).to_crs(crs).geometry:
        for c in (ln.coords[0], ln.coords[-1]): cnt[(round(c[0], 1), round(c[1], 1))] += 1
    jn = unary_union([Point(c).buffer(25) for c, k in cnt.items() if k >= 3])
jn_m = features.rasterize([(jn, 1)], out_shape=(H, W), transform=tr).astype(bool) if not jn.is_empty else np.zeros((H, W), bool)
painted = road_m & ndimage.binary_opening(green | concrete, iterations=2)   # (b) painted over
cand = holes | painted
lab, n = ndimage.label(cand)
paved, veg = [], []
inner = ndimage.binary_erosion(filled, iterations=2)
for i, sl in enumerate(ndimage.find_objects(lab), start=1):
    if sl is None:
        continue
    blob = lab[sl] == i
    area = blob.sum() * px * px
    if not (10 <= area <= 4000):
        continue
    if not inner[sl][blob].mean() > 0.9:                         # must sit inside the road, not on its edge
        continue
    ring = ndimage.binary_dilation(blob, iterations=max(1, int(1.5 / px))) & ~blob
    if road_m[sl][ring].mean() < 0.75 if ring.any() else True:   # mostly surrounded by road
        continue
    eroded = ndimage.binary_erosion(blob, iterations=max(1, int(1.0 / px)))
    if eroded.sum() == 0:                                         # thinner than ~2 m: a road marking or a curb line
        continue
    from_hole = holes[sl][blob].mean() > 0.5
    if not from_hole and (area < 20 or not jn_m[sl][blob].any()) and green[sl][blob].mean() <= 0.35:
        continue                                                  # bright blob away from junctions: a car
    g = green[sl][blob].mean()
    m = np.zeros((H, W), bool); m[sl] = blob
    geoms = [shape(s) for s, v in features.shapes(m.astype(np.uint8), mask=m, transform=tr)]
    poly = unary_union(geoms).buffer(0.3).buffer(-0.3).simplify(0.3)
    # paved only when Tile2Net itself saw sidewalk there; dry grass and bare soil are not green
    # but not walkable either, so everything else is left out of both layers
    sw_share = poly.intersection(t2n_sw).area / max(poly.area, 1e-6)
    (paved if (sw_share > 0.5 and g <= 0.35) else veg).append(poly)

P, V = unary_union(paved) if paved else Polygon(), unary_union(veg) if veg else Polygon()
road2 = road.difference(P).difference(V)
sw2 = unary_union([sw, P]).difference(road2)
parts = lambda gg: [x for x in getattr(gg, "geoms", [gg]) if x.geom_type == "Polygon" and x.area > 2]
gpd.GeoDataFrame(geometry=parts(road2), crs=crs).to_file(f"{out}_road.gpkg", driver="GPKG")
gpd.GeoDataFrame(geometry=parts(sw2), crs=crs).to_file(f"{out}_sidewalk.gpkg", driver="GPKG")
gpd.GeoDataFrame({"type": ["paved"] * len(paved) + ["not paved"] * len(veg)}, geometry=paved + veg, crs=crs) \
    .to_file(f"{out}_islands.gpkg", driver="GPKG")
print(f"paved islands -> sidewalk: {len(paved)} ({P.area:.0f} m2) | non-paved islands left out: {len(veg)} ({V.area:.0f} m2) | "
      f"road {road.area/1e4:.2f} -> {road2.area/1e4:.2f} ha")
