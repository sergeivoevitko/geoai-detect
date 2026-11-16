"""Sidewalk completion prototype.
For each side of each street segment: if detected sidewalk covers >= MIN_COVER of that side,
fill the missing stretches with a strip along the road edge at the measured sidewalk width.
usage: swfill.py road.gpkg sidewalk.gpkg centerline.gpkg out_prefix [building] [parking] [crosswalk]"""
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, substring

MIN_COVER = 0.40     # share of a side that must already have detected sidewalk
MAX_SW = 5.0         # m, widest sidewalk considered
STEP = 2.0           # m, station spacing along the street
roadp, swp, clp, out = sys.argv[1:5]
extra = sys.argv[5:8]
r = gpd.read_file(roadp); crs = r.crs
U = lambda p: unary_union(list(gpd.read_file(p).to_crs(crs).geometry)) if p else Polygon()
road, sw = U(roadp), U(swp)
blocked = unary_union([U(p) for p in extra if p] + [road])
cl = gpd.read_file(clp).to_crs(crs)


def station_box(ln, t0, t1, side, depth):
    seg = substring(ln, t0, t1)
    if seg.length < 0.5:
        return Polygon()
    return seg.buffer(side * depth, single_sided=True, cap_style=2)


fills, n_sides, n_filled, n_median = [], 0, 0, 0
for _, rec in cl.iterrows():
    ln = rec.geometry
    if ln is None or ln.length < 20 or rec.get("class") == "highway":
        continue
    hw = float(rec.get("width_m", 8.0)) / 2
    for side in (1, -1):
        n_sides += 1
        band = ln.buffer(side * (hw + MAX_SW), single_sided=True, cap_style=2).difference(road)
        sws = sw.intersection(band)
        ts = np.arange(4.0, ln.length - 4.0, STEP)        # skip the junction ends
        if len(ts) < 4:
            continue
        cov = np.array([station_box(ln, t, t + STEP, side, hw + MAX_SW).intersection(sws).area > 0.5 for t in ts])
        if cov.mean() < MIN_COVER or cov.all():
            continue
        covered_len = cov.sum() * STEP
        w_sw = float(np.clip(sws.area / max(covered_len, 1), 1.5, MAX_SW))
        strip = ln.buffer(side * (hw + w_sw), single_sided=True, cap_style=2).difference(road)
        gaps = unary_union([station_box(ln, t, t + STEP, side, hw + w_sw + 1) for t, c in zip(ts, cov) if not c])
        f = strip.intersection(gaps).difference(blocked).difference(sw)
        f = f.buffer(-0.4).buffer(0.4)
        own = ln.buffer(hw + 1.0, cap_style=2)
        parts = []
        for p in getattr(f, "geoms", [f]):
            if p.geom_type != "Polygon" or p.area < 4:
                continue
            other_road = road.intersection(p.buffer(4.0)).difference(own)     # road beyond the strip's outer edge
            if other_road.area > 5:                                            # a median between two carriageways
                n_median += 1; continue
            parts.append(p)
        if parts:
            fills += parts; n_filled += 1

add = unary_union(fills) if fills else Polygon()
gpd.GeoDataFrame({"source": ["inferred"] * len(fills)}, geometry=fills, crs=crs).to_file(f"{out}_sw_added.gpkg", driver="GPKG")
allsw = unary_union([sw, add]).buffer(0.3).buffer(-0.3).difference(road)
gpd.GeoDataFrame(geometry=[p for p in getattr(allsw, "geoms", [allsw]) if p.area > 2], crs=crs).to_file(f"{out}_sidewalk.gpkg", driver="GPKG")
print(f"medians skipped {n_median} | street sides checked {n_sides}, filled on {n_filled} sides, added {add.area:.0f} m2 "
      f"(sidewalk {sw.area/1e4:.2f} -> {allsw.area/1e4:.2f} ha), overlap with road {allsw.intersection(road).area:.1f} m2")
