"""Make road and sidewalk layers share one curb line (no overlap).
usage: resolve.py road.gpkg sidewalk.(shp|gpkg) centerline.gpkg out_prefix"""
import sys, geopandas as gpd
from shapely.ops import unary_union
roadp, sidep, clp, out = sys.argv[1:5]
r = gpd.read_file(roadp); crs = r.crs
road = unary_union(list(r.geometry))
side = unary_union(list(gpd.read_file(sidep).to_crs(crs).geometry))
cl = unary_union(list(gpd.read_file(clp).to_crs(crs).geometry))
ov = road.intersection(side)
road_wins, side_wins = [], []
for p in getattr(ov, "geoms", [ov]):
    if p.geom_type != "Polygon" or p.area < 1:
        continue
    core = p.buffer(-1.5)                       # survives => wider than ~3 m
    (road_wins if (not core.is_empty and p.intersects(cl)) else side_wins).append(p)
rw, sw = unary_union(road_wins), unary_union(side_wins)
road2 = road.difference(sw.buffer(0.05)) if side_wins else road
side2 = side.difference(road2.buffer(0.05))
smooth = lambda g: g.buffer(0.5, join_style=1).buffer(-1.0, join_style=1).buffer(0.5, join_style=1).simplify(0.5, preserve_topology=True)
side2 = smooth(side2).difference(road2.buffer(0.05))   # one curb line: sidewalk is cut by the final road edge
parts = lambda g: [x for x in getattr(g, "geoms", [g]) if x.geom_type == "Polygon" and x.area > 3]
gpd.GeoDataFrame(geometry=parts(road2), crs=crs).to_file(f"{out}_road.gpkg", driver="GPKG")
gpd.GeoDataFrame(geometry=parts(side2), crs=crs).to_file(f"{out}_sidewalk.gpkg", driver="GPKG")
print(f"overlap {ov.area:.0f} m2 -> road kept {rw.area:.0f} m2, sidewalk kept {sw.area:.0f} m2 | "
      f"remaining overlap {unary_union(parts(road2)).intersection(unary_union(parts(side2))).area:.1f} m2")
