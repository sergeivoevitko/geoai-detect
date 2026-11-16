import sys, os, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from PIL import Image
img, app, ped, out = sys.argv[1:5]
d = rasterio.open(img); S = 900 / max(d.width, d.height)
H, W = int(d.height * S), int(d.width * S)
base = d.read([1, 2, 3], out_shape=(3, H, W)).transpose(1, 2, 0).copy()
t = d.transform * rasterio.Affine.scale(d.width / W, d.height / H)
COL = {"road": [245, 130, 49], "sidewalk": [176, 108, 255], "crosswalk": [0, 255, 255], "parking": [140, 140, 140]}
def paint(paths):
    v = base.copy()
    for n, p in paths:
        if not os.path.exists(p): continue
        g = gpd.read_file(p).to_crs(d.crs)
        if not len(g): continue
        m = features.rasterize([(x, 1) for x in g.geometry], out_shape=(H, W), transform=t).astype(bool)
        v[m] = (0.35 * v[m] + 0.65 * np.array(COL[n])).astype(np.uint8)
    return v
a = paint([(n, f"{app}_{n}.shp") for n in ("parking", "road", "sidewalk")])
b = paint([(n, f"{ped}_{n}.gpkg" if os.path.exists(f"{ped}_{n}.gpkg") else f"{ped}_{n}.shp") for n in ("parking", "road", "sidewalk", "crosswalk")])
sep = np.full((H, 8, 3), 255, np.uint8)
Image.fromarray(np.concatenate([base, sep, a, sep, b], 1)).save(out, quality=88)
