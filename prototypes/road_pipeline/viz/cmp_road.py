import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from PIL import Image
img, pa, pb, out = sys.argv[1:5]
d = rasterio.open(img); S = 900 / max(d.width, d.height)
H, W = int(d.height * S), int(d.width * S)
base = d.read([1, 2, 3], out_shape=(3, H, W)).transpose(1, 2, 0).copy()
t = d.transform * rasterio.Affine.scale(d.width / W, d.height / H)
tiles = [base]
for p in (pa, pb):
    v = base.copy()
    g = gpd.read_file(p + "_road.shp").to_crs(d.crs)
    if len(g):
        m = features.rasterize([(x, 1) for x in g.geometry], out_shape=(H, W), transform=t).astype(bool)
        v[m] = (0.45 * v[m] + [135, 70, 25]).astype(np.uint8)
    c = gpd.read_file(p + "_road_centerline.shp").to_crs(d.crs)
    if len(c):
        e = features.rasterize([(x.buffer(abs(d.res[0]) / S * 0.9), 1) for x in c.geometry], out_shape=(H, W), transform=t).astype(bool)
        v[e] = [255, 230, 0]
    tiles.append(v)
sep = np.full((H, 8, 3), 255, np.uint8)
Image.fromarray(np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)).save(out, quality=88)
