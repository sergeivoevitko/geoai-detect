import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from PIL import Image
img, cur, other, out = sys.argv[1:5]
d = rasterio.open(img)
S = 900 / max(d.width, d.height)
H, W = int(d.height * S), int(d.width * S)
base = d.read([1, 2, 3], out_shape=(3, H, W)).transpose(1, 2, 0).copy()
t = d.transform * rasterio.Affine.scale(d.width / W, d.height / H)
tiles = [base]
for p in (cur, other):
    g = gpd.read_file(p).to_crs(d.crs); v = base.copy()
    m = features.rasterize([(x, 1) for x in g.geometry], out_shape=(H, W), transform=t).astype(bool)
    v[m] = (0.5 * v[m] + [115, 10, 35]).astype(np.uint8)
    e = features.rasterize([(x.boundary, 1) for x in g.geometry], out_shape=(H, W), transform=t).astype(bool)
    v[e] = [255, 0, 0]; tiles.append(v)
sep = np.full((H, 8, 3), 255, np.uint8)
Image.fromarray(np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)).save(out, quality=88)
