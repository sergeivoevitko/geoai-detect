import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from rasterio.windows import Window
from PIL import Image
img, out, fx, fy, fw = sys.argv[1:6]; paths = sys.argv[6:]
d = rasterio.open(img); w = int(d.width*float(fw)); win = Window(int(d.width*float(fx)), int(d.height*float(fy)), w, w)
H = W = 700; S = W / w
base = d.read([1,2,3], window=win, out_shape=(3,H,W)).transpose(1,2,0).copy()
t = d.window_transform(win) * rasterio.Affine.scale(1/S)
px = abs(d.res[0]) / S
tiles = []
for p in paths:
    g = gpd.read_file(p).to_crs(d.crs); v = base.copy()
    m = features.rasterize([(x,1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool)
    v[m] = (0.5*v[m] + 0.5*np.array([230,25,75])).astype(np.uint8)
    e = features.rasterize([(x.boundary.buffer(px*0.8),1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool)
    v[e] = [255,255,0]; tiles.append(v)
sep = np.full((H,8,3),255,np.uint8); row=[base]
for x in tiles: row += [sep, x]
Image.fromarray(np.concatenate(row,1)).save(out, quality=92)
