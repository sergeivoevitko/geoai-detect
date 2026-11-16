import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from rasterio.windows import Window
from PIL import Image
img, a, b, out, fx, fy, fw = sys.argv[1:8]
d = rasterio.open(img); w = int(d.width * float(fw)); win = Window(int(d.width*float(fx)), int(d.height*float(fy)), w, w)
S = 900 / w; H = W = 900
base = d.read([1,2,3], window=win, out_shape=(3,H,W)).transpose(1,2,0).copy()
t = d.window_transform(win) * rasterio.Affine.scale(1/S)
tiles=[base]
for p in (a,b):
    g = gpd.read_file(p).to_crs(d.crs); v = base.copy()
    if len(g):
        m = features.rasterize([(x,1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool)
        v[m] = (0.4*v[m] + [0,60,150]).astype(np.uint8)
        e = features.rasterize([(x.boundary,1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool)
        v[e] = [0,255,255]
    tiles.append(v)
sep = np.full((H,8,3),255,np.uint8)
Image.fromarray(np.concatenate([tiles[0],sep,tiles[1],sep,tiles[2]],1)).save(out, quality=90)
