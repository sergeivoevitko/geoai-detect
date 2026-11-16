import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from rasterio.windows import Window
from PIL import Image
img, side, roadA, roadB, out, crop = sys.argv[1:7]
c = [float(v) for v in crop.split(",")]
d = rasterio.open(img); win = Window(int(d.width*c[0]), int(d.height*c[1]), int(d.width*c[2]), int(d.width*c[2]))
S = 800 / win.width; H = W = 800
base = d.read([1,2,3], window=win, out_shape=(3,H,W), boundless=True).transpose(1,2,0).copy()
t = d.window_transform(win) * rasterio.Affine.scale(1/S)
def m(p):
    g = gpd.read_file(p).to_crs(d.crs)
    return features.rasterize([(x,1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool) if len(g) else np.zeros((H,W),bool)
ms = m(side)
def paint(rp):
    v = base.copy(); mr = m(rp)
    v[ms & ~mr] = (0.35*v[ms & ~mr] + 0.65*np.array([176,108,255])).astype(np.uint8)
    v[mr & ~ms] = (0.4*v[mr & ~ms] + 0.6*np.array([245,130,49])).astype(np.uint8)
    v[mr & ms] = [255, 0, 0]                                   # overlap = red
    return v
sep = np.full((H,8,3),255,np.uint8)
Image.fromarray(np.concatenate([base, sep, paint(roadA), sep, paint(roadB)],1)).save(out, quality=90)
