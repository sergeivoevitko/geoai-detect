import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from rasterio.windows import Window
from PIL import Image
img, r0, s0, r1, s1, isl, out, crop = sys.argv[1:9]
c = [float(v) for v in crop.split(",")]
d = rasterio.open(img); win = Window(int(d.width*c[0]), int(d.height*c[1]), int(d.width*c[2]), int(d.width*c[2]))
S = 800 / win.width; H = W = 800
base = d.read([1,2,3], window=win, out_shape=(3,H,W), boundless=True).transpose(1,2,0).copy()
t = d.window_transform(win) * rasterio.Affine.scale(1/S); px = abs(d.res[0]) / S
def m(geoms): return features.rasterize([(x,1) for x in geoms], out_shape=(H,W), transform=t).astype(bool) if len(geoms) else np.zeros((H,W),bool)
G = lambda p: list(gpd.read_file(p).to_crs(d.crs).geometry)
def paint(rp, sp, outline):
    v = base.copy(); mr, ms = m(G(rp)), m(G(sp))
    v[mr] = (0.4*v[mr] + 0.6*np.array([245,130,49])).astype(np.uint8)
    v[ms] = (0.35*v[ms] + 0.65*np.array([176,108,255])).astype(np.uint8)
    if outline:
        ig = gpd.read_file(isl).to_crs(d.crs)
        for typ, col in (("paved",[255,255,255]),("not paved",[0,255,0])):
            e = m([x.boundary.buffer(px*1.2) for x in ig[ig["type"]==typ].geometry]); v[e] = col
    return v
sep = np.full((H,8,3),255,np.uint8)
Image.fromarray(np.concatenate([base, sep, paint(r0,s0,False), sep, paint(r1,s1,True)],1)).save(out, quality=90)
