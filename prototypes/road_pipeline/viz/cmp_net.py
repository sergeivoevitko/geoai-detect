import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from rasterio.windows import Window
from PIL import Image
img, r0, c0, r1, c1, out = sys.argv[1:7]
crop = [float(v) for v in sys.argv[7].split(",")] if len(sys.argv) > 7 else None
d = rasterio.open(img)
win = Window(int(d.width*crop[0]), int(d.height*crop[1]), int(d.width*crop[2]), int(d.width*crop[2])) if crop else Window(0, 0, d.width, d.height)
S = 900 / max(win.width, win.height); H, W = int(win.height*S), int(win.width*S)
base = d.read([1,2,3], window=win, out_shape=(3,H,W), boundless=True).transpose(1,2,0).copy()
t = d.window_transform(win) * rasterio.Affine.scale(1/S); px = abs(d.res[0]) / S
def paint(rp, cp):
    v = base.copy()
    g = gpd.read_file(rp).to_crs(d.crs)
    m = features.rasterize([(x,1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool)
    v[m] = (0.4*v[m] + 0.6*np.array([245,130,49])).astype(np.uint8)
    e = features.rasterize([(x.boundary.buffer(px*0.7),1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool)
    v[e] = [255,255,255]
    c = gpd.read_file(cp).to_crs(d.crs)
    for src, col in (("detected",[255,235,0]), ("inferred",[0,229,255])):
        sub = c[c["source"] == src] if "source" in c.columns else (c if src == "detected" else c.iloc[0:0])
        if len(sub):
            mm = features.rasterize([(x.buffer(px*1.1),1) for x in sub.geometry], out_shape=(H,W), transform=t).astype(bool)
            v[mm] = col
    return v
sep = np.full((H,8,3),255,np.uint8)
Image.fromarray(np.concatenate([base, sep, paint(r0,c0), sep, paint(r1,c1)],1)).save(out, quality=90)
