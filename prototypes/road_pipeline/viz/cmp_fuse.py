import sys, numpy as np, rasterio, geopandas as gpd
from rasterio import features
from PIL import Image
img, base, added, out = sys.argv[1:5]
d = rasterio.open(img); S = 900 / max(d.width, d.height); H, W = int(d.height*S), int(d.width*S)
b0 = d.read([1,2,3], out_shape=(3,H,W)).transpose(1,2,0).copy()
t = d.transform * rasterio.Affine.scale(d.width/W, d.height/H)
def m(p): g = gpd.read_file(p).to_crs(d.crs); return features.rasterize([(x,1) for x in g.geometry], out_shape=(H,W), transform=t).astype(bool) if len(g) else np.zeros((H,W),bool)
v1 = b0.copy(); mb = m(base); v1[mb] = (0.35*v1[mb] + 0.65*np.array([245,130,49])).astype(np.uint8)
v2 = v1.copy(); ma = m(added); v2[ma] = (0.3*v2[ma] + 0.7*np.array([0,200,255])).astype(np.uint8)
sep = np.full((H,8,3),255,np.uint8)
Image.fromarray(np.concatenate([b0,sep,v1,sep,v2],1)).save(out, quality=88)
