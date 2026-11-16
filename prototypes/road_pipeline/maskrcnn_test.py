"""Run a torchvision Mask R-CNN state dict (e.g. Esri arcgis.learn MaskRCNN) on a georeferenced image.
usage: maskrcnn_test.py weights.pth image.tif out.gpkg model_gsd chip [gt.geojson] [--norm imagenet|none]"""
import sys, time
sys.path.insert(0, "/home/user/geoai-detect")
import numpy as np, torch, rasterio, geopandas as gpd
from rasterio import features
from rasterio.enums import Resampling
from shapely.geometry import shape, box
from torchvision.models.detection import maskrcnn_resnet50_fpn
from geoai.vectorize import merge_instances

w, img_path, out, model_gsd, chip = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
gt = sys.argv[6] if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else None
torch.set_num_threads(4)
m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=chip, max_size=chip)
m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True)); m.eval()
m.roi_heads.score_thresh = 0.5; m.roi_heads.detections_per_img = 300

ds = rasterio.open(img_path)
s = abs(ds.res[0]) / model_gsd
H, W = int(ds.height * s), int(ds.width * s)
a = ds.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.bilinear).astype(np.float32) / 255.0
tr = ds.transform * rasterio.Affine.scale(ds.width / W, ds.height / H)
ov = chip // 5; step = chip - ov
items = []
t0 = time.time()
with torch.inference_mode():
    for y in range(0, max(H - ov, 1), step):
        for x in range(0, max(W - ov, 1), step):
            y0, x0 = min(y, max(0, H - chip)), min(x, max(0, W - chip))
            t = torch.from_numpy(a[:, y0:y0 + chip, x0:x0 + chip])
            o = m([t])[0]
            wt = tr * rasterio.Affine.translation(x0, y0)
            for mk, sc in zip(o["masks"][:, 0].numpy(), o["scores"].numpy()):
                b = mk > 0.5
                if b.sum() < 20:
                    continue
                # drop objects cut by an interior tile edge (a neighbouring tile sees them whole)
                edge = (x0 > 0 and b[:, :3].any()) or (y0 > 0 and b[:3].any()) or \
                       (x0 + chip < W and b[:, -3:].any()) or (y0 + chip < H and b[-3:].any())
                if edge and b.sum() < (ov * ov):
                    continue
                for g, v in features.shapes(b.astype(np.uint8), mask=b, transform=wt):
                    p = shape(g).buffer(0)
                    if p.area > 10:
                        items.append({"geometry": p, "score": float(sc)})
merged = merge_instances(items)
g = gpd.GeoDataFrame({"score": [i["score"] for i in merged]}, geometry=[i["geometry"] for i in merged], crs=ds.crs)
g = g[g.area >= 15]
g.to_file(out, driver="GPKG")
line = f"{len(g)} buildings, {g.area.sum() / 1e4:.2f} ha, {time.time() - t0:.0f}s"
if gt:
    from evaluate import evaluate
    r = evaluate(g, gpd.read_file(gt).to_crs(ds.crs), box(*ds.bounds))
    line += (f" | area IoU {r['area']['iou']:.3f} F1 {r['area']['f1']:.3f} | object P {r['object']['precision']:.3f} "
             f"R {r['object']['recall']:.3f} F1 {r['object']['f1']:.3f}")
print(line)
