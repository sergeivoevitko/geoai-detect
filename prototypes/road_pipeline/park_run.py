"""Esri ParkingLotClassification_USA (mmseg DeepLabV3+ R101-D8) rebuilt in plain PyTorch.
usage: park_run.py weights.pth image.tif out.gpkg [res=0.3]"""
import sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, rasterio, geopandas as gpd
from rasterio import features
from rasterio.enums import Resampling
from shapely.geometry import shape
from torchvision.models import resnet101

def cbr(i, o, k=1, d=1, groups=1):
    m = nn.Sequential(); m.conv = nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, groups=groups, bias=False)
    m.bn = nn.BatchNorm2d(o); m.act = nn.ReLU(True); return m
def dsc(i, o, d=1):
    m = nn.Sequential(); m.depthwise_conv = cbr(i, i, 3, d, groups=i); m.pointwise_conv = cbr(i, o, 1); return m

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        r = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
                                  nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
                                  nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.maxpool = r.maxpool; self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
    def forward(self, x):
        x = self.maxpool(self.stem(x)); c1 = self.layer1(x)
        return c1, self.layer4(self.layer3(self.layer2(c1)))

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), cbr(2048, 512))
        self.aspp_modules = nn.ModuleList([cbr(2048, 512)] + [dsc(2048, 512, d) for d in (12, 24, 36)])
        self.bottleneck = cbr(2560, 512, 3)
        self.c1_bottleneck = cbr(256, 48)
        self.sep_bottleneck = nn.Sequential(dsc(560, 512), dsc(512, 512))
        self.conv_seg = nn.Conv2d(512, 2, 1)
    def forward(self, c1, x):
        p = F.interpolate(self.image_pool(x), size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.bottleneck(torch.cat([p] + [m(x) for m in self.aspp_modules], 1))
        x = F.interpolate(x, size=c1.shape[2:], mode="bilinear", align_corners=False)
        return self.conv_seg(self.sep_bottleneck(torch.cat([x, self.c1_bottleneck(c1)], 1)))

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = Backbone(); self.decode_head = Head()
    def forward(self, x):
        y = self.decode_head(*self.backbone(x))
        return F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)

w, img, out = sys.argv[1:4]; res = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
torch.set_num_threads(4)
sd = torch.load(w, map_location="cpu", weights_only=True)
sd = {k: v for k, v in sd.items() if not k.startswith("auxiliary_head")}
net = Net(); print(net.load_state_dict(sd, strict=True)); net.eval()
ds = rasterio.open(img); s = abs(ds.res[0]) / res
H, W = int(ds.height * s), int(ds.width * s)
a = ds.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.bilinear).astype(np.float32) / 255.0
a = (a - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
tr = ds.transform * rasterio.Affine.scale(ds.width / W, ds.height / H)
C, ov = 512, 64; step = C - 2 * ov
prob = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
pad = np.pad(a, ((0, 0), (0, max(0, C - H)), (0, max(0, C - W))), mode="reflect"); PH, PW = pad.shape[1:]
ys = list(range(0, PH - C + 1, step)); xs = list(range(0, PW - C + 1, step))
if ys[-1] != PH - C: ys.append(PH - C)
if xs[-1] != PW - C: xs.append(PW - C)
t0 = time.time()
with torch.inference_mode():
    for y in ys:
        for x in xs:
            t = torch.from_numpy(pad[:, y:y + C, x:x + C].copy())[None]
            p = (torch.softmax(net(t), 1) + torch.softmax(net(t.flip(3)), 1).flip(3))[0, 1].numpy() / 2
            y1, x1 = min(y + C, H), min(x + C, W)
            prob[y:y1, x:x1] += p[:y1 - y, :x1 - x]; cnt[y:y1, x:x1] += 1
m = (prob / np.maximum(cnt, 1)) > 0.5
geoms = [shape(g).buffer(0) for g, _ in features.shapes(m.astype(np.uint8), mask=m, transform=tr)]
g = gpd.GeoDataFrame(geometry=[x for x in geoms if x.area > 50], crs=ds.crs); g.to_file(out, driver="GPKG")
print(f"parking: {len(g)} areas, {g.area.sum()/1e4:.2f} ha, {time.time()-t0:.0f}s")
