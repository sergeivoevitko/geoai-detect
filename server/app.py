"""GeoAI Detect web server.

Run:  uvicorn server.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from shapely import wkt as swkt
from shapely.geometry import mapping, shape
from shapely.ops import transform as stransform

from geoai import models, sources
from geoai.export import export_shapefiles, zip_shapefiles, safe_name
from geoai.pipeline import DetectOptions, detect_image, interactive_segment
from geoai.raster import GeoImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS = os.path.join(ROOT, "uploads")
OUTPUTS = os.path.join(ROOT, "outputs")
REGISTRY = os.path.join(UPLOADS, "registry.json")
os.makedirs(UPLOADS, exist_ok=True); os.makedirs(OUTPUTS, exist_ok=True)

app = FastAPI(title="GeoAI Detect")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "web")), name="static")

_lock = threading.RLock()
_jobs: dict[str, dict] = {}
_images: dict[str, GeoImage] = {}
_features: dict[str, list[dict]] = {}


# ------------------------------------------------------------- persistence
def registry() -> dict:
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=1)


def register(path: str, name: str) -> str:
    iid = uuid.uuid4().hex[:10]
    reg = registry(); reg[iid] = {"name": name, "path": path}; save_registry(reg)
    return iid


def get_image(iid: str) -> GeoImage:
    with _lock:
        if iid not in _images:
            reg = registry()
            if iid not in reg:
                raise HTTPException(404, "image not found")
            _images[iid] = GeoImage(reg[iid]["path"])
        return _images[iid]


def feats_path(iid):
    return os.path.join(OUTPUTS, iid, "features.json")


def get_features(iid: str) -> list[dict]:
    with _lock:
        if iid not in _features:
            p = feats_path(iid)
            items = []
            if os.path.exists(p):
                with open(p) as f:
                    for r in json.load(f):
                        r["geometry"] = swkt.loads(r.pop("wkt")); items.append(r)
            _features[iid] = items
        return _features[iid]


def save_features(iid: str):
    os.makedirs(os.path.dirname(feats_path(iid)), exist_ok=True)
    with _lock:
        rows = [{**{k: v for k, v in f.items() if k != "geometry"}, "wkt": f["geometry"].wkt}
                for f in _features.get(iid, [])]
    tmp = feats_path(iid) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f)
    os.replace(tmp, feats_path(iid))


def add_features(iid: str, new: list[dict]):
    with _lock:
        items = get_features(iid)
        for f in new:
            f["fid"] = uuid.uuid4().hex[:12]
            items.append(f)
    save_features(iid)


@lru_cache(maxsize=64)
def to_wgs84(crs_str: str):
    return Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True).transform


@lru_cache(maxsize=64)
def from_wgs84(crs_str: str):
    return Transformer.from_crs("EPSG:4326", crs_str, always_xy=True).transform


def image_info(iid: str) -> dict:
    img = get_image(iid)
    info = img.info()
    info["id"] = iid
    info["name"] = registry()[iid]["name"]
    info["bounds_wgs84"] = list(transform_bounds(img.crs, "EPSG:4326", *img.ds.bounds, densify_pts=21))
    with Reader(img.path) as src:
        info["max_zoom"] = int(src.maxzoom)
    return info


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(ROOT, "web", "index.html")) as f:
        return f.read()


@lru_cache(maxsize=1)
def _engines():
    return models.engines()


@app.get("/api/config")
def config():
    eng = _engines()
    reg = registry()
    return {"device": models.DEVICE, "engines": eng,
            "default_engine": "hybrid",
            "last_image": list(reg)[-1] if reg else None}


# ----------------------------------------------------------------- images
@app.get("/api/images")
def list_images():
    return [{"id": k, "name": v["name"]} for k, v in registry().items() if os.path.exists(v["path"])]


@app.post("/api/images")
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "image.tif")
    iid_dir = os.path.join(UPLOADS, uuid.uuid4().hex[:8]); os.makedirs(iid_dir)
    path = os.path.join(iid_dir, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        GeoImage(path).close()
    except Exception as e:
        shutil.rmtree(iid_dir, ignore_errors=True)
        raise HTTPException(400, f"Not a usable georeferenced raster: {e}")
    return image_info(register(path, name))


class FetchReq(BaseModel):
    type: str = "xyz"
    url: str
    bbox: list[float]           # WGS84 west, south, east, north
    gsd: float = 0.3
    layer: str = ""
    crs: str = "EPSG:3857"


@app.post("/api/images/fetch")
def fetch(req: FetchReq):
    d = os.path.join(UPLOADS, uuid.uuid4().hex[:8]); os.makedirs(d)
    path = os.path.join(d, "service.tif")
    try:
        if req.type == "wms":
            bb = transform_bounds("EPSG:4326", req.crs, *req.bbox)
            sources.fetch_wms(req.url, req.layer, bb, req.crs, req.gsd, path)
        else:
            lat = (req.bbox[1] + req.bbox[3]) / 2
            z = sources.zoom_for_gsd(req.gsd, lat)
            sources.fetch_xyz(req.url, tuple(req.bbox), z, path)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Fetch failed: {e}")
    return image_info(register(path, f"service {req.type} {req.bbox[0]:.4f},{req.bbox[1]:.4f}"))


@app.get("/api/images/{iid}")
def get_info(iid: str):
    return image_info(iid)


_EMPTY_PNG = None


@app.get("/api/images/{iid}/tiles/{z}/{x}/{y}.png")
def tile(iid: str, z: int, x: int, y: int):
    img = get_image(iid)
    try:
        with Reader(img.path) as src:
            data = src.tile(x, y, z, indexes=img.bands, tilesize=256)
    except TileOutsideBounds:
        return Response(status_code=204)
    if img._stretch is not None:
        lo, hi = img._stretch
        data.rescale(in_range=list(zip(lo.tolist(), hi.tolist())))
    return Response(data.render(img_format="PNG"), media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


# --------------------------------------------------------------- features
@app.get("/api/images/{iid}/features")
def features_geojson(iid: str):
    img = get_image(iid)
    tf = to_wgs84(img.crs.to_string())
    out = []
    for f in get_features(iid):
        out.append({"type": "Feature", "id": f["fid"],
                    "properties": {"fid": f["fid"], "class": f["label"], "score": round(f["score"], 3),
                                   "source": f.get("source", "auto")},
                    "geometry": mapping(stransform(tf, f["geometry"]))})
    return {"type": "FeatureCollection", "features": out}


@app.delete("/api/images/{iid}/features/{fid}")
def delete_feature(iid: str, fid: str):
    with _lock:
        items = get_features(iid)
        _features[iid] = [f for f in items if f["fid"] != fid]
    save_features(iid)
    return {"ok": True}


@app.delete("/api/images/{iid}/features")
def delete_features(iid: str, label: str | None = None):
    with _lock:
        get_features(iid)
        _features[iid] = [f for f in _features[iid] if label is not None and f["label"] != label]
    save_features(iid)
    return {"ok": True}


# -------------------------------------------------------------- detection
class DetectReq(BaseModel):
    classes: list[str]
    engine: str = "hybrid"
    threshold: float = 0.3
    tile_size: int = 1024
    min_area: float = 4.0
    regularize: bool = True
    replace: bool = True


@app.post("/api/images/{iid}/detect")
def start_detect(iid: str, req: DetectReq):
    img = get_image(iid)
    jid = uuid.uuid4().hex[:10]
    _jobs[jid] = {"status": "running", "progress": 0.0, "message": "Loading AI models…"}
    sam3_ok = next(e["available"] for e in _engines() if e["id"] == "sam3")
    opt = DetectOptions(classes=[c.strip().lower() for c in req.classes if c.strip()], engine=req.engine, sam3_ok=sam3_ok,
                        threshold=req.threshold, tile_size=req.tile_size,
                        overlap=max(128, req.tile_size // 6), min_area=req.min_area, regularize=req.regularize)

    def run():
        try:
            def prog(p, m):
                _jobs[jid].update(progress=p, message=m)
            feats = detect_image(img, opt, prog)
            with _lock:
                items = get_features(iid)
                if req.replace:
                    _features[iid] = [f for f in items if not (f.get("source") == "auto" and (
                        f["label"] in opt.classes or f["label"].rsplit("_centerline", 1)[0] in opt.classes))]
            add_features(iid, feats)
            _jobs[jid].update(status="done", progress=1.0)
        except Exception as e:  # noqa
            import traceback; traceback.print_exc()
            _jobs[jid].update(status="error", message=f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "job not found")
    return _jobs[jid]


class SegmentReq(BaseModel):
    label: str = "object"
    points: list[list[float]] | None = None   # [[lng, lat, 1|0], ...]
    box: list[float] | None = None            # [west, south, east, north]
    commit: bool = False


@app.post("/api/images/{iid}/segment")
def segment(iid: str, req: SegmentReq):
    img = get_image(iid)
    crs = img.crs.to_string()
    fw = from_wgs84(crs)
    pts = lbl = bx = None
    if req.points:
        pts = [fw(p[0], p[1]) for p in req.points]
        lbl = [int(p[2]) if len(p) > 2 else 1 for p in req.points]
    if req.box:
        x0, y0 = fw(req.box[0], req.box[1]); x1, y1 = fw(req.box[2], req.box[3])
        bx = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if not pts and not bx:
        raise HTTPException(400, "give points or a box")
    polys, score = interactive_segment(img, iid, pts, lbl, bx)
    if not polys:
        raise HTTPException(422, "Nothing segmented here. Try another click or a box.")
    from shapely.ops import unary_union
    from geoai.vectorize import regularize_building
    geom = unary_union(polys)
    label = req.label.strip().lower() or "object"
    if label in {"building", "buildings", "house"} and geom.geom_type == "Polygon":
        geom = regularize_building(geom, img.gsd * 1.5)
    else:
        geom = geom.simplify(img.gsd, preserve_topology=True)
    if req.commit:
        add_features(iid, [{"label": label, "geometry": geom, "score": score, "source": "manual"}])
    feat = {"type": "Feature", "properties": {"class": label},
            "geometry": mapping(stransform(to_wgs84(crs), geom))}
    return {"feature": feat, "score": score}


# ----------------------------------------------------------------- export
@app.get("/api/images/{iid}/export.zip")
def export(iid: str):
    img = get_image(iid)
    feats = get_features(iid)
    if not feats:
        raise HTTPException(400, "nothing to export")
    name = safe_name(os.path.splitext(registry()[iid]["name"])[0])
    out_dir = os.path.join(OUTPUTS, iid, "shp")
    shutil.rmtree(out_dir, ignore_errors=True)
    shps = export_shapefiles(feats, img.crs, out_dir)
    z = zip_shapefiles(shps, os.path.join(OUTPUTS, iid, f"{name}_detections.zip"))
    return FileResponse(z, filename=os.path.basename(z), media_type="application/zip")
