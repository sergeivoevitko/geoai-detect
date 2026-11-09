"""GeoAI Detect work area: backend API.

Run as a desktop window:   python run_app.py        (Windows: "GeoAI Detect.bat")
Run in a browser:          uvicorn server.app:app --port 8000

Concepts
  source   an image file on disk (read in place) or an online service (WMTS / WMS / XYZ)
  run      one detection on a rectangle of a source; its pixels are saved as runs/<id>/aoi.tif
  project  all detected features of the current work session (any number of runs)
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime
from functools import lru_cache

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyproj import CRS, Transformer
from rasterio.warp import transform_bounds
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from shapely import wkt as swkt
from shapely.geometry import box, mapping
from shapely.ops import transform as stransform

from geoai import models, sources
from geoai.export import write_layers
from geoai.pipeline import DetectOptions, detect_image, interactive_segment
from geoai.raster import GeoImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("GEOAI_WORKSPACE", os.path.join(ROOT, "workspace"))
for d in ("runs", "uploads", "exports", "tilecache"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
SOURCES_JSON = os.path.join(WORK, "sources.json")
PROJECT_JSON = os.path.join(WORK, "project.json")

# GDAL reads WMTS/WMS services for the live map; keep a disk cache of their tiles
os.environ.setdefault("GDAL_DEFAULT_WMS_CACHE_PATH", os.path.join(WORK, "tilecache"))
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.jp2,.img")

app = FastAPI(title="GeoAI Detect")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "web")), name="static")

_lock = threading.RLock()
_jobs: dict[str, dict] = {}
_rasters: dict[str, GeoImage] = {}
_features: list[dict] | None = None

CLASS_ORDER = ["building", "road", "road_centerline", "sidewalk", "parking", "tree", "solar panel"]


# ================================================================ persistence
def _load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def sources_reg() -> dict:
    return _load(SOURCES_JSON, {})


def project() -> dict:
    return _load(PROJECT_JSON, {"runs": {}, "created": datetime.now().isoformat(timespec="seconds")})


def features() -> list[dict]:
    global _features
    with _lock:
        if _features is None:
            _features = []
            p = project()
            for r in p.get("features", []):
                r = dict(r); r["geometry"] = swkt.loads(r.pop("wkt"))
                _features.append(r)
        return _features


def save_project(p: dict | None = None):
    with _lock:
        p = p or project()
        p["features"] = [{**{k: v for k, v in f.items() if k != "geometry"}, "wkt": f["geometry"].wkt}
                         for f in features()]
        _save(PROJECT_JSON, p)


@lru_cache(maxsize=64)
def _tf(src: str, dst: str):
    return Transformer.from_crs(src, dst, always_xy=True).transform


def raster(path: str) -> GeoImage:
    with _lock:
        if path not in _rasters:
            _rasters[path] = GeoImage(path)
        return _rasters[path]


# ==================================================================== pages
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/config")
def config():
    from geoai.solar import yolo_available
    return {"device": models.DEVICE, "solar_model": yolo_available(),
            "desktop": os.environ.get("GEOAI_DESKTOP") == "1",
            "classes": [
                {"id": "building", "name": "Buildings", "default": True},
                {"id": "road", "name": "Roads", "default": True,
                 "options": [{"id": "centerlines", "name": "Centerlines", "default": True},
                             {"id": "sidewalks", "name": "Sidewalks", "default": True},
                             {"id": "parking", "name": "Parking lots", "default": True}]},
                {"id": "tree", "name": "Trees", "default": True},
                {"id": "solar panel", "name": "Solar panels", "default": True},
            ]}


# ================================================================== sources
def _source_info(sid: str) -> dict:
    s = sources_reg().get(sid)
    if not s:
        raise HTTPException(404, "source not found")
    info = {"id": sid, **{k: v for k, v in s.items() if k not in ("url",)}}
    info["has_key"] = "url" in s
    return info


@app.get("/api/sources")
def list_sources():
    return [_source_info(k) for k in sources_reg()]


class FileReq(BaseModel):
    path: str


def _register_file(path: str, name: str | None = None) -> dict:
    try:
        img = GeoImage(path)
    except Exception as e:
        raise HTTPException(400, f"Not a usable georeferenced image: {e}")
    with Reader(path) as src:
        maxz = int(src.maxzoom); minz = int(src.minzoom)
    sid = uuid.uuid4().hex[:10]
    reg = sources_reg()
    reg[sid] = {"kind": "file", "name": name or os.path.basename(path), "path": path,
                "crs": img.crs.to_string(), "gsd_m": round(img.gsd_meters, 4),
                "size": [img.ds.width, img.ds.height], "bands": img.ds.count,
                "bounds_wgs84": list(transform_bounds(img.crs, "EPSG:4326", *img.ds.bounds, densify_pts=21)),
                "min_zoom": minz, "max_zoom": maxz + 2}
    _save(SOURCES_JSON, reg)
    return _source_info(sid)


@app.post("/api/sources/file")
def add_file(req: FileReq):
    """Desktop mode: register a local image in place (no copy, any size)."""
    p = req.path.strip().strip('"')
    if not os.path.exists(p):
        raise HTTPException(400, f"File not found: {p}")
    return _register_file(p)


@app.post("/api/sources/upload")
async def upload(file: UploadFile = File(...)):
    """Browser mode: upload a file."""
    d = os.path.join(WORK, "uploads", uuid.uuid4().hex[:8]); os.makedirs(d)
    path = os.path.join(d, os.path.basename(file.filename or "image.tif"))
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return _register_file(path)
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise


class ServiceReq(BaseModel):
    kind: str                   # wmts | wms | xyz
    url: str
    name: str = ""
    layer: str = ""
    crs: str = ""               # WMS only: request CRS for downloads (default EPSG:3857)


@app.post("/api/services/inspect")
def inspect_service(req: ServiceReq):
    """WMTS: list layers so the user can pick one."""
    if req.kind != "wmts":
        return {"layers": []}
    try:
        ls = sources.wmts_layers(req.url)
    except Exception as e:
        raise HTTPException(400, f"Could not read the service: {e}")
    return {"layers": [{"id": l["id"], "title": l["title"] or l["id"], "bbox_wgs84": l["bbox_wgs84"],
                        "levels_m": sorted({round(m["res"], 3) for s in l["sets"].values() if s
                                            for m in s["matrices"]})} for l in ls]}


@app.post("/api/sources/service")
def add_service(req: ServiceReq):
    s = {"kind": req.kind, "url": req.url.strip(), "layer": req.layer.strip(), "name": req.name.strip()}
    if req.kind == "wmts":
        ls = sources.wmts_layers(s["url"])
        L = next((l for l in ls if l["id"] == s["layer"]), ls[0] if ls else None)
        if L is None:
            raise HTTPException(400, "The service has no layers")
        s["layer"] = L["id"]
        sets = [v for v in L["sets"].values() if v]
        s["crs"] = sets[0]["crs"] if sets else "EPSG:3857"
        s["bounds_wgs84"] = L["bbox_wgs84"] or [34.2, 29.4, 35.9, 33.4]
        s["levels_m"] = sorted({round(m["res"], 3) for v in sets for m in v["matrices"]})
        s["name"] = s["name"] or (L["title"] or L["id"])
    elif req.kind == "wms":
        s["crs"] = req.crs or "EPSG:3857"
        s["bounds_wgs84"] = [34.2, 29.4, 35.9, 33.4]
        s["name"] = s["name"] or (s["layer"] or "WMS")
    elif req.kind == "xyz":
        s["crs"] = "EPSG:3857"
        s["bounds_wgs84"] = [-180, -85, 180, 85]
        s["name"] = s["name"] or "XYZ tiles"
    else:
        raise HTTPException(400, "kind must be wmts, wms or xyz")
    s["min_zoom"], s["max_zoom"] = 0, 22
    sid = uuid.uuid4().hex[:10]
    reg = sources_reg(); reg[sid] = s; _save(SOURCES_JSON, reg)
    return _source_info(sid)


@app.delete("/api/sources/{sid}")
def remove_source(sid: str):
    reg = sources_reg(); reg.pop(sid, None); _save(SOURCES_JSON, reg)
    return {"ok": True}


def _gdal_url(s: dict) -> str:
    if s["kind"] == "wmts":
        return f"WMTS:{s['url']},layer={s['layer']}"
    raise ValueError


_http = requests.Session()
_http.headers["User-Agent"] = "geoai-detect/0.2 (personal use)"


@app.get("/api/sources/{sid}/tiles/{z}/{x}/{y}.png")
def source_tile(sid: str, z: int, x: int, y: int):
    s = sources_reg().get(sid)
    if not s:
        raise HTTPException(404)
    try:
        if s["kind"] == "file":
            img = raster(s["path"])
            with Reader(s["path"]) as src:
                data = src.tile(x, y, z, indexes=img.bands, tilesize=256)
            if img._stretch is not None:
                lo, hi = img._stretch
                data.rescale(in_range=list(zip(lo.tolist(), hi.tolist())))
            png = data.render(img_format="PNG")
        elif s["kind"] == "wmts":
            with Reader(_gdal_url(s)) as src:
                data = src.tile(x, y, z, indexes=(1, 2, 3), tilesize=256)
            png = data.render(img_format="PNG")
        elif s["kind"] == "xyz":
            url = (s["url"].replace("{z}", str(z)).replace("{x}", str(x))
                   .replace("{-y}", str((1 << z) - 1 - y)).replace("{y}", str(y)))
            r = _http.get(url, timeout=30); r.raise_for_status()
            return Response(r.content, media_type=r.headers.get("content-type", "image/png"),
                            headers={"Cache-Control": "max-age=86400"})
        elif s["kind"] == "wms":
            import mercantile
            b = mercantile.xy_bounds(x, y, z)
            sep = "&" if "?" in s["url"] else "?"
            url = (f"{s['url']}{sep}SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={s['layer']}&STYLES="
                   f"&SRS=EPSG:3857&BBOX={b.left},{b.bottom},{b.right},{b.top}&WIDTH=256&HEIGHT=256"
                   f"&FORMAT=image/png&TRANSPARENT=TRUE")
            r = _http.get(url, timeout=30); r.raise_for_status()
            return Response(r.content, media_type=r.headers.get("content-type", "image/png"),
                            headers={"Cache-Control": "max-age=86400"})
        else:
            raise HTTPException(400)
    except TileOutsideBounds:
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception:
        return Response(status_code=204)
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


# ============================================================ detection runs
class DetectReq(BaseModel):
    source: str
    bbox: list[float]                   # rectangle, WGS84 west, south, east, north
    classes: list[str]
    options: dict = {}                  # centerlines / sidewalks / parking: bool
    gsd: float = 0.15                   # services: download resolution (m/px)
    custom: list[str] = []              # extra free-text classes (open-vocabulary detector)


class Cancelled(Exception):
    pass


def _prepare_aoi(s: dict, bbox: list[float], gsd: float, out: str, prog) -> str:
    """Write the rectangle's pixels to a GeoTIFF in the source's own CRS."""
    if s["kind"] == "file":
        import rasterio
        from rasterio.windows import from_bounds
        src = raster(s["path"]).ds
        bb = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        bb = (max(bb[0], src.bounds.left), max(bb[1], src.bounds.bottom),
              min(bb[2], src.bounds.right), min(bb[3], src.bounds.top))
        if bb[0] >= bb[2] or bb[1] >= bb[3]:
            raise ValueError("The rectangle does not overlap the image.")
        win = from_bounds(*bb, transform=src.transform).round_offsets().round_lengths()
        prof = src.profile.copy()
        prof.update(driver="GTiff", width=win.width, height=win.height, transform=src.window_transform(win),
                    compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        prof.pop("photometric", None)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(src.read(window=win))
        return out
    prog(0.02, "Downloading imagery for the rectangle…")
    if s["kind"] == "wmts":
        return sources.fetch_wmts(s["url"], tuple(bbox), gsd, out, layer=s["layer"])
    if s["kind"] == "wms":
        crs = s.get("crs") or "EPSG:3857"
        bb = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
        return sources.fetch_wms(s["url"], s["layer"], bb, crs, gsd, out)
    if s["kind"] == "xyz":
        z = sources.zoom_for_gsd(gsd, (bbox[1] + bbox[3]) / 2)
        return sources.fetch_xyz(s["url"], tuple(bbox), z, out)
    raise ValueError("unknown source kind")


MARGIN_M = 20.0
OBJECT_CLASSES = {"building", "solar panel", "parking"}     # kept whole if their centre is inside


def _expand_bbox(bbox, meters):
    import math
    dlat = meters / 111320.0
    dlon = meters / (111320.0 * math.cos(math.radians((bbox[1] + bbox[3]) / 2)))
    return [bbox[0] - dlon, bbox[1] - dlat, bbox[2] + dlon, bbox[3] + dlat]


def _clip_to_rect(feats, bbox, crs):
    """Objects (buildings, panels, lots): keep whole if the centre is inside the rectangle.
    Surfaces and lines (roads, sidewalks, trees, centerlines): cut exactly at the rectangle."""
    from shapely.geometry import Polygon
    from shapely.ops import transform as st
    w, s_, e, n = bbox
    ring = [(w + (e - w) * t, s_) for t in np.linspace(0, 1, 11)] + [(e, s_ + (n - s_) * t) for t in np.linspace(0, 1, 11)] + \
           [(e - (e - w) * t, n) for t in np.linspace(0, 1, 11)] + [(w, n - (n - s_) * t) for t in np.linspace(0, 1, 11)]
    rect = st(_tf("EPSG:4326", crs), Polygon(ring)).buffer(0)
    out = []
    for f in feats:
        g = f["geometry"]
        if f["label"] in OBJECT_CLASSES:
            if rect.contains(g.representative_point()):
                out.append(f)
            continue
        if not g.intersects(rect):
            continue
        c = g.intersection(rect)
        parts = [c] if c.geom_type in ("Polygon", "LineString") else \
                [p for p in getattr(c, "geoms", []) if p.geom_type == g.geom_type]
        for p in parts:
            if p.is_empty or (p.geom_type == "Polygon" and p.area < 1.0) or (p.geom_type == "LineString" and p.length < 1.0):
                continue
            out.append({**f, "geometry": p})
    return out


@app.post("/api/detect")
def start_detect(req: DetectReq):
    s = sources_reg().get(req.source)
    if not s:
        raise HTTPException(404, "source not found")
    w, so, e, n = req.bbox
    if e <= w or n <= so:
        raise HTTPException(400, "Draw a rectangle first")
    classes = [c for c in req.classes] + [c.strip().lower() for c in req.custom if c.strip()]
    if not classes:
        raise HTTPException(400, "Choose at least one class")
    rid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    jid = rid
    rdir = os.path.join(WORK, "runs", rid); os.makedirs(rdir)
    _jobs[jid] = {"status": "running", "progress": 0.0, "message": "Starting…", "cancel": False, "run": rid}
    opt_flags = {"centerlines": True, "sidewalks": True, "parking": True, **(req.options or {})}

    def run():
        t0 = time.time()

        def prog(p, m):
            if _jobs[jid]["cancel"]:
                raise Cancelled()
            _jobs[jid].update(progress=float(p), message=m)
        try:
            # detect on the rectangle plus a margin (context for the models at the edges), then
            # keep only what belongs to the rectangle
            aoi = _prepare_aoi(s, _expand_bbox(req.bbox, MARGIN_M), req.gsd, os.path.join(rdir, "aoi.tif"), prog)
            img = GeoImage(aoi)
            opt = DetectOptions(classes=classes, engine="hybrid", overlap=170,
                                sidewalks=opt_flags["sidewalks"])
            prog(0.05, "Loading AI models…")
            feats = detect_image(img, opt, lambda p, m: prog(0.05 + 0.93 * p, m))
            if not opt_flags["centerlines"]:
                feats = [f for f in feats if not f["label"].endswith("_centerline")]
            if not opt_flags["parking"]:
                feats = [f for f in feats if f["label"] != "parking"]
            crs = img.crs.to_string()
            feats = _clip_to_rect(feats, req.bbox, crs)
            with _lock:
                for f in feats:
                    f.update(fid=uuid.uuid4().hex[:12], run=rid, crs=crs)
                    features().append(f)
                p = project()
                p["runs"][rid] = {"source": req.source, "source_name": s["name"], "bbox": req.bbox,
                                  "classes": classes, "aoi": aoi, "crs": crs,
                                  "gsd_m": round(img.gsd_meters, 3), "seconds": int(time.time() - t0),
                                  "count": len(feats)}
                save_project(p)
            _jobs[jid].update(status="done", progress=1.0,
                              message=f"Found {len(feats)} objects in {int(time.time() - t0)} s")
        except Cancelled:
            shutil.rmtree(rdir, ignore_errors=True)
            _jobs[jid].update(status="cancelled", message="Cancelled")
        except Exception as ex:  # noqa
            import traceback; traceback.print_exc()
            shutil.rmtree(rdir, ignore_errors=True)
            _jobs[jid].update(status="error", message=f"{type(ex).__name__}: {ex}")

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in _jobs[jid].items() if k != "cancel"}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    if jid in _jobs:
        _jobs[jid]["cancel"] = True
    return {"ok": True}


@app.get("/api/estimate")
def estimate(source: str, w: float, s: float, e: float, n: float, gsd: float = 0.15):
    """Rough size / time estimate for the rectangle (CPU numbers measured on 4 cores)."""
    src = sources_reg().get(source)
    if not src:
        raise HTTPException(404)
    t = _tf("EPSG:4326", "EPSG:2039" if 29 < s < 34 and 34 < w < 36 else "EPSG:3857")
    x0, y0 = t(w, s); x1, y1 = t(e, n)
    area_ha = abs(x1 - x0) * abs(y1 - y0) / 1e4
    g = src.get("gsd_m", gsd) if src["kind"] == "file" else gsd
    mpix = area_ha * 1e4 / (g * g) / 1e6
    cpu = models.DEVICE == "cpu"
    lo, hi = (0.2, 0.6) if cpu else (0.02, 0.08)          # minutes per hectare at ~15 cm
    scale = (0.15 / max(g, 0.05)) ** 2
    return {"area_ha": round(area_ha, 2), "megapixels": round(mpix, 1),
            "minutes": [round(area_ha * lo * scale, 1), round(area_ha * hi * scale, 1)],
            "too_big": mpix > 900}


# =================================================================== project
@app.get("/api/project")
def get_project():
    p = project()
    counts: dict[str, int] = {}
    for f in features():
        counts[f["label"]] = counts.get(f["label"], 0) + 1
    return {"runs": [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "aoi"}} for k, v in p["runs"].items()],
            "counts": counts}


@app.get("/api/features")
def features_geojson():
    out = []
    for f in features():
        g = stransform(_tf(f["crs"], "EPSG:4326"), f["geometry"])
        out.append({"type": "Feature", "id": f["fid"],
                    "properties": {"fid": f["fid"], "class": f["label"], "score": round(f.get("score", 1.0), 3),
                                   "source": f.get("source", "auto"), "run": f.get("run"),
                                   **(f.get("attrs") or {})},
                    "geometry": mapping(g)})
    return {"type": "FeatureCollection", "features": out}


@app.delete("/api/features/{fid}")
def delete_feature(fid: str):
    global _features
    with _lock:
        _features = [f for f in features() if f["fid"] != fid]
        save_project()
    return {"ok": True}


@app.delete("/api/runs/{rid}")
def delete_run(rid: str):
    global _features
    with _lock:
        _features = [f for f in features() if f.get("run") != rid]
        p = project(); r = p["runs"].pop(rid, None); save_project(p)
    if r:
        shutil.rmtree(os.path.dirname(r["aoi"]), ignore_errors=True)
    return {"ok": True}


@app.post("/api/project/clear")
def clear_project():
    global _features
    with _lock:
        p = project()
        for r in p["runs"].values():
            shutil.rmtree(os.path.dirname(r["aoi"]), ignore_errors=True)
        _features = []
        save_project({"runs": {}, "created": datetime.now().isoformat(timespec="seconds")})
    return {"ok": True}


# ======================================================= interactive editing
class SegmentReq(BaseModel):
    source: str = ""
    label: str = "building"
    points: list[list[float]] | None = None   # [[lng, lat, 1|0], ...]
    box: list[float] | None = None            # [west, south, east, north]
    commit: bool = False


def _raster_for(lng: float, lat: float, source: str):
    """Image to segment on: the source file itself, else the latest run that covers the point."""
    s = sources_reg().get(source)
    if s and s["kind"] == "file":
        return s["path"], None
    p = project()
    for rid, r in reversed(list(p["runs"].items())):
        w, so, e, n = r["bbox"]
        if w <= lng <= e and so <= lat <= n and os.path.exists(r["aoi"]):
            return r["aoi"], rid
    raise HTTPException(422, "For services, click inside an area that was already detected "
                             "(the rectangle's imagery is needed for editing).")


@app.post("/api/segment")
def segment(req: SegmentReq):
    from shapely.ops import unary_union
    from geoai.vectorize import regularize_building
    ref = req.points[0] if req.points else [(req.box[0] + req.box[2]) / 2, (req.box[1] + req.box[3]) / 2]
    path, rid = _raster_for(ref[0], ref[1], req.source)
    img = raster(path)
    crs = img.crs.to_string()
    fw = _tf("EPSG:4326", crs)
    pts = lbl = bx = None
    if req.points:
        pts = [fw(p[0], p[1]) for p in req.points]
        lbl = [int(p[2]) if len(p) > 2 else 1 for p in req.points]
    if req.box:
        x0, y0 = fw(req.box[0], req.box[1]); x1, y1 = fw(req.box[2], req.box[3])
        bx = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    polys, score = interactive_segment(img, path, pts, lbl, bx)
    if not polys:
        raise HTTPException(422, "Nothing segmented here. Try another click or a box.")
    geom = unary_union(polys)
    label = req.label.strip().lower() or "building"
    if label == "building" and geom.geom_type == "Polygon":
        geom = regularize_building(geom, img.gsd * 1.5)
    else:
        geom = geom.simplify(img.gsd, preserve_topology=True)
    if req.commit:
        with _lock:
            features().append({"fid": uuid.uuid4().hex[:12], "label": label, "geometry": geom, "score": score,
                               "source": "manual", "run": rid, "crs": crs})
            save_project()
    return {"feature": {"type": "Feature", "properties": {"class": label},
                        "geometry": mapping(stransform(_tf(crs, "EPSG:4326"), geom))}, "score": score}


# ==================================================================== export
class ExportReq(BaseModel):
    formats: list[str] = ["gdb"]        # gdb, shp
    crs: str = "source"                 # source | EPSG:2039 | EPSG:4326 | any EPSG code
    folder: str = ""                    # desktop: write here; empty -> zip for download
    name: str = ""
    classes: list[str] | None = None    # None = all


def _target_crs(req_crs: str) -> str:
    if req_crs and req_crs != "source":
        CRS.from_user_input(req_crs)    # validates
        return req_crs
    crss = [f["crs"] for f in features()]
    return max(set(crss), key=crss.count) if crss else "EPSG:4326"


@app.post("/api/export")
def export(req: ExportReq):
    feats = [f for f in features() if req.classes is None or f["label"] in req.classes]
    if not feats:
        raise HTTPException(400, "Nothing to export yet")
    target = _target_crs(req.crs)
    name = req.name.strip() or f"geoai_{datetime.now():%Y%m%d_%H%M}"
    folder = req.folder.strip().strip('"')
    zip_it = not folder
    out_dir = folder or os.path.join(WORK, "exports", uuid.uuid4().hex[:8])
    os.makedirs(out_dir, exist_ok=True)
    try:
        written = write_layers(feats, target, out_dir, name, req.formats)
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")
    if not zip_it:
        return {"ok": True, "paths": written, "crs": target}
    zpath = os.path.join(out_dir, name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in written:
            if os.path.isdir(p):
                for r, _, fs in os.walk(p):
                    for fn in fs:
                        full = os.path.join(r, fn)
                        z.write(full, os.path.relpath(full, out_dir))
            else:
                z.write(p, os.path.relpath(p, out_dir))
    return {"ok": True, "download": f"/api/download/{os.path.basename(out_dir)}/{name}.zip", "crs": target}


@app.get("/api/download/{d}/{fname}")
def download(d: str, fname: str):
    p = os.path.join(WORK, "exports", os.path.basename(d), os.path.basename(fname))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, filename=fname, media_type="application/zip")
