"""Fetch imagery from online services (XYZ/TMS tiles or WMS) into a GeoTIFF.

The resulting GeoTIFF is in the service's native CRS (EPSG:3857 for XYZ tiles,
the requested CRS for WMS), so the exported Shapefiles match the service exactly.
"""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor

import mercantile
import numpy as np
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds

UA = {"User-Agent": "geoai-detect/0.1 (personal use)"}
MAX_PIXELS = 400_000_000


def _get(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.content


def fetch_xyz(url_template: str, bbox_wgs84: tuple[float, float, float, float], zoom: int,
              out_path: str, tile_size: int = 256, workers: int = 8) -> str:
    """Download XYZ tiles covering bbox (west, south, east, north) at a zoom level.

    url_template: e.g. https://server/{z}/{x}/{y}.png  ({-y} for TMS is supported)
    """
    w, s, e, n = bbox_wgs84
    tiles = list(mercantile.tiles(w, s, e, n, zoom))
    if not tiles:
        raise ValueError("bbox produced no tiles")
    xs = [t.x for t in tiles]; ys = [t.y for t in tiles]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W, H = (x1 - x0 + 1) * tile_size, (y1 - y0 + 1) * tile_size
    if W * H > MAX_PIXELS:
        raise ValueError(f"Area too large at zoom {zoom} ({W}x{H} px). Use a smaller bbox or lower zoom.")
    mosaic = np.zeros((H, W, 3), np.uint8)

    def load(t):
        url = (url_template.replace("{z}", str(t.z)).replace("{x}", str(t.x))
               .replace("{-y}", str((1 << t.z) - 1 - t.y)).replace("{y}", str(t.y)))
        try:
            img = Image.open(io.BytesIO(_get(url))).convert("RGB")
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.BILINEAR)
            return t, np.asarray(img)
        except Exception:
            return t, None

    with ThreadPoolExecutor(workers) as ex:
        for t, arr in ex.map(load, tiles):
            if arr is not None:
                r, c = (t.y - y0) * tile_size, (t.x - x0) * tile_size
                mosaic[r:r + tile_size, c:c + tile_size] = arr

    ul = mercantile.xy_bounds(x0, y0, zoom)
    lr = mercantile.xy_bounds(x1, y1, zoom)
    transform = from_bounds(ul.left, lr.bottom, lr.right, ul.top, W, H)
    _write(out_path, mosaic, transform, "EPSG:3857")
    return out_path


def fetch_wms(base_url: str, layer: str, bbox: tuple[float, float, float, float], crs: str,
              gsd: float, out_path: str, version: str = "1.3.0", fmt: str = "image/png",
              style: str = "", chunk: int = 2048) -> str:
    """Download a WMS GetMap mosaic for bbox (minx, miny, maxx, maxy) in `crs` at `gsd`
    units/pixel. Requests are chunked to respect typical server size limits."""
    minx, miny, maxx, maxy = bbox
    W = int(math.ceil((maxx - minx) / gsd)); H = int(math.ceil((maxy - miny) / gsd))
    if W * H > MAX_PIXELS:
        raise ValueError(f"Area too large ({W}x{H} px) at gsd={gsd}")
    mosaic = np.zeros((H, W, 3), np.uint8)
    axis_swap = version == "1.3.0" and crs.upper() in ("EPSG:4326",)
    sep = "&" if "?" in base_url else "?"
    jobs = []
    for r in range(0, H, chunk):
        for c in range(0, W, chunk):
            h, w = min(chunk, H - r), min(chunk, W - c)
            bx0 = minx + c * gsd; bx1 = bx0 + w * gsd
            by1 = maxy - r * gsd; by0 = by1 - h * gsd
            b = (by0, bx0, by1, bx1) if axis_swap else (bx0, by0, bx1, by1)
            crs_key = "CRS" if version == "1.3.0" else "SRS"
            url = (f"{base_url}{sep}SERVICE=WMS&VERSION={version}&REQUEST=GetMap&LAYERS={layer}"
                   f"&STYLES={style}&{crs_key}={crs}&BBOX={','.join(map(str, b))}"
                   f"&WIDTH={w}&HEIGHT={h}&FORMAT={fmt}")
            jobs.append((r, c, h, w, url))

    def load(j):
        r, c, h, w, url = j
        img = Image.open(io.BytesIO(_get(url, 120))).convert("RGB")
        return r, c, h, w, np.asarray(img)

    with ThreadPoolExecutor(4) as ex:
        for r, c, h, w, arr in ex.map(load, jobs):
            mosaic[r:r + h, c:c + w] = arr[:h, :w]

    transform = from_bounds(minx, maxy - H * gsd, minx + W * gsd, maxy, W, H)
    _write(out_path, mosaic, transform, crs)
    return out_path


def wmts_layers(capabilities_url: str) -> list[dict]:
    """List layers and their tile matrix sets from a WMTS GetCapabilities URL."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(_get(capabilities_url, 60))
    ns = {"w": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1"}
    sets = {}
    for S in root.findall(".//w:Contents/w:TileMatrixSet", ns):
        crs = S.findtext("ows:SupportedCRS", namespaces=ns) or ""
        epsg = "EPSG:" + crs.rstrip("/").split(":")[-1].split("/")[-1]
        mats = []
        for m in S.findall("w:TileMatrix", ns):
            tl = [float(v) for v in m.findtext("w:TopLeftCorner", namespaces=ns).split()]
            mats.append({
                "id": m.findtext("ows:Identifier", namespaces=ns),
                "res": float(m.findtext("w:ScaleDenominator", namespaces=ns)) * 0.00028,  # m/px (projected CRS)
                "tl": tl, "tw": int(m.findtext("w:TileWidth", namespaces=ns)),
                "th": int(m.findtext("w:TileHeight", namespaces=ns)),
                "mw": int(m.findtext("w:MatrixWidth", namespaces=ns)), "mh": int(m.findtext("w:MatrixHeight", namespaces=ns)),
            })
        sets[S.findtext("ows:Identifier", namespaces=ns)] = {"crs": epsg, "matrices": mats}
    get_url = None
    for op in root.findall(".//ows:Operation", ns):
        if op.get("name") == "GetTile":
            g = op.find(".//ows:Get", ns)
            if g is not None:
                get_url = g.get("{http://www.w3.org/1999/xlink}href")
    layers = []
    for L in root.findall(".//w:Contents/w:Layer", ns):
        bb = L.find("ows:WGS84BoundingBox", ns)
        wgs = None
        if bb is not None:
            lo = [float(v) for v in bb.findtext("ows:LowerCorner", namespaces=ns).split()]
            hi = [float(v) for v in bb.findtext("ows:UpperCorner", namespaces=ns).split()]
            wgs = lo + hi
        rurl = L.find("w:ResourceURL[@resourceType='tile']", ns)
        layers.append({
            "id": L.findtext("ows:Identifier", namespaces=ns), "title": L.findtext("ows:Title", namespaces=ns),
            "formats": [f.text for f in L.findall("w:Format", ns)],
            "style": (L.find("w:Style/ows:Identifier", ns).text if L.find("w:Style/ows:Identifier", ns) is not None else "default"),
            "sets": {s.findtext("w:TileMatrixSet", namespaces=ns): sets.get(s.findtext("w:TileMatrixSet", namespaces=ns))
                     for s in L.findall("w:TileMatrixSetLink", ns)},
            "bbox_wgs84": wgs, "template": rurl.get("template") if rurl is not None else None, "kvp": get_url,
        })
    return layers


def fetch_wmts(capabilities_url: str, bbox_wgs84: tuple[float, float, float, float], gsd: float, out_path: str,
               layer: str | None = None, matrix_set: str | None = None, workers: int = 8) -> str:
    """Download a WMTS mosaic covering bbox at the level closest to `gsd` m/px.

    The GeoTIFF is written in the tile matrix set's own CRS (e.g. ITM EPSG:2039),
    so detections are exported in that CRS without resampling."""
    from rasterio.warp import transform_bounds
    layers = wmts_layers(capabilities_url)
    L = next((l for l in layers if l["id"] == layer), None) if layer else layers[0]
    if L is None:
        raise ValueError(f"layer {layer!r} not in service; available: {[l['id'] for l in layers]}")
    sid = matrix_set or next(iter(L["sets"]))
    S = L["sets"][sid]
    crs = S["crs"]
    # finest level that is not finer than needed (avoid downloading 4x the pixels)
    mats = sorted(S["matrices"], key=lambda m: m["res"])
    M = next((m for m in mats if m["res"] >= gsd * 0.75), mats[-1])
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, *bbox_wgs84, densify_pts=21)
    x0, y0 = M["tl"][0], M["tl"][1]
    if crs in ("EPSG:4326",) or (abs(x0) <= 90 and abs(y0) > 90):   # lat/lon axis order
        x0, y0 = y0, x0
    tw, th, res = M["tw"], M["th"], M["res"]
    c0 = int((minx - x0) // (tw * res)); c1 = int((maxx - x0) // (tw * res))
    r0 = int((y0 - maxy) // (th * res)); r1 = int((y0 - miny) // (th * res))
    W, H = (c1 - c0 + 1) * tw, (r1 - r0 + 1) * th
    if W * H > MAX_PIXELS:
        raise ValueError(f"Area too large at {res:.3f} m/px ({W}x{H} px). Use a smaller area or coarser resolution.")
    fmt = "image/jpeg" if "image/jpeg" in L["formats"] else L["formats"][0]
    base = L["kvp"] or capabilities_url.split("?")[0] + "?"
    sep = "" if base.endswith(("?", "&")) else ("&" if "?" in base else "?")

    def url(r, c):
        if L["template"]:
            return (L["template"].replace("{TileMatrixSet}", sid).replace("{TileMatrix}", M["id"])
                    .replace("{TileRow}", str(r)).replace("{TileCol}", str(c)).replace("{Style}", L["style"]))
        return (f"{base}{sep}SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER={L['id']}&STYLE={L['style']}"
                f"&TILEMATRIXSET={sid}&TILEMATRIX={M['id']}&TILEROW={r}&TILECOL={c}&FORMAT={fmt}")

    mosaic = np.zeros((H, W, 3), np.uint8)

    def load(rc):
        r, c = rc
        try:
            img = Image.open(io.BytesIO(_get(url(r, c)))).convert("RGB")
            return r, c, np.asarray(img)
        except Exception:
            return r, c, None

    jobs = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    with ThreadPoolExecutor(workers) as ex:
        for r, c, arr in ex.map(load, jobs):
            if arr is not None:
                rr, cc = (r - r0) * th, (c - c0) * tw
                mosaic[rr:rr + th, cc:cc + tw] = arr[:th, :tw]
    left = x0 + c0 * tw * res; top = y0 - r0 * th * res
    transform = from_bounds(left, top - H * res, left + W * res, top, W, H)
    _write(out_path, mosaic, transform, crs)
    return out_path


def zoom_for_gsd(gsd_m: float, lat: float) -> int:
    """Pick the XYZ zoom whose resolution is closest to the requested meters/pixel."""
    z0 = 156543.03392 * math.cos(math.radians(lat))
    return max(0, min(22, round(math.log2(z0 / gsd_m))))


def _write(path, rgb, transform, crs):
    H, W, _ = rgb.shape
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
                       crs=crs, transform=transform, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512, photometric="RGB") as dst:
        dst.write(rgb.transpose(2, 0, 1))
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
