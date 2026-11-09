"""End-to-end detection: GeoTIFF -> tiles -> AI masks -> merged polygons (source CRS)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import rasterio.features
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import Point, shape

from . import landcover, models
from .raster import GeoImage
from .vectorize import clean_mask, mask_to_polygons, postprocess, is_stuff


@dataclass
class DetectOptions:
    classes: list[str]
    engine: str = "hybrid"           # hybrid | sam3 | gdino_sam2
    sam3_ok: bool = False            # use SAM 3 for open-vocabulary classes in hybrid mode
    tta: bool = True                 # test-time augmentation (flips) for land-cover model
    refine_buildings: bool = True    # SAM 2 edge refinement of land-cover buildings
    road_graph: bool = True          # SAM-Road network: centerlines + connected road polygons
    road_min_half_width: float = 2.5   # m, guaranteed corridor around each centerline
    road_max_half_width: float = 6.0   # m, "pavement" pixels (sidewalk/parking-like) allowed this far
    road_wide_half_width: float = 22.0 # m, "road" pixels allowed this far (multi-lane highways)
    junction_radius: float = 15.0      # m, intersections: fill paved pixels within this radius
    bridge_max_gap: float = 70.0       # m, join road ends across paved gaps up to this length
    road_grow_max: float = 60.0        # m, grow road surface through connected road pixels
    road_fine: str = "auto"            # second road pass at finer resolution: auto | always | never
    road_fine_gsd: float = 0.5         # m/px of the fine pass
    dense_building_frac: float = 0.3   # "auto": run the fine pass when buildings cover more than this
    spur_min_m: float = 6.0            # m, prune centerline side-branches shorter than this
    sidewalks: bool = True             # split road surface into carriageway + sidewalk
    sidewalk_max_m: float = 4.5        # m, widest sidewalk strip considered
    review_threshold: float = 0.6      # segments with lower agreement are flagged for review
    threshold: float = 0.3
    tile_size: int = 1024
    overlap: int = 192
    min_area: float = 4.0            # m^2 (CRS units^2)
    regularize: bool = True
    simplify_px: float = 1.0
    target_gsd: float | None = None  # resample to this m/px before inference (None = native)


ProgressFn = Callable[[float, str], None]
SOLAR_NAMES = {"solar panel", "solar panels", "solar", "pv", "photovoltaic", "solar water heater"}


def _edge_cut(mask: np.ndarray, touches: tuple, margin: int = 4) -> bool:
    """True if the mask touches an interior tile seam (object is likely cut off and
    a neighbouring overlapping tile will see it whole)."""
    l, t, r, b = touches
    return ((l and mask[:, :margin].any()) or (t and mask[:margin, :].any()) or
            (r and mask[:, -margin:].any()) or (b and mask[-margin:, :].any()))


def detect_image(img: GeoImage, opt: DetectOptions, progress: ProgressFn | None = None) -> list[dict]:
    """Run detection for all requested classes.

    engine "hybrid" (recommended): classes the land-cover model knows (road, tree, building,
    water, …) come from pixel-wise segmentation (+ SAM 2 edge refinement for buildings);
    any other class goes through the open-vocabulary detector (SAM 3 if available,
    else GroundingDINO + SAM 2).
    """
    progress = progress or (lambda p, m: None)
    t0 = time.time()
    features: list[dict] = []
    lc_classes, ov_classes = [], list(opt.classes)
    solar_classes = []
    if opt.engine == "hybrid":
        solar_classes = [c for c in opt.classes if c.strip().lower() in SOLAR_NAMES]
        lc_classes = [c for c in opt.classes if landcover.landcover_class(c)]
        ov_classes = [c for c in opt.classes if not landcover.landcover_class(c) and c not in solar_classes]
        if solar_classes and not any(landcover.landcover_class(c) == "building" for c in lc_classes):
            lc_classes.append("building")          # panels are searched on detected roofs
            drop_buildings = True
        else:
            drop_buildings = False
    ov_engine = opt.engine if opt.engine != "hybrid" else ("sam3" if opt.sam3_ok else "gdino_sam2")
    share = len(lc_classes) / max(1, len(opt.classes)) if ov_classes else 1.0

    if lc_classes:
        features += _detect_landcover(img, opt, lc_classes,
                                      lambda p, m: progress(0.9 * share * p, m))
    if solar_classes:
        from .solar import detect_on_roofs, detect_yolo, merge, yolo_available
        roofs = [f["geometry"] for f in features if f["label"] == "__roof_raw"] or \
                [f["geometry"] for f in features if landcover.landcover_class(f["label"]) == "building"]
        sol = detect_on_roofs(img, roofs, progress=lambda p, m: progress(0.9 * share, m))
        if yolo_available():
            # rules find small water-heater collectors; the trained model finds PV arrays and
            # panels on red-tile roofs. They overlap only ~50%, so both are used and merged.
            sol = merge(sol, detect_yolo(img, roofs, progress=lambda p, m: progress(0.9 * share, m)), img.gsd)
        for it in sol:
            features.append({"label": solar_classes[0], "geometry": it["geometry"], "score": it["score"],
                             "source": "auto"})
        if drop_buildings:
            features = [f for f in features if landcover.landcover_class(f["label"]) != "building"]
    features = [f for f in features if f["label"] != "__roof_raw"]
    if ov_classes:
        o = DetectOptions(**{**opt.__dict__, "classes": ov_classes, "engine": ov_engine})
        base = 0.9 * share if lc_classes else 0.0
        features += _detect_open_vocab(img, o, lambda p, m: progress(base + (0.95 - base) * p, m))
    progress(1.0, f"Found {len(features)} objects in {int(time.time() - t0)} s")
    return features


def _detect_landcover(img: GeoImage, opt: DetectOptions, classes: list[str], progress: ProgressFn) -> list[dict]:
    cls_map, conf, names = landcover.segment_raster(img, tile=opt.tile_size, tta=opt.tta,
                                                    progress=lambda p, m: progress(0.8 * p, m))
    idx_of = {v: k for k, v in names.items()}
    gsd = img.gsd
    min_px = max(8, int(opt.min_area / (gsd * gsd)))
    out = []
    for label in classes:
        progress(0.82, f"Vectorizing {label}…")
        k = idx_of[landcover.landcover_class(label)]
        mask = cls_map == k
        stuff = is_stuff(label)
        if landcover.landcover_class(label) == "road" and opt.road_graph:
            from .roads import extract_roads
            cl, mask, side, parking = extract_roads(img, cls_map, idx_of, opt,
                                                    lambda p, m: progress(0.82 + 0.1 * p, m))
            for c in cl:
                out.append({"label": f"{label}_centerline", "source": "auto", **c})
            for extra, emask, emin in (("sidewalk", side, 2.0), ("parking", parking, 150.0)):
                if not emask.any():
                    continue
                emask = clean_mask(emask, min_pixels=min_px, close_iter=1)
                for geom, v in rasterio.features.shapes(emask.astype(np.uint8), mask=emask,
                                                        transform=img.ds.transform, connectivity=8):
                    g = shape(geom).buffer(0)
                    for it in postprocess(extra, [{"geometry": g, "score": 0.8}], gsd, emin, 1.5, False):
                        out.append({"label": extra, "geometry": it["geometry"], "score": it["score"],
                                    "source": "auto"})
        mask = clean_mask(mask, min_pixels=min_px, close_iter=2 if stuff else 1)
        if not mask.any():
            continue
        lab, n = ndimage.label(mask)
        scores = ndimage.mean(conf, lab, np.arange(1, n + 1)) / 255.0
        if landcover.landcover_class(label) == "building" and opt.refine_buildings:
            items = _refine_buildings_with_sam(img, lab, n, scores,
                                               lambda p, m: progress(0.85 + 0.15 * p, m))
            # full building area before splitting: used as the search area for solar panels, so
            # roof edges that the roof-by-roof split leaves out are still searched
            for it in items:
                out.append({"label": "__roof_raw", "geometry": it["geometry"], "score": it["score"],
                            "source": "auto"})
            from .buildings import split_large, split_with_sam
            # touching buildings merged into one blob: SAM roof-by-roof first, edge split after
            items = split_with_sam(img, items, progress=lambda p, m: progress(0.85 + 0.15 * p, m))
            items = split_large(img, items, min_area=1200.0)
        else:
            items = []
            for geom, v in rasterio.features.shapes(lab.astype(np.int32), mask=mask,
                                                    transform=img.ds.transform, connectivity=8):
                g = shape(geom)
                if not g.is_valid:
                    g = g.buffer(0)
                items.append({"geometry": g, "score": float(scores[int(v) - 1])})
        simplify = opt.simplify_px * (1.5 if stuff else 1.0)
        for it in postprocess(label, items, gsd, opt.min_area, simplify, opt.regularize):
            out.append({"label": label, "geometry": it["geometry"], "score": it["score"], "source": "auto"})
    return out


def _refine_buildings_with_sam(img: GeoImage, lab: np.ndarray, n: int, scores, progress: ProgressFn):
    """Sharpen each building blob from the land-cover map with a SAM 2 box prompt.

    SAM's mask is used only when it agrees with the land-cover blob (IoU >= 0.5), so a
    bad prompt can never make a footprint worse than the land-cover result."""
    objs = ndimage.find_objects(lab)
    T = 1024
    groups: dict[tuple, list[int]] = {}
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        cy = (sl[0].start + sl[0].stop) // 2; cx = (sl[1].start + sl[1].stop) // 2
        groups.setdefault((cy // (T // 2), cx // (T // 2)), []).append(i)
    items = []
    done = 0
    for (gy, gx), ids in groups.items():
        # window centred on this group cell, large enough to contain its buildings
        r0 = min(objs[i][0].start for i in ids); r1 = max(objs[i][0].stop for i in ids)
        c0 = min(objs[i][1].start for i in ids); c1 = max(objs[i][1].stop for i in ids)
        size = max(T, r1 - r0 + 64, c1 - c0 + 64)
        row0 = max(0, (r0 + r1) // 2 - size // 2); col0 = max(0, (c0 + c1) // 2 - size // 2)
        win = Window(col0, row0, size, size)
        rgb, valid = img.read_rgb(win)
        boxes = []
        for i in ids:
            sl = objs[i]
            pad = 3
            boxes.append([sl[1].start - col0 - pad, sl[0].start - row0 - pad,
                          sl[1].stop - col0 + pad, sl[0].stop - row0 + pad])
        masks, ious = models.sam2_boxes(rgb, np.array(boxes, np.float32), key=("bld", id(img), row0, col0, size))
        wt = img.ds.window_transform(win)
        for i, m, q in zip(ids, masks, ious):
            sl = objs[i]
            blob = np.zeros(m.shape, bool)
            rr = slice(sl[0].start - row0, sl[0].stop - row0); cc = slice(sl[1].start - col0, sl[1].stop - col0)
            blob[rr, cc] = lab[sl] == (i + 1)
            m = m & valid
            inter = (m & blob).sum(); union = (m | blob).sum()
            use = m if union and inter / union >= 0.5 else blob
            use = clean_mask(use, min_pixels=8)
            for p in mask_to_polygons(use, wt):
                items.append({"geometry": p, "score": float(scores[i]) * 0.5 + float(q) * 0.5})
        done += 1
        progress(done / len(groups), f"Refining building edges {done}/{len(groups)}")
    return items


def _detect_open_vocab(img: GeoImage, opt: DetectOptions, progress: ProgressFn) -> list[dict]:
    gsd = img.gsd
    # Instance objects are dropped on seams; overlap must exceed the largest object.
    n_tiles = img.tile_count(opt.tile_size, opt.overlap)
    raw: dict[str, list[dict]] = {c: [] for c in opt.classes}
    t0 = time.time()
    done = 0
    for tile in img.tiles(opt.tile_size, opt.overlap):
        for label in opt.classes:
            dets = models.detect(opt.engine, tile.image, label, opt.threshold)
            stuff = is_stuff(label)
            for d in dets:
                m = d["mask"] & tile.valid
                m = clean_mask(m, min_pixels=max(8, int(opt.min_area / (gsd * gsd))))
                if not m.any():
                    continue
                if not stuff and _edge_cut(m, tile.touches):
                    # keep it only if it would otherwise be lost (object bigger than overlap)
                    ys, xs = np.nonzero(m)
                    if (xs.max() - xs.min()) < opt.overlap * 0.9 and (ys.max() - ys.min()) < opt.overlap * 0.9:
                        continue
                for p in mask_to_polygons(m, tile.transform):
                    raw[label].append({"geometry": p, "score": d["score"]})
        done += 1
        eta = (time.time() - t0) / done * (n_tiles - done)
        progress(0.95 * done / n_tiles, f"Tile {done}/{n_tiles} ({', '.join(opt.classes)}) · ~{int(eta)} s left")

    progress(0.96, "Merging and cleaning polygons…")
    features = []
    for label, items in raw.items():
        for it in postprocess(label, items, gsd, opt.min_area, opt.simplify_px, opt.regularize):
            features.append({"label": label, "geometry": it["geometry"], "score": it["score"], "source": "auto"})
    return features


# ------------------------------------------------------------- interactive
def interactive_segment(img: GeoImage, image_key: str, points_xy=None, point_labels=None,
                        box_xy=None, window_px: int = 1024) -> tuple[list, float]:
    """Segment one object from clicks or a box given in source-CRS coordinates.

    Returns (polygons in source CRS, score)."""
    inv = ~img.ds.transform
    pts = [inv * (x, y) for x, y in (points_xy or [])]
    bx = None
    if box_xy is not None:
        (c0, r0), (c1, r1) = inv * (box_xy[0], box_xy[1]), inv * (box_xy[2], box_xy[3])
        bx = (min(c0, c1), min(r0, r1), max(c0, c1), max(r0, r1))

    # choose a window: the box with generous context, or a fixed window around the clicks
    if bx is not None:
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        size = int(max(512, min(4096, max(bw, bh) * 1.6)))
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    else:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        size = int(max(window_px, min(4096, spread * 2.5)))
        # anchor on the first click so follow-up clicks reuse the cached embedding
        cx, cy = pts[0]
        # snap window origin to a grid to raise cache hit rate
    step = size // 4
    col0 = int(round((cx - size / 2) / step) * step)
    row0 = int(round((cy - size / 2) / step) * step)
    win = Window(col0, row0, size, size)
    rgb, valid = img.read_rgb(win)
    key = (image_key, col0, row0, size)

    local_pts = [[p[0] - col0, p[1] - row0] for p in pts] or None
    local_box = None if bx is None else (bx[0] - col0, bx[1] - row0, bx[2] - col0, bx[3] - row0)
    mask, score = models.sam2_prompt(rgb, key, local_pts, point_labels if pts else None, local_box)
    mask &= valid
    if local_box is not None:  # constrain to (slightly expanded) box
        pad = 0.05 * max(local_box[2] - local_box[0], local_box[3] - local_box[1])
        c0, r0, c1, r1 = [int(v) for v in (local_box[0] - pad, local_box[1] - pad, local_box[2] + pad, local_box[3] + pad)]
        keep = np.zeros_like(mask); keep[max(r0, 0):r1, max(c0, 0):c1] = True
        mask &= keep
    mask = clean_mask(mask, min_pixels=8)
    polys = mask_to_polygons(mask, img.ds.window_transform(win))
    # keep the part(s) under positive clicks, or the largest part for a box
    if points_xy:
        pos = [Point(p) for p, l in zip(points_xy, point_labels) if l == 1]
        hit = [p for p in polys if any(p.buffer(img.gsd * 2).contains(q) for q in pos)]
        polys = hit or polys
    if polys and box_xy is not None:
        polys = [max(polys, key=lambda p: p.area)]
    return polys, score
