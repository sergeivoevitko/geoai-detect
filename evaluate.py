#!/usr/bin/env python
"""Measure detection accuracy against reference polygons.

  python evaluate.py predicted.shp reference.geojson [--image image.tif]

Reports
  * pixel/area IoU, precision, recall, F1  (how well the covered area matches)
  * object-level precision / recall / F1   (a predicted object counts as correct
    if it overlaps a reference object with IoU >= 0.5)
"""
import argparse

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import box


def evaluate(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, extent=None, iou_thr=0.5) -> dict:
    ref = ref.to_crs(pred.crs) if ref.crs != pred.crs else ref
    if extent is not None:
        ref = ref[ref.intersects(extent)].copy(); ref["geometry"] = ref.intersection(extent)
        pred = pred[pred.intersects(extent)].copy(); pred["geometry"] = pred.intersection(extent)
    ref = ref[~ref.is_empty & (ref.area > 0)]
    pred = pred[~pred.is_empty & (pred.area > 0)]
    P = unary_union(pred.geometry.buffer(0).values); R = unary_union(ref.geometry.buffer(0).values)
    inter = P.intersection(R).area
    area = {
        "iou": inter / max(P.union(R).area, 1e-9),
        "precision": inter / max(P.area, 1e-9),
        "recall": inter / max(R.area, 1e-9),
    }
    area["f1"] = 2 * area["precision"] * area["recall"] / max(area["precision"] + area["recall"], 1e-9)

    # object matching (greedy by IoU)
    sidx = ref.sindex
    matched_ref = set(); tp = 0
    ious = []
    for g in pred.geometry:
        best, bj = 0.0, None
        for j in sidx.query(g, predicate="intersects"):
            if j in matched_ref:
                continue
            r = ref.geometry.iloc[j]
            i = g.intersection(r).area
            u = g.union(r).area
            if u and i / u > best:
                best, bj = i / u, j
        if bj is not None and best >= iou_thr:
            matched_ref.add(bj); tp += 1; ious.append(best)
    obj = {"tp": tp, "n_pred": len(pred), "n_ref": len(ref),
           "precision": tp / max(len(pred), 1), "recall": tp / max(len(ref), 1),
           "mean_iou_matched": float(np.mean(ious)) if ious else 0.0}
    obj["f1"] = 2 * obj["precision"] * obj["recall"] / max(obj["precision"] + obj["recall"], 1e-9)
    return {"area": area, "object": obj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("ref")
    ap.add_argument("--image", help="restrict to this image's footprint")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    pred = gpd.read_file(a.pred); ref = gpd.read_file(a.ref)
    extent = None
    if a.image:
        import rasterio
        with rasterio.open(a.image) as d:
            extent = box(*d.bounds)
            pred = pred.to_crs(d.crs); ref = ref.to_crs(d.crs)
    r = evaluate(pred, ref, extent, a.iou)
    ar, ob = r["area"], r["object"]
    print(f"Area   IoU {ar['iou']:.3f}  P {ar['precision']:.3f}  R {ar['recall']:.3f}  F1 {ar['f1']:.3f}")
    print(f"Object P {ob['precision']:.3f}  R {ob['recall']:.3f}  F1 {ob['f1']:.3f}  "
          f"({ob['tp']} matched / {ob['n_pred']} predicted / {ob['n_ref']} reference, "
          f"mean IoU of matches {ob['mean_iou_matched']:.3f})")


if __name__ == "__main__":
    main()
