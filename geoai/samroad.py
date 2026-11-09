"""SAM-Road: road network graph extraction with a fine-tuned Segment Anything ViT-B.

Upstream: https://github.com/htcr/sam_road (MIT, CVPRW 2024 best paper).
Weights:  https://huggingface.co/congrui/sam_road (trained on City-Scale, 1 m/px, US cities).

Produces, for a georeferenced raster:
  * road surface probability  -> road polygons
  * road graph (nodes + edges) -> road centerlines (LineStrings)
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from rasterio.transform import Affine
from scipy.spatial import KDTree
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

from .models import DEVICE, _lock

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "sam_road")
REPO_ID = os.environ.get("GEOAI_SAMROAD", "congrui/sam_road")
CKPT = "cityscale_vitb_512_e10.ckpt"
TRAIN_GSD = 1.0   # m/px of City-Scale


@lru_cache(maxsize=1)
def load_samroad():
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    import yaml
    from addict import Dict
    from huggingface_hub import hf_hub_download
    from model import SAMRoad   # vendored

    with open(os.path.join(_VENDOR, "config", "toponet_vitb_512_cityscale.yaml")) as f:
        cfg = Dict(yaml.safe_load(f))
    # The original SAM checkpoint is only used to initialise weights before the
    # fine-tuned checkpoint overwrites all of them, so an empty stand-in is enough.
    empty = os.path.join(tempfile.gettempdir(), "geoai_empty_sam.pth")
    if not os.path.exists(empty):
        torch.save({}, empty)
    cfg.SAM_CKPT_PATH = empty
    # nothing to resize in the empty stand-in; the trained checkpoint holds the
    # already-resized positional embeddings
    SAMRoad.resize_sam_pos_embed = lambda self, sd, *a, **k: sd
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):   # upstream prints every parameter name
        net = SAMRoad(cfg)
    # weights_only=True: refuse to unpickle anything but tensors/plain containers, so a
    # tampered checkpoint cannot execute code on load.
    ck = torch.load(hf_hub_download(REPO_ID, CKPT), map_location="cpu", weights_only=True)
    net.load_state_dict(ck["state_dict"], strict=True)
    return net.to(DEVICE).eval(), cfg


def _patch_starts(n: int, patch: int, stride: int) -> list[int]:
    if n <= patch:
        return [0]
    s = list(range(0, n - patch, stride)) + [n - patch]
    return sorted(set(s))


def _nms(points, scores, radius):
    if len(points) == 0:
        return points
    order = np.argsort(scores)[::-1]
    pts = points[order]
    keep = []
    tree = KDTree(pts)
    removed = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if removed[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(pts[i], radius):
            if j != i:
                removed[j] = True
    return pts[keep]


@torch.inference_mode()
def infer(rgb: np.ndarray, batch: int = 8, progress=None):
    """rgb: HxWx3 uint8 at ~1 m/px. Returns (road_prob HxW float, nodes Nx2 (x,y), edges Mx2, scores M)."""
    net, cfg = load_samroad()
    P = cfg.PATCH_SIZE
    H0, W0 = rgb.shape[:2]
    # pad to at least one patch
    H, W = max(H0, P), max(W0, P)
    img = np.zeros((H, W, 3), np.uint8); img[:H0, :W0] = rgb
    stride = P // 2
    patches = [(x, y) for y in _patch_starts(H, P, stride) for x in _patch_starts(W, P, stride)]
    t = torch.from_numpy(img).float()
    kp = torch.zeros((H, W)); road = torch.zeros((H, W)); cnt = torch.zeros((H, W))
    feats = []
    for i in range(0, len(patches), batch):
        bp = patches[i:i + batch]
        x = torch.stack([t[y:y + P, x0:x0 + P] for x0, y in bp]).to(DEVICE)
        with _lock:
            scores, emb = net.infer_masks_and_img_features(x)
        scores = scores.cpu()
        feats.append((bp, emb))
        for k, (x0, y) in enumerate(bp):
            kp[y:y + P, x0:x0 + P] += scores[k, :, :, 0]
            road[y:y + P, x0:x0 + P] += scores[k, :, :, 1]
            cnt[y:y + P, x0:x0 + P] += 1
        if progress:
            progress(0.7 * min(1.0, (i + batch) / len(patches)), "SAM-Road: road masks")
    kp = (kp / cnt.clamp(min=1)).numpy(); road = (road / cnt.clamp(min=1)).numpy()

    # graph vertices: intersections first, then points sampled along roads
    def pts(mask, thr):
        rc = np.column_stack(np.where(mask > thr)); return rc[:, ::-1].astype(np.float32), mask[mask > thr]
    p0, s0 = pts(kp, cfg.ITSC_THRESHOLD); k0 = _nms(p0, s0, cfg.ITSC_NMS_RADIUS)
    p1, s1 = pts(road, cfg.ROAD_THRESHOLD); k1 = _nms(p1, s1, cfg.ROAD_NMS_RADIUS)
    cand = np.concatenate([k0, k1]) if len(k0) or len(k1) else np.zeros((0, 2), np.float32)
    prio = np.concatenate([np.ones(len(k0)), np.zeros(len(k1))])
    nodes = _nms(cand, prio, cfg.ROAD_NMS_RADIUS) if len(cand) else cand
    if len(nodes) < 2:
        return road[:H0, :W0], nodes, np.zeros((0, 2), int), np.zeros(0)

    # topology: which neighbouring vertices are connected by a road
    edge_sum = defaultdict(float); edge_n = defaultdict(float)
    for bi, (bp, emb) in enumerate(feats):
        data = {"points": [], "pairs": [], "valid": []}; maps = []
        for (x0, y) in bp:
            inside = np.nonzero((nodes[:, 0] >= x0) & (nodes[:, 0] < x0 + P) &
                                (nodes[:, 1] >= y) & (nodes[:, 1] < y + P))[0]
            pp = nodes[inside] - np.array([[x0, y]], np.float32)
            n = len(pp)
            if n:
                d, idx = KDTree(pp).query(pp, k=cfg.MAX_NEIGHBOR_QUERIES + 1, distance_upper_bound=cfg.NEIGHBOR_RADIUS)
                idx = idx[:, 1:]
            else:
                idx = np.zeros((0, cfg.MAX_NEIGHBOR_QUERIES), int)
            src = np.tile(np.arange(n)[:, None], (1, cfg.MAX_NEIGHBOR_QUERIES))
            valid = idx < n
            tgt = np.where(valid, idx, src)
            data["points"].append(pp); data["pairs"].append(np.stack([src, tgt], -1)); data["valid"].append(valid)
            maps.append(inside)
        L = max(x.shape[0] for x in data["points"])
        if L == 0:
            continue
        col = {k: np.stack([np.pad(x, [(0, L - x.shape[0])] + [(0, 0)] * (x.ndim - 1)) for x in v])
               for k, v in data.items()}
        with _lock:
            ts = net.infer_toponet(emb, torch.tensor(col["points"], device=DEVICE),
                                   torch.tensor(col["pairs"], device=DEVICE),
                                   torch.tensor(col["valid"], device=DEVICE))
        ts = torch.nan_to_num(ts, nan=-100.0).squeeze(-1).cpu().numpy()
        b_i, s_i, p_i = np.nonzero(col["valid"])
        for b, s, p in zip(b_i, s_i, p_i):
            a, c = col["pairs"][b, s, p]
            e = (int(maps[b][a]), int(maps[b][c]))
            edge_sum[e] += float(ts[b, s, p]); edge_n[e] += 1
        if progress:
            progress(0.7 + 0.3 * (bi + 1) / len(feats), "SAM-Road: road graph")
    und = defaultdict(list)
    for e, s in edge_sum.items():
        und[tuple(sorted(e))].append(s / edge_n[e])
    edges, escores = [], []
    for e, ss in und.items():
        sc = float(np.mean(ss))
        if sc > cfg.TOPO_THRESHOLD and e[0] != e[1]:
            edges.append(e); escores.append(sc)
    keep = (nodes[:, 0] < W0) & (nodes[:, 1] < H0)
    return road[:H0, :W0], nodes, np.array(edges, int).reshape(-1, 2), np.array(escores)


def graph_to_lines(nodes: np.ndarray, edges: np.ndarray, transform: Affine) -> list[LineString]:
    """Edges in pixel space -> merged centerline LineStrings in the raster's CRS."""
    if len(edges) == 0:
        return []
    segs = []
    for a, b in edges:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        segs.append(LineString([transform * (x0 + 0.5, y0 + 0.5), transform * (x1 + 0.5, y1 + 0.5)]))
    merged = linemerge(unary_union(segs))
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]
