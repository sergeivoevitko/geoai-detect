"""AI model wrappers.

Engines for automatic detection (text prompt -> instance masks):
  * sam3        Meta SAM 3: native text-prompted "segment every <concept>".
                Most accurate. Gated on Hugging Face: accept the licence at
                https://huggingface.co/facebook/sam3 and set HF_TOKEN.
  * gdino_sam2  GroundingDINO (open-vocabulary boxes) + SAM 2.1 (box -> mask).
                Fully open weights, works out of the box.

Interactive (real-time) segmentation from clicks / boxes uses SAM 2.1.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

GDINO_ID = os.environ.get("GEOAI_GDINO", "IDEA-Research/grounding-dino-base")
SAM3_ID = os.environ.get("GEOAI_SAM3", "facebook/sam3")


def pick_device() -> str:
    if os.environ.get("GEOAI_DEVICE"):
        return os.environ["GEOAI_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
# large SAM is noticeably more precise on edges; small is ~4x faster on CPU
SAM2_ID = os.environ.get("GEOAI_SAM2", "facebook/sam2.1-hiera-large" if DEVICE == "cuda" else "facebook/sam2.1-hiera-small")
if DEVICE == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))

_lock = threading.RLock()   # models are not re-entrant; serialize inference


def _hf_kwargs():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"token": tok} if tok else {}


# --------------------------------------------------------------------- loaders
@lru_cache(maxsize=1)
def load_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(DEVICE).eval()
    return proc, model


@lru_cache(maxsize=1)
def load_sam2():
    from transformers import Sam2Processor, Sam2Model
    proc = Sam2Processor.from_pretrained(SAM2_ID)
    model = Sam2Model.from_pretrained(SAM2_ID).to(DEVICE).eval()
    return proc, model


@lru_cache(maxsize=1)
def load_sam3():
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained(SAM3_ID, **_hf_kwargs())
    model = Sam3Model.from_pretrained(SAM3_ID, **_hf_kwargs()).to(DEVICE).eval()
    return proc, model


def sam3_available() -> bool:
    """True if SAM 3 weights are cached locally or downloadable with the current token."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(SAM3_ID, "config.json", **_hf_kwargs())
        return True
    except Exception:
        return False


def engines() -> list[dict]:
    return [
        {"id": "hybrid", "name": "Hybrid: land-cover + SAM refine (recommended)", "available": True},
        {"id": "sam3", "name": "SAM 3 (text prompt, most accurate)", "available": sam3_available()},
        {"id": "gdino_sam2", "name": "GroundingDINO + SAM 2.1 (open)", "available": True},
    ]


# ---------------------------------------------------------------- detection
def prompt_for(label: str) -> str:
    """Aerial-view phrasing helps open-vocabulary models a lot."""
    l = label.strip().lower()
    aliases = {
        "building": "building roof", "buildings": "building roof", "house": "house roof",
        "tree": "tree", "trees": "tree", "road": "road", "roads": "road",
        "car": "car", "cars": "car", "solar panel": "solar panel",
        "pool": "swimming pool", "swimming pool": "swimming pool",
    }
    return aliases.get(l, l)


@torch.inference_mode()
def detect_gdino_sam2(rgb: np.ndarray, label: str, threshold: float = 0.3,
                      text_threshold: float = 0.25, max_box_frac: float = 0.8) -> list[dict]:
    """Returns [{"mask": HxW bool, "score": float}] for one class on one tile."""
    gproc, gmodel = load_gdino()
    sproc, smodel = load_sam2()
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    text = prompt_for(label) + "."
    with _lock:
        inputs = gproc(images=img, text=text, return_tensors="pt").to(DEVICE)
        out = gmodel(**inputs)
        res = gproc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[(H, W)])[0]
    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    if len(boxes) == 0:
        return []
    # GroundingDINO often adds one box around the whole tile: drop it
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = area < max_box_frac * H * W
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return []
    boxes, scores = _nms(boxes, scores, 0.7)

    masks, ious = sam2_boxes(rgb, boxes)
    out = []
    for m, s, iou in zip(masks, scores, ious):
        if m.sum() == 0:
            continue
        out.append({"mask": m, "score": float(s) * 0.5 + float(iou) * 0.5})
    return out


@torch.inference_mode()
def detect_sam3(rgb: np.ndarray, label: str, threshold: float = 0.3) -> list[dict]:
    proc, model = load_sam3()
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    with _lock:
        inputs = proc(images=img, text=prompt_for(label), return_tensors="pt").to(DEVICE)
        out = model(**inputs)
        res = proc.post_process_instance_segmentation(
            out, threshold=threshold, mask_threshold=0.5, target_sizes=[(H, W)])[0]
    masks = res["masks"].cpu().numpy().astype(bool)
    scores = res["scores"].cpu().numpy()
    return [{"mask": m, "score": float(s)} for m, s in zip(masks, scores) if m.any()]


def detect(engine: str, rgb: np.ndarray, label: str, threshold: float) -> list[dict]:
    if engine == "sam3":
        return detect_sam3(rgb, label, threshold)
    return detect_gdino_sam2(rgb, label, threshold)


# ----------------------------------------------------------- SAM 2 prompting
class EmbeddingCache:
    """Caches SAM image embeddings so repeated clicks on the same area are instant."""

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self.items: dict = {}

    def get(self, key, rgb: np.ndarray):
        if key in self.items:
            v = self.items.pop(key); self.items[key] = v
            return v
        sproc, smodel = load_sam2()
        inputs = sproc(images=Image.fromarray(rgb), return_tensors="pt").to(DEVICE)
        with _lock, torch.inference_mode():
            emb = smodel.get_image_embeddings(inputs["pixel_values"])
        v = (emb, inputs["original_sizes"])
        self.items[key] = v
        if len(self.items) > self.maxsize:
            self.items.pop(next(iter(self.items)))
        return v


_cache = EmbeddingCache()


@torch.inference_mode()
def sam2_boxes(rgb: np.ndarray, boxes: np.ndarray, key=None, batch: int = 32):
    """Segment each box. Returns (masks NxHxW bool, iou scores N)."""
    sproc, smodel = load_sam2()
    emb, orig = _cache.get(key or ("tile", id(rgb), rgb.shape, int(rgb[::64, ::64].sum())), rgb)
    all_masks, all_ious = [], []
    for i in range(0, len(boxes), batch):
        b = boxes[i:i + batch].tolist()
        inputs = sproc(input_boxes=[b], original_sizes=orig.tolist(),
                       return_tensors="pt")
        with _lock:
            out = smodel(input_boxes=inputs["input_boxes"].to(DEVICE), image_embeddings=emb,
                         multimask_output=False)
        m = sproc.post_process_masks(out.pred_masks.cpu(), orig.cpu())[0]  # (n,1,H,W)
        all_masks.append(m[:, 0].numpy().astype(bool))
        all_ious.append(out.iou_scores[0, :, 0].cpu().numpy())
    return np.concatenate(all_masks), np.concatenate(all_ious)


@torch.inference_mode()
def sam2_prompt(rgb: np.ndarray, key, points=None, labels=None, box=None) -> tuple[np.ndarray, float]:
    """Interactive segmentation on a window. points: [[x, y], ...] pixel coords in rgb."""
    sproc, smodel = load_sam2()
    emb, orig = _cache.get(key, rgb)
    kw = {}
    if points:
        kw["input_points"] = [[points]]
        kw["input_labels"] = [[labels]]
    if box is not None:
        kw["input_boxes"] = [[list(box)]]
    inputs = sproc(original_sizes=orig.tolist(), return_tensors="pt", **kw)
    multi = box is None and points is not None and len(points) == 1
    fwd = {k: inputs[k].to(DEVICE) for k in ("input_points", "input_labels", "input_boxes") if k in inputs}
    with _lock:
        out = smodel(image_embeddings=emb, multimask_output=multi, **fwd)
    masks = sproc.post_process_masks(out.pred_masks.cpu(), orig.cpu())[0][0]  # (k,H,W)
    ious = out.iou_scores[0, 0].cpu().numpy()
    best = int(np.argmax(ious))
    return masks[best].numpy().astype(bool), float(ious[best])


def _nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][iou < thr]
    return boxes[keep], scores[keep]
