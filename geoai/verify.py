"""Local AI verification of detections with a vision-language model (VLM).

For each feature flagged for review, the image around it is cropped and shown to a
VLM twice side by side: left the plain photo, right the same photo with the detection
outlined. The model names what lies under the highlight (road, roof, grass, ...); the
detection is CORRECT if that matches its class, otherwise WRONG (kept flagged for review).

Backends (both fully local, nothing leaves the machine):
  * transformers  default model Qwen/Qwen3-VL-2B-Instruct (Apache-2.0, ~4 GB, CPU-friendly);
                  set GEOAI_VLM=Qwen/Qwen3-VL-4B-Instruct on a GPU for better judgement.
  * ollama        any vision model you have pulled, e.g. `ollama pull qwen2.5vl:3b`,
                  `llama3.2-vision`, `gemma3:4b`; set GEOAI_VLM_BACKEND=ollama, GEOAI_VLM=<name>.

Small VLMs are good at "is this a road at all / is a big part missing / is a parking lot
included", and weak at judging edges to the metre. Treat verdicts as a second opinion.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from rasterio.windows import from_bounds
from shapely.geometry import box

from .models import DEVICE, _lock

BACKEND = os.environ.get("GEOAI_VLM_BACKEND", "transformers")
VLM_ID = os.environ.get("GEOAI_VLM", "Qwen/Qwen3-VL-2B-Instruct" if BACKEND == "transformers" else "qwen2.5vl:3b")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CROP_PX = 448

# A classification question works far better than asking a small VLM to grade a detection:
# in tests Qwen3-VL-2B answered "PARTIAL" to everything when asked CORRECT/PARTIAL/WRONG, but
# named the surface correctly (road vs roof vs grass) when asked what lies under the highlight.
CHOICES = ["ROAD", "ROOF", "GRASS", "TREES", "PARKING", "WATER", "OTHER"]
PROMPT = ("Aerial photo. Look only at the area highlighted in magenta in the right half. "
          "What is mostly under the magenta area? Answer with one word: " + ", ".join(CHOICES) + ".")
EXPECTED = {"road": {"ROAD"}, "building": {"ROOF"}, "tree": {"TREES"}, "water": {"WATER"},
            "grass": {"GRASS"}, "parking lot": {"PARKING", "ROAD"}}


@lru_cache(maxsize=1)
def _load_hf():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(VLM_ID)
    dtype = torch.bfloat16 if DEVICE in ("cuda", "cpu") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=dtype).to(DEVICE).eval()
    return proc, model


def ask(image: Image.Image, question: str, max_new_tokens: int = 48) -> str:
    if BACKEND == "ollama":
        import requests
        buf = io.BytesIO(); image.save(buf, "PNG")
        r = requests.post(f"{OLLAMA_URL}/api/generate", timeout=600, json={
            "model": VLM_ID, "prompt": question, "stream": False,
            "images": [base64.b64encode(buf.getvalue()).decode()], "options": {"temperature": 0}})
        r.raise_for_status()
        return r.json().get("response", "")
    import torch
    proc, model = _load_hf()
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(DEVICE)
    with _lock, torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse(text: str) -> str:
    m = re.search(r"\b(" + "|".join(CHOICES) + r")\b", text.upper())
    return m.group(1) if m else "OTHER"


def render_pair(img, geom_crs, highlight_geoms, pad_m: float = 15.0) -> Image.Image:
    """Left: plain crop. Right: same crop with highlight_geoms filled magenta."""
    minx, miny, maxx, maxy = geom_crs.buffer(pad_m).bounds
    side = max(maxx - minx, maxy - miny)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    bb = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    win = from_bounds(*bb, transform=img.ds.transform).round_offsets().round_lengths()
    rgb, _ = img.read_rgb(win)
    base = Image.fromarray(rgb).resize((CROP_PX, CROP_PX), Image.BILINEAR)
    over = base.copy().convert("RGBA")
    layer = Image.new("RGBA", over.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    wt = img.ds.window_transform(win)
    sx, sy = CROP_PX / max(1, win.width), CROP_PX / max(1, win.height)
    inv = ~wt
    clip = box(*bb)
    for g in highlight_geoms:
        g = g.intersection(clip)
        polys = [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
        for p in polys:
            ring = [((inv * c)[0] * sx, (inv * c)[1] * sy) for c in p.exterior.coords]
            holes = [[((inv * c)[0] * sx, (inv * c)[1] * sy) for c in h.coords] for h in p.interiors]
            d.polygon(ring, fill=(255, 0, 200, 110), outline=(255, 0, 200, 255))
            for h in holes:
                d.polygon(h, fill=(0, 0, 0, 0))
    over = Image.alpha_composite(over, layer).convert("RGB")
    pair = Image.new("RGB", (CROP_PX * 2 + 8, CROP_PX), (255, 255, 255))
    pair.paste(base, (0, 0)); pair.paste(over, (CROP_PX + 8, 0))
    return pair


def verify_features(img, features: list[dict], max_items: int = 50, only_flagged: bool = True,
                    progress=None) -> int:
    """Adds attrs ai_verdict / ai_note to features in place. Returns number checked.

    Road centerlines are judged together with the road polygon around them."""
    polys_by_class: dict[str, list] = {}
    for f in features:
        if f["geometry"].geom_type in ("Polygon", "MultiPolygon"):
            polys_by_class.setdefault(f["label"], []).append(f["geometry"])
    todo = [f for f in features if (not only_flagged or (f.get("attrs") or {}).get("review") == 1)]
    todo = todo[:max_items]
    for i, f in enumerate(todo):
        label = f["label"].replace("_centerline", "")
        g = f["geometry"]
        if g.geom_type in ("LineString", "MultiLineString"):
            hl = [p for p in polys_by_class.get(label, []) if p.intersects(g.buffer(30))]
            hl = hl or [g.buffer(3)]
        else:
            hl = [g]
        pair = render_pair(img, g, hl)
        seen = parse(ask(pair, PROMPT, max_new_tokens=6))
        expected = EXPECTED.get(label, {label.upper()})
        verdict = "CORRECT" if seen in expected else "WRONG"
        a = f.setdefault("attrs", {})
        a["ai_verdict"] = verdict
        a["ai_note"] = f"AI sees {seen.lower()}"
        if verdict == "CORRECT":
            a["review"] = 0
        if progress:
            progress((i + 1) / len(todo), f"AI verification {i + 1}/{len(todo)}: {verdict}")
    return len(todo)
