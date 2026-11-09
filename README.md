# GeoAI Detect

Detect objects (buildings, roads, trees, water, cars, anything you can name) in
georeferenced aerial imagery with AI, review and correct them on a map in real
time, and export **one Shapefile per class in the exact CRS of the source image**.

## How it works

```
GeoTIFF / XYZ / WMS ──► tiles ──► AI ──► masks ──► polygons (source CRS) ──► merge & clean ──► .shp
                                   │
                                   ├─ Roads: SAM-Road (fine-tuned Segment Anything, CVPRW 2024) predicts
                                   │   the connected road network -> road_centerline.shp; road polygons =
                                   │   land-cover road/pavement pixels along those centerlines
                                   │
                                   ├─ Land-cover model (UperNet-Swin, OpenEarthMap): road, tree,
                                   │   building, water, grass, farmland, bare soil, pavement
                                   │   · resampled to the model's training resolution (~0.4 m)
                                   │   · sliding window + 4-way flip test-time augmentation
                                   │   · buildings: every footprint is re-segmented with SAM 2.1
                                   │     for sharp edges, then squared up
                                   │
                                   └─ Open-vocabulary (any other class, e.g. "car", "solar panel"):
                                       SAM 3 text prompts (if you have access), otherwise
                                       GroundingDINO boxes -> SAM 2.1 masks
```

**Real-time AI refinement** (web app): click an object or drag a box and SAM 2.1
segments it in about a second (much faster on a GPU). Shift+click excludes areas.
Delete wrong detections with one click. Edits are saved and included in the export.

## Install

Requires Python 3.10+. An NVIDIA GPU is strongly recommended (10-50x faster), but CPU works.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# GPU (CUDA 12.x): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Model weights (about 2.5 GB) download automatically from Hugging Face on first use.

### Optional: SAM 3 (most accurate open-vocabulary model)

1. Request access at https://huggingface.co/facebook/sam3 (free; Meta reviews requests).
2. Create a read token at https://huggingface.co/settings/tokens
3. `export HF_TOKEN=hf_...` before starting the app. It is then used automatically for
   classes the land-cover model does not cover, and offered as an engine in the UI.

## Use

### Web app

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000:

1. **Imagery**: upload a GeoTIFF, or pan the map and fetch from an XYZ/WMS service.
2. **Detect**: choose classes (add your own), then press *Detect objects*.
3. **Refine**: *Click* tool (P), *Box* tool (B), *Delete* tool (S). Enter accepts, Esc clears.
4. **Export**: downloads a .zip with `building.shp`, `road.shp`, `tree.shp`, …

### Command line

```bash
python cli.py image.tif --classes building road tree --out results/
python cli.py image.tif --classes car "solar panel" --engine gdino_sam2 --threshold 0.35
python cli.py --xyz "https://server/{z}/{x}/{y}.jpg" --bbox W S E N --gsd 0.3 --classes building
```

### Measure accuracy

```bash
python evaluate.py results/image_building.shp my_reference_buildings.shp --image image.tif
```

Reports area IoU / precision / recall / F1 and object-level F1 (IoU >= 0.5 match).

## Third-party code

`third_party/sam_road` is vendored from https://github.com/htcr/sam_road (MIT) and its SAM
submodule https://github.com/htcr/segment-anything-road (Apache-2.0). Weights:
https://huggingface.co/congrui/sam_road (trained on 1 m/px US imagery; images are resampled to 1 m
for it). Checkpoints are loaded with `torch.load(weights_only=True)`.

## Shapefile attributes

| field   | meaning                                               |
|---------|-------------------------------------------------------|
| id      | sequential id                                         |
| class   | class name                                            |
| score   | model confidence 0-1                                  |
| source  | `auto` (detected) or `manual` (added in the web app)  |
| area_m2 | area in m² (computed in UTM if the image is in degrees)|
| perim_m | perimeter in m                                        |

## Tips for accuracy

* Best results at 0.1-0.5 m/px imagery. Images are automatically resampled for the land-cover model.
* Use 3-band RGB. For 4+ band images, bands 1,2,3 are used as RGB.
* Lower the confidence threshold to find more objects, raise it for fewer false positives.
* Turn off *Square-up building footprints* for irregular buildings.
* Use the click/box tools on what the automatic pass missed; it is the fastest way to a clean map.

## Configuration (environment variables)

| variable          | default                                                   |
|-------------------|-----------------------------------------------------------|
| `GEOAI_DEVICE`    | auto (`cuda` > `mps` > `cpu`)                             |
| `GEOAI_SAM2`      | `facebook/sam2.1-hiera-large` on GPU, `-small` on CPU     |
| `GEOAI_GDINO`     | `IDEA-Research/grounding-dino-base`                       |
| `GEOAI_LANDCOVER` | `kishan4444/upernet-swin-openearthmap-road-segmentation`  |
| `HF_TOKEN`        | Hugging Face token (needed for SAM 3 only)                |
