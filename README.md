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

## Roads (designed for 10-30 cm city imagery)

* **Network**: SAM-Road predicts connected centerlines -> `road_centerline.shp`
* **Width**: measured per segment (75th percentile, so trees/shadows that hide part of the road
  do not make it narrower). "Road" pixels may extend up to 22 m from a centerline (multi-lane
  roads), "pavement" pixels only 6 m (keeps parking lots and plazas out).
* **Junctions / gaps**: the surface grows through connected road and shadow pixels up to 60 m, but a
  grown area is kept only if it bridges two or more road parts (fills big intersections and shadows
  lying across a street, rejects building shadows beside a road). Facing road ends are joined
  across paved, tree-covered or shadowed gaps up to 70 m (`kind = gap_fill`).
* **Occlusion**: each road is continued at its measured width under tree canopy and cast shadows,
  "as if the tree were not there". `occluded` = share of the segment that was hidden.
* **Roundabouts**: small round loops in the network are detected (`kind = roundabout`) and the
  centre island is cut out as a hole.
* **Agreement check**: every centerline gets `agree` (0-1) from three independent signals (SAM-Road
  probability, land-cover under the line, width stability). Segments below 0.6 get `review = 1`;
  the web app has a *needs review* list that jumps to each one (✓ ok / delete).

### Centerlines, carriageway and sidewalks

* Centerlines are traced along the middle of the detected road surface (medial axis, side spurs
  pruned), so roundabouts get a ring around the island and divided roads get one line per carriageway.
* `road.shp` is the **carriageway** (driving surface incl. parked cars); `sidewalk.shp` holds the
  paved strips (up to 4.5 m) along it. The split uses the land-cover model's "road" vs
  "developed space" classes. Parking lanes are sometimes labelled sidewalk.

### Parking lots, medians, wide boulevards

* `parking.shp`: paved areas densely filled with parked cars (car-density map from the image),
  at least ~12 m wide and away from through-roads; also seeded by clusters of short "aisle" lines.
  Parking areas are removed from the road surface and get no centerlines.
* Planted medians: road growth may not enter areas that are mostly vegetation or bare ground, and
  network lines running inside a median next to a real road are dropped.
* Wide roads: the surface is smoothed and thin painted markings filled before tracing, side branches
  shorter than ~2x the road half-width are pruned, and near-duplicate parallel lines are removed.

## Large building complexes

Footprints over 1200 m² are split with a watershed on the image gradient; a split is kept only
where the photo shows a real roof edge between the parts.

## Solar panels

Class `solar panel` (shown in blue). Searched on detected roofs as mid-dark, colour-neutral,
rectangular patches (solar water-heater collectors and PV modules); roof shadows are rejected by
colour (they are strongly blue) and parapet shadows by shape. Open-vocabulary detectors boxed whole
roofs at 15 cm, so they are not used for this class.

## Local AI verification (optional, no internet needed)

`python cli.py image.tif --classes road --verify` or the *AI-verify flagged objects* button.
A local vision-language model looks at each flagged object and says what lies under it
(road / roof / grass / trees / parking / water). Matches clear the flag; mismatches stay flagged with
`ai_verdict = WRONG` and `ai_note`. Default model: Qwen3-VL-2B (Apache-2.0, ~4 GB, ~4-10 s per check on
CPU). Alternatives: `GEOAI_VLM=Qwen/Qwen3-VL-4B-Instruct` (better, GPU recommended), or Ollama
(`GEOAI_VLM_BACKEND=ollama GEOAI_VLM=qwen2.5vl:3b`, also `llama3.2-vision`, `gemma3`).
Small models are a second opinion for gross errors, not a judge of exact edges.

## CPU or GPU

Runs on CPU by default. With an NVIDIA GPU, install the CUDA build of PyTorch (below) and it is used
automatically, or choose explicitly: `--device cpu|cuda|mps` (CLI) / `GEOAI_DEVICE=cuda` (server).

## Imagery sources

* GeoTIFF upload (any CRS; outputs stay in that CRS)
* **WMTS** GetCapabilities URL (e.g. a commercial orthophoto service in ITM EPSG:2039 — tiles are
  mosaicked in the service's own CRS, so Shapefiles come out in ITM)
* XYZ tiles, WMS
Service URLs that contain a personal key are stored only in your browser, never in the repo.

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
