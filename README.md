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

Class `solar panel` (shown in blue). Two detectors, merged:

* **Rules**: mid-dark, colour-neutral rectangles on detected roofs. Finds the small solar
  water-heater collectors on Israeli roofs; roof and parapet shadows are rejected by colour/shape.
* **Trained model** (`finloop/yolov8s-seg-solar-panels`, MIT weights, needs the optional
  `ultralytics` package, AGPL-3.0): finds PV arrays, panels on red-tile roofs and large ground-mounted
  solar fields (tested on the Ketura field: ~21 ha found, no false hits on the date orchards next to it).
  The image is resampled to ~20 cm/px for it. Note: its weights file is a pickle that runs code on
  load; only use weights from a source you trust.

A local vision-language check of solar candidates was tested and rejected: Qwen3-VL 2B/4B answered
"solar" for almost every dark rectangle on a roof.

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

## Install and run on Windows

1. Install **Python 3.12** from https://www.python.org/downloads/ (tick *Add python.exe to PATH*).
2. Download this repository (green *Code* button → *Download ZIP*, or `git clone`) and unzip it.
3. Double-click **`install.bat`** (one time, ~10 minutes). With an NVIDIA GPU, see the note inside it.
4. Double-click **`GeoAI Detect.bat`**. The work area opens in its own window.
   (`GeoAI Detect (console).bat` keeps a log window open; `GeoAI Detect (browser).bat` opens the work area in
   your web browser instead of its own window. Engine messages are written to `workspace\\server.log`.)

The AI models (about 6 GB) download automatically the first time a detection runs.
The window uses Microsoft Edge WebView2, which is built into Windows 10 and 11.

### The work area

| Step | How |
|------|-----|
| **1. Imagery** | **📂 Open image**: any georeferenced GeoTIFF / JP2 / IMG / VRT, opened in place (no copy, any size). **🌐 Add service**: WMTS GetCapabilities URL (e.g. Ofek), WMS or XYZ. Services are shown live on the map; only the rectangle is downloaded, at the resolution you choose (10-50 cm). Several sources can be listed; the radio button picks the one used for detection, checkbox/slider control visibility. |
| **2. Rectangle** | **⬚ Rectangle** tool (R), drag on the map. The panel shows area, megapixels and an estimated run time. |
| **3. Detect** | Tick what to detect: Buildings, Roads (with Centerlines, Sidewalks, Parking lots), Trees, Solar panels, or type any other object. **▶ Run detection**; progress and Cancel are shown. Results are cut to the rectangle (buildings/panels kept whole). |
| **4. Review** | Click an object for its attributes. **🎯 Click-add** (P) / **▣ Box-add** (B) add a missed object with real-time AI, **🗑 Delete** (D) removes one. Every run is listed and can be zoomed to or deleted. |
| **5. Export** | **⬇ Export**: File Geodatabase (.gdb, one feature class per class) and/or Shapefiles, in the imagery's CRS, ITM (EPSG:2039), WGS84 or any EPSG code; choose layers and the output folder. |

Results, sources and the rectangle are kept between sessions (folder `workspace/`).
Service URLs with personal keys stay on your computer only.

### Other ways to run

```bash
python run_app.py --browser          # work area in the web browser instead of a window
python run_app.py --device cuda      # force GPU (default: auto)
python cli.py image.tif --classes building road tree "solar panel" --out results/   # batch, no GUI
```

On Linux/macOS: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python run_app.py`.

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
