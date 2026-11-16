# Road pipeline prototypes (not wired into the app yet)

Scratch scripts used to test the planned road/sidewalk/building post-processing on
Jaffa, Ramat Aviv, Old North (Tel Aviv municipal orthophoto 2025, 15 cm) and Modi'in
(Antiquities Authority orthophoto, 33 cm). Inputs are the app's own shapefile outputs.

Planned road sequence, in order:

| step | script | what it does |
|---|---|---|
| 1 fusion | `fuse.py` | Tile2Net road + SAM 3 "road" fill-ins (only where not building/parking and SAM-Road agrees) + SAM-Road centerline fill; centerlines traced from the fused road; SAM 3 parking minus road |
| 2 network cleanup | `net.py` | prune stubs, close gaps <= 25 m, one node per junction, merge chains, straight/curve/roundabout-circle geometry, attributes (width, lanes, class, source), even road edges |
| 3 under-tree fill | `occl.py` | centerline pieces over tree canopy that bridge road pieces become road (`occluded`) |
| 4 curb line | `resolve.py` | road and sidewalk share one edge, no overlap |
| 5 sidewalk completion | `swfill.py` | fill gaps on street sides that already have >= 40 % detected sidewalk; never in medians |
| 6 islands | `islands.py` | paved islands (Tile2Net saw sidewalk) -> sidewalk; others left out of the road |
| 7 smoothing | `smooth.py` (method A) | outline smoothing; `ortho.py` squares building footprints |

`park_run.py` / `maskrcnn_test.py`: plain-PyTorch rebuilds used to test Esri weights (weights not included).
`viz/`: comparison image renderers.
