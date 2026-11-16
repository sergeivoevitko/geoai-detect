Vendored network code from Tile2Net (https://github.com/VIDA-NYU/tile2net, BSD-3-Clause, see LICENSE),
files `src/tile2net/tileseg/network/{hrnetv2,mynn,ocr_utils,ocrnet,utils}.py` and
`src/tile2net/tileseg/utils/attr_dict.py` at the commit in UPSTREAM_COMMIT.
Only the import lines were changed (package-relative); `_shim.py` replaces Tile2Net's config/logger.
Weights (`satellite_2021.pth`) are downloaded at first use from Tile2Net's figshare record
(DOI 10.6084/m9.figshare.33315570.v1) and checked against its SHA-256.
