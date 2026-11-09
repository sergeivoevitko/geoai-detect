Sample imagery used for testing (not committed, download with):

    for f in uc_berkeley.tif naip_test.tif trees_brazil.tif whu_building_test.tif sam_demo_image.tif; do
      curl -L -o sample_data/$f https://huggingface.co/datasets/giswqs/geospatial/resolve/main/$f
    done

`naip_test_buildings.geojson` is the reference building layer for `naip_test.tif`
(use with `python evaluate.py`).
