"""Convert Meta's SAM 3 checkpoint (sam3.pt, e.g. from the SAM3 .dlpk) to the transformers format.

The Esri package ships the *video* SAM 3 checkpoint: its keys start with "detector." (image model)
and "tracker." (video tracker). The stock converter expects image-model keys only and silently
produces a broken model (all scores ~0.5) if given the video checkpoint. This keeps the
"detector." part and strips the prefix.

needs the converter from the transformers source tree:
  src/transformers/models/sam3/convert_sam3_to_hf.py  (put its folder on PYTHONPATH)
usage: python convert_sam3.py sam3.pt out_dir
then:  set GEOAI_SAM3=out_dir
"""
import sys

import convert_sam3_to_hf as c

src, dst = sys.argv[1:3]
orig = c.load_original_state_dict
c.load_original_state_dict = lambda p: {k[len("detector."):]: v for k, v in orig(p).items() if k.startswith("detector.")} \
    or orig(p)
c.convert_sam3_checkpoint(src, dst)
