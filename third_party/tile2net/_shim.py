"""Minimal stand-ins for the parts of Tile2Net's config/logger that the network code reads.

Values are Tile2Net's own defaults (src/tile2net/tileseg/config.py) for inference with
satellite_2021.pth (arch ocrnet.HRNet_Mscale, 4 classes). Not part of upstream."""
import logging

import torch

from .attr_dict import AttrDict

logger = logging.getLogger("tile2net")


def fmt_scale(prefix, scale):
    return f"{prefix}_{float(scale)}x"


cfg = AttrDict()
cfg.OPTIONS = AttrDict(); cfg.OPTIONS.TORCH_VERSION = 2.0; cfg.OPTIONS.INIT_DECODER = False
cfg.DATASET = AttrDict(); cfg.DATASET.NUM_CLASSES = 4
cfg.LOSS = AttrDict(); cfg.LOSS.OCR_ALPHA = 0.4; cfg.LOSS.OCR_AUX_RMI = False; cfg.LOSS.SUPERVISED_MSCALE_WT = 0
M = cfg.MODEL = AttrDict()
M.BNFUNC = torch.nn.BatchNorm2d
M.ALIGN_CORNERS = False
M.N_SCALES = None
M.MSCALE_LO_SCALE = 0.5
M.OCR_ASPP = False
M.SEGATTN_BOT_CH = 256
M.ASPP_BOT_CH = 256
M.MSCALE_INNER_3x3 = True
M.MSCALE_DROPOUT = False
M.MSCALE_OLDARCH = False
M.HRNET_CHECKPOINT = ""           # no ImageNet init: the full checkpoint is loaded afterwards
M.OCR = AttrDict(); M.OCR.MID_CHANNELS = 512; M.OCR.KEY_CHANNELS = 256
X = M.OCR_EXTRA = AttrDict(); X.FINAL_CONV_KERNEL = 1
for name, mods, br, block, blocks, ch in (
        ("STAGE1", 1, 1, "BOTTLENECK", [4], [64]),
        ("STAGE2", 1, 2, "BASIC", [4, 4], [48, 96]),
        ("STAGE3", 4, 3, "BASIC", [4, 4, 4], [48, 96, 192]),
        ("STAGE4", 3, 4, "BASIC", [4, 4, 4, 4], [48, 96, 192, 384])):
    s = X[name] = AttrDict()
    s.NUM_MODULES = mods; s.NUM_BRANCHES = br; s.NUM_RANCHES = br
    s.BLOCK = block; s.NUM_BLOCKS = blocks; s.NUM_CHANNELS = ch; s.FUSE_METHOD = "SUM"
