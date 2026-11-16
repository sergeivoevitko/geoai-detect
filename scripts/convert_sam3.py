"""Convert Meta's SAM 3 checkpoint to the format GeoAI Detect loads (one-time setup).

Takes either the SAM3 .dlpk package (its sam3.pt is read straight from the zip, no Esri code is
run) or a sam3.pt file, and writes the converted model to ~/.cache/geoai/sam3, where the app
finds it automatically.

    python scripts/convert_sam3.py C:\\path\\to\\SAM3.dlpk
    python scripts/convert_sam3.py C:\\path\\to\\sam3.pt  [output_folder]

Needs about 8 GB of free RAM and 4 GB of disk. The tokenizer (a few MB) is downloaded once.

Why not the stock converter alone: the SAM 3 checkpoint that ships in the .dlpk is the *video*
model (keys "detector.*" + "tracker.*"). The stock converter expects image-model keys and silently
produces a broken model from it (every score ~0.5). Here the "detector." part is kept and the
prefix removed. The checkpoint is opened with torch.load(weights_only=True): tensors only, no code.
"""
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "sam3_convert"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.environ.get("GEOAI_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "geoai"), "sam3")
    tmp = None
    if src.lower().endswith(".dlpk"):
        z = zipfile.ZipFile(src)
        name = next((n for n in z.namelist() if n.endswith("/sam3.pt") or n == "sam3.pt"), None)
        if name is None:
            print("no sam3.pt inside", src)
            return 1
        tmp = tempfile.mkdtemp(prefix="geoai_sam3_")
        print(f"extracting {name} ...")
        src = z.extract(name, tmp)

    import convert_sam3_to_hf as conv   # vendored from transformers (Apache-2.0)

    original = conv.load_original_state_dict

    def detector_only(path):
        sd = original(path)
        det = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
        return det or sd                 # an image-only checkpoint is used as is

    conv.load_original_state_dict = detector_only
    conv.convert_sam3_checkpoint(src, out)
    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nSAM 3 is ready in {out}. Restart GeoAI Detect and tick 'Use SAM 3'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
