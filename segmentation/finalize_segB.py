"""Finalize approach-B dataset from cached TS masks.

For every case with a cached TS mandible mask:
  - target = mask-BBOX-cropped (union of mandible + teeth_lower), 128^3, scaled
    to [0,3071] (fixes the empty-target bug: crop is defined by the mask itself,
    so it can never miss the structure; also captures the FULL mandible).
  - input  = FULL OPG (no lower crop), 256x256.

Output: real_all_segB/<case>/{ct_xray_data.h5, opg_256.png, mask_128.nii.gz, qa.png}
CPU-only; no GPU, no TotalSegmentator (reuses cached masks).
"""
from __future__ import annotations
import os, re, sys, json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from PIL import Image
import h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, r".")
from opg_full import preprocess_opg_full

TS = r"./data/ts_cache"
OUT = r"./data/real_all_segB"
DRIVE = r"<DATASET_ROOT>"
GAN = r"<EXTERNAL_OPG_ROOT>"


def raw_drive_opg(num):
    needle = f"case ({num})"
    for dp, dn, fn in os.walk(DRIVE):
        if needle in dp and "org_opg.png" in fn:
            return np.asarray(Image.open(os.path.join(dp, "org_opg.png")))
    return None


def raw_ext_opg(num):
    cd = os.path.join(GAN, str(num))
    if not os.path.isdir(cd): return None
    for sub in os.listdir(cd):
        sp = os.path.join(cd, sub)
        if os.path.isdir(sp) and sum(1 for f in os.listdir(sp) if f.endswith(".dcm")) == 1:
            r = sitk.ImageSeriesReader(); ids = r.GetGDCMSeriesIDs(sp)
            if not ids: continue
            r.SetFileNames(r.GetGDCMSeriesFileNames(sp, ids[0]))
            a = sitk.GetArrayFromImage(r.Execute())
            return a[0] if a.ndim == 3 else a
    return None


def mask_bbox(mask, pad_frac=0.08):
    idx = np.where(mask)
    if len(idx[0]) == 0: return None
    sl = []
    for ax in range(3):
        lo, hi = int(idx[ax].min()), int(idx[ax].max())
        pad = int(round((hi - lo + 1) * pad_frac))
        lo = max(0, lo - pad); hi = min(mask.shape[ax] - 1, hi + pad)
        sl.append(slice(lo, hi + 1))
    return tuple(sl)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = sorted(d for d in os.listdir(TS)
                   if os.path.isfile(os.path.join(TS, d, "mandible.nii.gz")))
    print(f"Finalizing {len(cases)} cases (mask-bbox crop + full OPG)")
    summary = []
    for i, cid in enumerate(cases):
        try:
            mand = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(TS, cid, "mandible.nii.gz")))
            tlp = os.path.join(TS, cid, "teeth_lower.nii.gz")
            tl = sitk.GetArrayFromImage(sitk.ReadImage(tlp)) if os.path.isfile(tlp) else None
            union = (mand > 0)
            if tl is not None: union = union | (tl > 0)
            bb = mask_bbox(union)
            if bb is None:
                print(f"[{i+1}/{len(cases)}] {cid} EMPTY mask, skip"); summary.append((cid, "empty")); continue
            sub = union[bb].astype(np.float32)
            sub = ndi.gaussian_filter(sub, 0.6)
            m128 = ndi.zoom(sub, [128 / s for s in sub.shape], order=1)
            target = (np.clip(m128, 0, 1) * 3071.0).astype(np.float64)
            # full OPG
            if cid.startswith("case_"):
                raw = raw_drive_opg(int(re.search(r"_(\d+)$", cid).group(1)))
            else:
                raw = raw_ext_opg(cid[len("ext_"):])
            if raw is None:
                print(f"[{i+1}/{len(cases)}] {cid} no raw OPG, skip"); summary.append((cid, "no_opg")); continue
            opg = preprocess_opg_full(raw, 256)
            cd = os.path.join(OUT, cid); os.makedirs(cd, exist_ok=True)
            with h5py.File(os.path.join(cd, "ct_xray_data.h5"), "w") as h:
                h.create_dataset("ct", data=target); h.create_dataset("xray1", data=opg.astype(np.uint8))
            Image.fromarray(opg).save(os.path.join(cd, "opg_256.png"))
            sitk.WriteImage(sitk.GetImageFromArray((target / 3071 * 255).astype(np.uint8)),
                            os.path.join(cd, "mask_128.nii.gz"))
            fg = float((target > 1535).mean()) * 100
            summary.append((cid, f"ok fg={fg:.1f}%"))
            if i < 6 or i % 30 == 0:
                print(f"[{i+1}/{len(cases)}] {cid} ok fg128={fg:.1f}%")
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {cid} FAIL {e}"); summary.append((cid, f"fail {e}"))
    nok = sum(1 for _, s in summary if s.startswith("ok"))
    with open(os.path.join(OUT, "finalize_summary.json"), "w") as f:
        json.dump(dict(summary), f, indent=2)
    print(f"\n=== finalized {nok}/{len(cases)} ===")


if __name__ == "__main__":
    main()
