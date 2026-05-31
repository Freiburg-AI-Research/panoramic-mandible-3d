"""Build two aligned datasets from cached TS masks for the two-model approach:
  - real_all_segM: target = MANDIBLE only
  - real_all_segT: target = TEETH_LOWER only
Both share the SAME mandible-union bbox crop (so predictions align in one 128^3
frame and can be overlaid/colored), the SAME full OPG input, and reduced blur
(0.3) for sharper structures.

CPU-only.
"""
from __future__ import annotations
import os, re, sys, json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from PIL import Image
import h5py
sys.path.insert(0, r".")
from opg_full import preprocess_opg_full

TS = r"./data/ts_cache"
OUT_M = r"./data/real_all_segM"
OUT_T = r"./data/real_all_segT"
DRIVE = r"<DATASET_ROOT>"
GAN = r"<EXTERNAL_OPG_ROOT>"
BLUR = 0.3


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
            a = sitk.GetArrayFromImage(r.Execute()); return a[0] if a.ndim == 3 else a
    return None

def mask_bbox(mask, pad=0.08):
    idx = np.where(mask)
    if len(idx[0]) == 0: return None
    sl = []
    for ax in range(3):
        lo, hi = int(idx[ax].min()), int(idx[ax].max()); p = int(round((hi-lo+1)*pad))
        sl.append(slice(max(0, lo-p), min(mask.shape[ax]-1, hi+p)+1))
    return tuple(sl)

def to128(mask_sub):
    sub = ndi.gaussian_filter(mask_sub.astype(np.float32), BLUR)
    m = np.clip(ndi.zoom(sub, [128/s for s in sub.shape], order=1), 0, 1)
    return (m * 3071.0).astype(np.float64)

def main():
    os.makedirs(OUT_M, exist_ok=True); os.makedirs(OUT_T, exist_ok=True)
    cases = sorted(d for d in os.listdir(TS)
                   if os.path.isfile(os.path.join(TS, d, "mandible.nii.gz")) and d != "holdout")
    print(f"Building segM + segT for {len(cases)} cases (shared union bbox, blur={BLUR})")
    nM = nT = 0
    for i, cid in enumerate(cases):
        try:
            mand = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(TS, cid, "mandible.nii.gz"))) > 0
            tlp = os.path.join(TS, cid, "teeth_lower.nii.gz")
            tl = (sitk.GetArrayFromImage(sitk.ReadImage(tlp)) > 0) if os.path.isfile(tlp) else np.zeros_like(mand)
            union = mand | tl
            bb = mask_bbox(union)
            if bb is None: continue
            tgtM = to128(mand[bb]); tgtT = to128(tl[bb])
            # OPG (full)
            if cid.startswith("case_"):
                raw = raw_drive_opg(int(re.search(r"_(\d+)$", cid).group(1)))
            else:
                raw = raw_ext_opg(cid[len("ext_"):])
            if raw is None: continue
            opg = preprocess_opg_full(raw, 256).astype(np.uint8)
            for OUT, tgt, lab in [(OUT_M, tgtM, "M"), (OUT_T, tgtT, "T")]:
                cd = os.path.join(OUT, cid); os.makedirs(cd, exist_ok=True)
                with h5py.File(os.path.join(cd, "ct_xray_data.h5"), "w") as h:
                    h.create_dataset("ct", data=tgt); h.create_dataset("xray1", data=opg)
                Image.fromarray(opg).save(os.path.join(cd, "opg_256.png"))
            nM += 1; nT += 1
            if i < 4 or i % 40 == 0:
                print(f"[{i+1}/{len(cases)}] {cid}  Mfg={float((tgtM>1535).mean())*100:.1f}%  Tfg={float((tgtT>1535).mean())*100:.2f}%")
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {cid} FAIL {e}")
    print(f"\n=== segM {nM} cases, segT {nT} cases ===")

if __name__ == "__main__":
    main()
