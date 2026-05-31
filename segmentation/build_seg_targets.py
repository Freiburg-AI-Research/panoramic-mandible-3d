"""Approach B: build clean segmentation-mask GAN targets via TotalSegmentator.

For each v5c case:
  1. Load the source full-res CBCT (primary or external-site DICOM).
  2. If very large (e.g. 0.25 mm), resample to 0.4 mm isotropic (avoids OOM).
  3. Run TotalSegmentator task=craniofacial_structures (cached/resumable).
  4. Union(mandible, teeth_lower) -> binary lower-jaw structure, light blur.
  5. Crop to the mandible 3D bbox (same algo as v5c) -> resample to 128^3.
  6. Scale to [0, 3071] so the X2CT-GAN normalisation maps it to occupancy [0,1].
  7. Write h5 (ct = mask target, xray1 = the existing clean v5c OPG).

Run with a Python environment that has TotalSegmentator.
Resumable: skips cases whose TS seg dir already has masks, and whose h5 exists.
"""
from __future__ import annotations
import argparse, json, os, re, time, traceback
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.signal import find_peaks

TS_CACHE = r"./data/ts_cache"
V5C_ROOT = r"./data/real_all_v5c"


# ---- source CBCT loaders ----
def read_drive_cbct(num: int):
    base = r"."
    for rng in ["1-25","26-36","51-75","75-101","102-125","126-152","153-175","176-203","204-220","221-241"]:
        d = os.path.join(base, rng, f"case ({num})", "DICOM")
        if os.path.isdir(d):
            r = sitk.ImageSeriesReader(); best=None;bn=0
            for dp,_,_ in os.walk(d):
                for sid in r.GetGDCMSeriesIDs(dp):
                    fn=r.GetGDCMSeriesFileNames(dp,sid)
                    if len(fn)>bn: bn=len(fn);best=fn
            if best: r.SetFileNames(best); return r.Execute()
    return None

def read_ext_cbct(num: str):
    gan=r"<EXTERNAL_OPG_ROOT>"
    cd=os.path.join(gan,str(num))
    if not os.path.isdir(cd): return None
    r=sitk.ImageSeriesReader(); best=None;bn=0
    for sub in os.listdir(cd):
        sp=os.path.join(cd,sub)
        if not os.path.isdir(sp): continue
        for sid in r.GetGDCMSeriesIDs(sp):
            fn=r.GetGDCMSeriesFileNames(sp,sid)
            if len(fn)>bn: bn=len(fn);best=fn
    if best: r.SetFileNames(best); return r.Execute()
    return None


def resample_iso(img: sitk.Image, spacing_mm: float) -> sitk.Image:
    osz = img.GetSize(); osp = img.GetSpacing()
    nsz = [int(round(osz[i]*osp[i]/spacing_mm)) for i in range(3)]
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing((spacing_mm,)*3)
    rf.SetSize(nsz)
    rf.SetOutputDirection(img.GetDirection())
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return rf.Execute(img)


# ---- mandible bbox (same as v5c) ----
def find_mandible_z_range_tight(ct, bone_hu=500.0, pad_mm_z=3.0, spacing_z_mm=0.6):
    Z,_,_=ct.shape; bone=ct>bone_hu
    prof=bone.sum(axis=(1,2)).astype(np.float64)
    if prof.sum()<1: return 0,Z-1
    sm=ndi.gaussian_filter1d(prof,2.0)
    peaks,_=find_peaks(sm,height=sm.max()*0.30,distance=8)
    if len(peaks)==0:
        cum=np.cumsum(sm);t=cum[-1];lo=int(np.searchsorted(cum,t*0.15));hi=int(np.searchsorted(cum,t*0.55))
    elif len(peaks)==1:
        p=int(peaks[0]);thr=sm[p]*0.35;lo=p
        while lo>0 and sm[lo]>thr: lo-=1
        hi=p
        while hi<Z-1 and sm[hi]>thr: hi+=1
    else:
        lp=int(peaks[0]);up=int(peaks[1]);valley=lp+int(np.argmin(sm[lp:up+1]));thr=sm[lp]*0.35;lo=lp
        while lo>0 and sm[lo]>thr: lo-=1
        hi=valley
    pz=int(round(pad_mm_z/spacing_z_mm));lo=max(0,lo-pz);hi=min(Z-1,hi+pz)
    if hi-lo<24: m=(lo+hi)//2;lo=max(0,m-16);hi=min(Z-1,m+16)
    return lo,hi

def find_mandible_bbox_3d(ct, spacing_zyx, bone_hu=500.0, pad_mm_xy=6.0):
    Z,Y,X=ct.shape
    z0,z1=find_mandible_z_range_tight(ct,bone_hu,spacing_z_mm=spacing_zyx[0])
    sub=ct[z0:z1+1]>bone_hu
    if sub.sum()<1: return slice(z0,z1+1),slice(0,Y),slice(0,X)
    ap=sub.sum(axis=0)>3; closed=ndi.binary_closing(ap,iterations=2)
    lab,n=ndi.label(closed)
    if n==0:
        ys=np.where(sub.any(axis=(0,2)))[0];xs=np.where(sub.any(axis=(0,1)))[0]
    else:
        sizes=np.bincount(lab.ravel());sizes[0]=0;big=int(sizes.argmax());lcc=lab==big
        ys=np.where(lcc.any(axis=1))[0];xs=np.where(lcc.any(axis=0))[0]
    y0,y1=int(ys.min()),int(ys.max());x0,x1=int(xs.min()),int(xs.max())
    yp=int(round(pad_mm_xy/spacing_zyx[1]));xp=int(round(pad_mm_xy/spacing_zyx[2]))
    y0=max(0,y0-yp);y1=min(Y-1,y1+yp);x0=max(0,x0-xp);x1=min(X-1,x1+xp)
    if y1-y0<48: m=(y0+y1)//2;y0=max(0,m-24);y1=min(Y-1,m+24)
    if x1-x0<48: m=(x0+x1)//2;x0=max(0,m-24);x1=min(X-1,m+24)
    return slice(z0,z1+1),slice(y0,y1+1),slice(x0,x1+1)


TS_EXE = r"TotalSegmentator"
TS_TIMEOUT_S = 420  # per-case hard timeout so one bad case can't freeze the batch


def run_ts_cached(img: sitk.Image, cache_dir: str):
    """Run TotalSegmentator as a SUBPROCESS with a hard timeout. Running TS in a
    separate process (rather than the in-process python_api) means a hang or
    crash on one pathological case is killed at TS_TIMEOUT_S and raised, so the
    batch skips it and continues instead of freezing."""
    import subprocess
    os.makedirs(cache_dir, exist_ok=True)
    mand_p = os.path.join(cache_dir, "mandible.nii.gz")
    if os.path.isfile(mand_p):
        return cache_dir  # cached
    nii = os.path.join(cache_dir, "_in.nii.gz")
    sitk.WriteImage(img, nii)
    cmd = [TS_EXE, "-i", nii, "-o", cache_dir, "--task", "craniofacial_structures"]
    try:
        subprocess.run(cmd, timeout=TS_TIMEOUT_S, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"TS timed out after {TS_TIMEOUT_S}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TS failed (exit {e.returncode})")
    finally:
        try: os.remove(nii)
        except OSError: pass
    if not os.path.isfile(mand_p):
        raise RuntimeError("TS produced no mandible mask")
    return cache_dir


def build_target(case_id: str, img: sitk.Image):
    spacing = img.GetSpacing()  # (x,y,z)
    # Resample ALL inputs to 0.75mm isotropic before TS. This caps GPU memory and
    # host RAM (the main hang/OOM cause on this machine for sustained TS runs) and
    # is plenty accurate since the mask is downsampled to 128^3 anyway.
    work = img
    if min(spacing) < 0.72:  # only downsample (skip if already coarse)
        work = resample_iso(img, 0.75)
    seg_dir = os.path.join(TS_CACHE, case_id)
    run_ts_cached(work, seg_dir)
    def load(roi):
        p = os.path.join(seg_dir, roi + ".nii.gz")
        return sitk.GetArrayFromImage(sitk.ReadImage(p)) if os.path.isfile(p) else None
    mand = load("mandible"); tl = load("teeth_lower")
    if mand is None:
        raise RuntimeError("no mandible mask")
    union = (mand > 0)
    if tl is not None:
        union = union | (tl > 0)
    union = union.astype(np.float32)  # (Z,Y,X) on `work` grid
    work_arr = sitk.GetArrayFromImage(work).astype(np.float32)
    sp_zyx = (work.GetSpacing()[2], work.GetSpacing()[1], work.GetSpacing()[0])
    bbox = find_mandible_bbox_3d(work_arr, sp_zyx)
    sub = union[bbox]
    # light anti-alias blur then resample to 128^3
    sub = ndi.gaussian_filter(sub, 0.6)
    zoom = [128/s for s in sub.shape]
    m128 = ndi.zoom(sub, zoom, order=1)
    m128 = np.clip(m128, 0, 1)
    target = (m128 * 3071.0).astype(np.float64)  # maps to [0,1] after GAN norm
    return target, bbox, float(union.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=None, help="subset of v5c case ids")
    args = ap.parse_args()
    import h5py
    os.makedirs(args.out, exist_ok=True)
    cases = args.cases or sorted(d for d in os.listdir(V5C_ROOT)
                                  if os.path.isdir(os.path.join(V5C_ROOT, d)))
    print(f"Building seg targets for {len(cases)} cases -> {args.out}")
    summary = []
    for i, cid in enumerate(cases):
        h5_out = os.path.join(args.out, cid, "ct_xray_data.h5")
        if os.path.isfile(h5_out):
            print(f"[{i+1}/{len(cases)}] {cid} -- exists, skip"); continue
        t0 = time.time()
        try:
            if cid.startswith("case_"):
                num = int(re.search(r"_(\d+)$", cid).group(1)); img = read_drive_cbct(num)
            elif cid.startswith("ext_"):
                img = read_ext_cbct(cid[len("ext_"):])
            else:
                img = None
            if img is None:
                raise RuntimeError("source CBCT not found")
            target, bbox, nun = build_target(cid, img)
            # OPG from v5c (already clean)
            opg_p = os.path.join(V5C_ROOT, cid, "opg_256.png")
            from PIL import Image
            opg = np.asarray(Image.open(opg_p).convert("L")).astype(np.uint8)
            os.makedirs(os.path.dirname(h5_out), exist_ok=True)
            with h5py.File(h5_out, "w") as h:
                h.create_dataset("ct", data=target)
                h.create_dataset("xray1", data=opg)
            # also save a nii for QA
            sitk.WriteImage(sitk.GetImageFromArray((target/3071*255).astype(np.uint8)),
                            os.path.join(args.out, cid, "mask_128.nii.gz"))
            dt = time.time()-t0
            print(f"[{i+1}/{len(cases)}] {cid} OK  ({dt:.0f}s)  union_vox={nun:.0f}  fg128={float((target>1535).mean())*100:.2f}%")
            summary.append(dict(case_id=cid, ok=True, seconds=dt))
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {cid} FAIL: {e}")
            summary.append(dict(case_id=cid, ok=False, error=str(e)))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    nok = sum(1 for s in summary if s.get("ok"))
    print(f"\n=== {nok}/{len(summary)} seg targets built ===")


if __name__ == "__main__":
    main()
