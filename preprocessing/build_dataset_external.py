"""Process the external-site dataset into the same v5 (real OPG + mandible-only)
format. Output id prefix: 'ext_<case>' so it doesn't collide with the
'case_<n>' ids from the primary set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from dataclasses import asdict, dataclass

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage import exposure


# ---------------------------------------------------------------------------
# Reuse the v5 CT mandible-crop + OPG preprocessor
# ---------------------------------------------------------------------------

def find_mandible_z_range_tight(ct: np.ndarray, bone_hu: float = 500.0,
                                  pad_mm_z: float = 3.0,
                                  spacing_z_mm: float = 0.25) -> tuple[int, int]:
    Z, _, _ = ct.shape
    bone = ct > bone_hu
    profile = bone.sum(axis=(1, 2)).astype(np.float64)
    if profile.sum() < 1:
        return 0, Z - 1
    smooth = ndi.gaussian_filter1d(profile, sigma=2.0)
    peaks, _ = find_peaks(smooth, height=smooth.max() * 0.30, distance=8)
    if len(peaks) == 0:
        cum = np.cumsum(smooth); total = cum[-1]
        lo = int(np.searchsorted(cum, total * 0.15))
        hi = int(np.searchsorted(cum, total * 0.55))
    elif len(peaks) == 1:
        peak = int(peaks[0]); thr = smooth[peak] * 0.35
        lo = peak
        while lo > 0 and smooth[lo] > thr: lo -= 1
        hi = peak
        while hi < Z - 1 and smooth[hi] > thr: hi += 1
    else:
        lower_peak = int(peaks[0]); upper_peak = int(peaks[1])
        valley = lower_peak + int(np.argmin(smooth[lower_peak:upper_peak + 1]))
        thr = smooth[lower_peak] * 0.35
        lo = lower_peak
        while lo > 0 and smooth[lo] > thr: lo -= 1
        hi = valley
    pad_z = int(round(pad_mm_z / spacing_z_mm))
    lo = max(0, lo - pad_z); hi = min(Z - 1, hi + pad_z)
    if hi - lo < 24:
        mid = (lo + hi) // 2; lo = max(0, mid - 16); hi = min(Z - 1, mid + 16)
    return lo, hi


def find_mandible_bbox_3d(ct: np.ndarray, bone_hu: float = 500.0,
                           spacing_zyx: tuple = (0.25, 0.25, 0.25),
                           pad_mm_xy: float = 6.0) -> tuple[slice, slice, slice]:
    Z, Y, X = ct.shape
    z_lo, z_hi = find_mandible_z_range_tight(ct, bone_hu=bone_hu, spacing_z_mm=spacing_zyx[0])
    sub_mask = ct[z_lo:z_hi + 1] > bone_hu
    if sub_mask.sum() < 1:
        return slice(z_lo, z_hi + 1), slice(0, Y), slice(0, X)
    axial_proj = sub_mask.sum(axis=0)
    axial_thresh = axial_proj > 3
    closed = ndi.binary_closing(axial_thresh, iterations=2)
    lab, n = ndi.label(closed)
    if n == 0:
        ys = np.where(sub_mask.any(axis=(0, 2)))[0]
        xs = np.where(sub_mask.any(axis=(0, 1)))[0]
    else:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        biggest = int(sizes.argmax())
        lcc = lab == biggest
        ys = np.where(lcc.any(axis=1))[0]
        xs = np.where(lcc.any(axis=0))[0]
    y_lo, y_hi = int(ys.min()), int(ys.max())
    x_lo, x_hi = int(xs.min()), int(xs.max())
    y_pad = int(round(pad_mm_xy / spacing_zyx[1]))
    x_pad = int(round(pad_mm_xy / spacing_zyx[2]))
    y_lo = max(0, y_lo - y_pad); y_hi = min(Y - 1, y_hi + y_pad)
    x_lo = max(0, x_lo - x_pad); x_hi = min(X - 1, x_hi + x_pad)
    if y_hi - y_lo < 48:
        mid = (y_lo + y_hi) // 2; y_lo = max(0, mid - 24); y_hi = min(Y - 1, mid + 24)
    if x_hi - x_lo < 48:
        mid = (x_lo + x_hi) // 2; x_lo = max(0, mid - 24); x_hi = min(X - 1, mid + 24)
    return slice(z_lo, z_hi + 1), slice(y_lo, y_hi + 1), slice(x_lo, x_hi + 1)


def window_and_resample(ct: np.ndarray, target: tuple = (128, 128, 128),
                         hu_lo: float = -100.0, hu_hi: float = 2500.0) -> np.ndarray:
    v = np.clip(ct.astype(np.float32), hu_lo, hu_hi)
    zoom = [t / s for t, s in zip(target, v.shape)]
    return ndi.zoom(v, zoom, order=1)


def normalize_ct_to_storage(ct: np.ndarray) -> np.ndarray:
    return np.clip(ct, -1024.0, 3071.0).astype(np.float64)


def preprocess_opg_real(opg: np.ndarray, target: int = 256,
                          lower_frac: float = 0.60) -> np.ndarray:
    if opg.ndim == 3:
        gray = opg.astype(np.float32).mean(axis=2)
    else:
        gray = opg.astype(np.float32)
    H, W = gray.shape
    content = gray > 8
    content = ndi.binary_opening(content, iterations=2)
    content = ndi.binary_closing(content, iterations=5)
    if content.sum() < 100:
        cropped = gray
    else:
        lbl, _ = ndi.label(content)
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        biggest = int(sizes.argmax())
        rows, cols = np.where(lbl == biggest)
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1
        cropped = gray[ymin:ymax, xmin:xmax]
    h, w = cropped.shape
    keep_h = int(round(h * lower_frac))
    cropped = cropped[h - keep_h : h, :]
    h, w = cropped.shape
    bottom_strip = cropped[int(h * 0.92):, :]
    if (bottom_strip > 230).mean() > 0.01 or (bottom_strip < 5).mean() > 0.05:
        cropped = cropped[: int(h * 0.92), :]
    lo, hi = np.percentile(cropped, [1.0, 99.0])
    if hi <= lo: hi = lo + 1.0
    norm = np.clip((cropped - lo) / (hi - lo), 0, 1)
    norm = exposure.equalize_adapthist(norm, clip_limit=0.03, nbins=256)
    norm = (norm * 255.0).astype(np.float32)
    h, w = norm.shape
    side = max(h, w)
    pad_y = (side - h) // 2; pad_x = (side - w) // 2
    padded = np.zeros((side, side), dtype=np.float32)
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = norm
    zoom = (target / side, target / side)
    out = ndi.zoom(padded, zoom, order=1)
    return out.astype(np.uint8)


def write_h5(out_path: str, ct: np.ndarray, opg: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("ct", data=ct)
        h.create_dataset("xray1", data=opg.astype(np.uint8))


def save_qa(out_path: str, raw_opg: np.ndarray, processed_opg: np.ndarray,
            ct: np.ndarray, bbox: tuple, spacing_mm: tuple) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(raw_opg, cmap="gray")
    ax[0, 0].set_title("Raw OPG"); ax[0, 0].axis("off")
    ax[0, 1].imshow(processed_opg, cmap="gray")
    ax[0, 1].set_title("Processed OPG 256x256"); ax[0, 1].axis("off")
    ax[0, 2].text(0.02, 0.85, f"CT shape: {tuple(ct.shape)}", transform=ax[0, 2].transAxes)
    ax[0, 2].text(0.02, 0.65,
                  f"Eff spacing: {spacing_mm[0]:.2f}x{spacing_mm[1]:.2f}x{spacing_mm[2]:.2f} mm",
                  transform=ax[0, 2].transAxes)
    ax[0, 2].axis("off")
    midz, midy, midx = [s // 2 for s in ct.shape]
    bw = lambda im: np.clip((im + 100) / 1700.0, 0, 1)
    ax[1, 0].imshow(bw(ct[midz, :, :]), cmap="gray"); ax[1, 0].set_title(f"Axial z={midz}"); ax[1, 0].axis("off")
    ax[1, 1].imshow(bw(ct[:, midy, :]), cmap="gray", aspect="auto"); ax[1, 1].set_title(f"Coronal y={midy}"); ax[1, 1].axis("off")
    ax[1, 2].imshow(bw(ct[:, :, midx]), cmap="gray", aspect="auto"); ax[1, 2].set_title(f"Sagittal x={midx}"); ax[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-case discovery
# ---------------------------------------------------------------------------

def find_cases(root: str) -> dict[str, dict]:
    """Return {case_id: {ct_dir, pano_dir}} for each patient with BOTH a multi-file
    CBCT series and a single-file panoramic series."""
    out = {}
    for entry in sorted(os.listdir(root)):
        case_dir = os.path.join(root, entry)
        if not os.path.isdir(case_dir):
            continue
        ct_dir, pano_dir = None, None
        # The data is at <root>/<case>/<series>/ -- nested one level
        candidates = []
        for sub in os.listdir(case_dir):
            sp = os.path.join(case_dir, sub)
            if not os.path.isdir(sp):
                continue
            n_dcm = sum(1 for f in os.listdir(sp) if f.lower().endswith(".dcm"))
            candidates.append((sub, sp, n_dcm))
        if not candidates:
            continue
        # CBCT = the largest-N candidate
        cbct = max(candidates, key=lambda c: c[2])
        # Panoramic = a candidate with exactly 1 dicom file
        pans = [c for c in candidates if c[2] == 1]
        if cbct[2] >= 50 and pans:
            out[f"ext_{entry}"] = dict(ct_dir=cbct[1], pano_dir=pans[0][1])
    return out


@dataclass
class CaseReport:
    case_id: str
    ok: bool
    ct_shape_raw: tuple = ()
    bbox: tuple = ()
    effective_spacing_mm: tuple = ()
    ct_mean: float = 0.0
    error: str = ""


def process_case(case_id: str, ct_dir: str, pano_dir: str, out_root: str) -> CaseReport:
    case_out = os.path.join(out_root, _sanitize(case_id))
    os.makedirs(case_out, exist_ok=True)
    try:
        # CT
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(ct_dir)
        if not ids:
            raise RuntimeError(f"no CBCT series in {ct_dir}")
        # Take the largest series
        best_files = None; best_n = 0
        for sid in ids:
            fnames = reader.GetGDCMSeriesFileNames(ct_dir, sid)
            if len(fnames) > best_n:
                best_n = len(fnames); best_files = fnames
        reader.SetFileNames(best_files)
        img = reader.Execute()
        spacing_raw = img.GetSpacing()  # (x, y, z)
        ct = sitk.GetArrayFromImage(img)  # (Z, Y, X)
        ct = np.clip(ct, -1024, ct.max())

        spacing_zyx = (spacing_raw[2], spacing_raw[1], spacing_raw[0])
        bbox = find_mandible_bbox_3d(ct, spacing_zyx=spacing_zyx)
        sub = ct[bbox]
        spacing_mm = (sub.shape[0] * spacing_zyx[0] / 128.0,
                      sub.shape[1] * spacing_zyx[1] / 128.0,
                      sub.shape[2] * spacing_zyx[2] / 128.0)
        ct128 = window_and_resample(sub, target=(128, 128, 128))
        ct_for_h5 = normalize_ct_to_storage(ct128)
        sitk.WriteImage(sitk.GetImageFromArray(ct128.astype(np.int16)),
                        os.path.join(case_out, "cropped_ct_128.nii.gz"))

        # Panoramic
        pano_reader = sitk.ImageSeriesReader()
        pano_ids = pano_reader.GetGDCMSeriesIDs(pano_dir)
        if not pano_ids:
            raise RuntimeError(f"no panoramic in {pano_dir}")
        pano_fnames = pano_reader.GetGDCMSeriesFileNames(pano_dir, pano_ids[0])
        pano_reader.SetFileNames(pano_fnames)
        pano_img = pano_reader.Execute()
        pano_arr = sitk.GetArrayFromImage(pano_img)
        if pano_arr.ndim == 3:
            pano_arr = pano_arr[0]
        opg256 = preprocess_opg_real(pano_arr, target=256, lower_frac=0.60)
        Image.fromarray(opg256).save(os.path.join(case_out, "opg_256.png"))

        write_h5(os.path.join(case_out, "ct_xray_data.h5"), ct_for_h5, opg256)
        save_qa(os.path.join(case_out, "qa.png"), pano_arr, opg256, ct128, bbox, spacing_mm)

        rep = CaseReport(case_id=case_id, ok=True,
                          ct_shape_raw=tuple(int(s) for s in ct.shape),
                          bbox=((bbox[0].start, bbox[0].stop), (bbox[1].start, bbox[1].stop), (bbox[2].start, bbox[2].stop)),
                          effective_spacing_mm=tuple(float(s) for s in spacing_mm),
                          ct_mean=float(ct128.mean()))
    except Exception as e:
        tb = traceback.format_exc()
        rep = CaseReport(case_id=case_id, ok=False, error=f"{e}\n{tb}")
    with open(os.path.join(case_out, "qa.json"), "w") as f:
        json.dump(asdict(rep), f, indent=2)
    return rep


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan-root", required=True,
                    help="Top-level dataset root (one sub-directory per case).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = find_cases(args.gan_root)
    print(f"Found {len(cases)} cases with CT+panoramic.")
    summary = []
    for i, (cid, info) in enumerate(cases.items()):
        print(f"[{i+1:>3}/{len(cases)}] {cid}", flush=True)
        rep = process_case(cid, info["ct_dir"], info["pano_dir"], args.out)
        if rep.ok:
            print(f"      ok  bbox={rep.bbox}  eff_spacing={rep.effective_spacing_mm}")
        else:
            print(f"      FAIL: {rep.error.splitlines()[0]}")
        summary.append(asdict(rep))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    n_ok = sum(1 for s in summary if s["ok"])
    print(f"\n=== {n_ok}/{len(summary)} cases succeeded ===")


if __name__ == "__main__":
    main()
