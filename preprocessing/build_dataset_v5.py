"""Pipeline v5 -- ORIGINAL clinical panoramic X-rays as input.

Key differences vs v3:
  * OPG source switches from BSP synthetic (segmented/Panoramic View.jpg) to the
    REAL clinical panoramic radiograph (test/org_opg.png) where available.
  * OPG preprocessor adapted for real radiographs: bottom-text removal,
    aggressive CLAHE, central content bbox.
  * CT pipeline IDENTICAL to v3 (tight mandible bbox, bone window, 128^3).

Outputs each successful case as <out>/<sanitized>/{ct_xray_data.h5, opg_256.png,
cropped_ct_128.nii.gz, qa.png, qa.json}.
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


def find_range_dirs_drive(drive_root: str) -> list[str]:
    out = []
    for d in os.listdir(drive_root):
        full = os.path.join(drive_root, d)
        if not os.path.isdir(full):
            continue
        for c in [full] + [os.path.join(full, sub) for sub in os.listdir(full) if os.path.isdir(os.path.join(full, sub))]:
            if os.path.isdir(c) and any(e.startswith("case") for e in os.listdir(c)):
                out.append(c)
                break
    return out


def build_case_index(drive_root: str, local_root: str) -> dict[str, dict]:
    """Find cases that have BOTH an org_opg.png AND a local DICOM tree."""
    drive_ranges = find_range_dirs_drive(drive_root)
    local_ranges = [os.path.join(local_root, r) for r in os.listdir(local_root)
                    if os.path.isdir(os.path.join(local_root, r))]
    org_opgs, local_cases = {}, {}
    for rp in drive_ranges:
        for sub in os.listdir(rp):
            if not sub.startswith("case"):
                continue
            # Search for any test/org_opg.png under this patient folder
            patient_dir = os.path.join(rp, sub)
            for cand in [os.path.join(patient_dir, "test", "org_opg.png"),
                          os.path.join(patient_dir, "org_opg.png")]:
                if os.path.isfile(cand):
                    org_opgs[sub] = cand
                    break
            # Also check nested duplicates
            if sub not in org_opgs:
                for dp, dn, fn in os.walk(patient_dir):
                    if "org_opg.png" in fn:
                        org_opgs[sub] = os.path.join(dp, "org_opg.png")
                        break
    for rp in local_ranges:
        if not os.path.isdir(rp):
            continue
        for sub in os.listdir(rp):
            if sub.startswith("case"):
                d = os.path.join(rp, sub, "DICOM")
                if os.path.isdir(d):
                    local_cases[sub] = d
    out = {}
    for cid in sorted(set(org_opgs) & set(local_cases), key=_case_num):
        out[cid] = dict(opg_path=org_opgs[cid], dicom_root=local_cases[cid])
    return out


def _case_num(name: str) -> int:
    m = re.search(r"\((\d+)\)", name)
    return int(m.group(1)) if m else -1


def read_largest_dicom_series(case_root: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    best_names, best_count = None, 0
    for dp, _, _ in os.walk(case_root):
        for sid in reader.GetGDCMSeriesIDs(dp):
            names = reader.GetGDCMSeriesFileNames(dp, sid)
            if len(names) > best_count:
                best_count, best_names = len(names), names
    if best_names is None:
        raise RuntimeError(f"No DICOM series in {case_root}")
    reader.SetFileNames(best_names)
    return reader.Execute()


# ---------------------------------------------------------------------------
# CT mandible-only crop (identical to v3)
# ---------------------------------------------------------------------------

def find_mandible_z_range_tight(ct: np.ndarray, bone_hu: float = 500.0,
                                  pad_mm_z: float = 3.0,
                                  spacing_z_mm: float = 0.6) -> tuple[int, int]:
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
                           spacing_zyx: tuple = (0.6, 0.5, 0.5),
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


# ---------------------------------------------------------------------------
# v5 OPG preprocessor: handle real clinical OPGs (not BSP synthetic)
# ---------------------------------------------------------------------------

def preprocess_opg_real(opg: np.ndarray, target: int = 256,
                          lower_frac: float = 0.60) -> np.ndarray:
    """Real-OPG-specific preprocessing.

    Real OPGs typically have:
      * Solid-black borders (rectangular frame)
      * Small label text at one corner (light or dark)
      * Wide aspect ratio with the dental content in the central horizontal band
    Strategy:
      1. Grayscale
      2. Find the non-trivial (gray-content) bounding box via percentile-based mask
      3. Crop to lower `lower_frac` (mandibular portion)
      4. CLAHE (clip_limit=0.03) for crisp tooth contrast
      5. Square-pad + resize to target.
    """
    if opg.ndim == 3:
        gray = opg.astype(np.float32).mean(axis=2)
    else:
        gray = opg.astype(np.float32)
    H, W = gray.shape

    # Build a "content" mask: anything that's not pure black border.
    # Use a low threshold (e.g., 8) -- the X-ray content is usually well above this.
    content = gray > 8
    # Morphologically clean -- drop tiny isolated dark pixels and small label text.
    content = ndi.binary_opening(content, iterations=2)
    content = ndi.binary_closing(content, iterations=5)
    if content.sum() < 100:
        cropped = gray
    else:
        # Keep largest connected blob.
        lbl, _ = ndi.label(content)
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        biggest = int(sizes.argmax())
        rows, cols = np.where(lbl == biggest)
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1
        cropped = gray[ymin:ymax, xmin:xmax]

    # Take lower portion (mandibular half) -- exclude max sinuses + skull edges.
    h, w = cropped.shape
    keep_h = int(round(h * lower_frac))
    cropped = cropped[h - keep_h : h, :]

    # Trim label text: scan bottom 10% for very-bright (>240) or very-dark (<5) regions and crop them.
    h, w = cropped.shape
    bottom_strip = cropped[int(h * 0.92):, :]
    if (bottom_strip > 230).mean() > 0.01 or (bottom_strip < 5).mean() > 0.05:
        # Strip the bottom 8 %.
        cropped = cropped[: int(h * 0.92), :]

    # Robust normalise.
    lo, hi = np.percentile(cropped, [1.0, 99.0])
    if hi <= lo: hi = lo + 1.0
    norm = np.clip((cropped - lo) / (hi - lo), 0, 1)

    # CLAHE -- a bit stronger than synthetic to push tooth contrast.
    norm = exposure.equalize_adapthist(norm, clip_limit=0.03, nbins=256)
    norm = (norm * 255.0).astype(np.float32)

    # Square pad + resize.
    h, w = norm.shape
    side = max(h, w)
    pad_y = (side - h) // 2; pad_x = (side - w) // 2
    padded = np.zeros((side, side), dtype=np.float32)
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = norm
    zoom = (target / side, target / side)
    out = ndi.zoom(padded, zoom, order=1)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Write + QA
# ---------------------------------------------------------------------------

def write_h5(out_path: str, ct: np.ndarray, opg: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("ct", data=ct)
        h.create_dataset("xray1", data=opg.astype(np.uint8))


def save_qa(out_path: str, raw_opg: np.ndarray, processed_opg: np.ndarray,
            ct: np.ndarray, bbox: tuple, spacing_mm: tuple) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(raw_opg, cmap="gray" if raw_opg.ndim == 2 else None)
    ax[0, 0].set_title("Raw ORIGINAL OPG"); ax[0, 0].axis("off")
    ax[0, 1].imshow(processed_opg, cmap="gray")
    ax[0, 1].set_title("Processed OPG (lower 60%, CLAHE) 256x256")
    ax[0, 1].axis("off")
    ax[0, 2].text(0.02, 0.85, f"CT shape: {tuple(ct.shape)}", transform=ax[0, 2].transAxes)
    ax[0, 2].text(0.02, 0.70, f"bbox z={bbox[0].start}:{bbox[0].stop} y={bbox[1].start}:{bbox[1].stop} x={bbox[2].start}:{bbox[2].stop}",
                  transform=ax[0, 2].transAxes, fontsize=8)
    ax[0, 2].text(0.02, 0.55, f"Eff spacing: {spacing_mm[0]:.2f}x{spacing_mm[1]:.2f}x{spacing_mm[2]:.2f} mm",
                  transform=ax[0, 2].transAxes, fontsize=9)
    ax[0, 2].text(0.02, 0.40, f"CT range: [{ct.min():.0f}, {ct.max():.0f}]",
                  transform=ax[0, 2].transAxes)
    ax[0, 2].text(0.02, 0.25, f"CT mean: {ct.mean():.1f}", transform=ax[0, 2].transAxes)
    ax[0, 2].axis("off")
    midz, midy, midx = [s // 2 for s in ct.shape]
    bw = lambda im: np.clip((im + 100) / 1700.0, 0, 1)
    ax[1, 0].imshow(bw(ct[midz, :, :]), cmap="gray")
    ax[1, 0].set_title(f"Axial z={midz}"); ax[1, 0].axis("off")
    ax[1, 1].imshow(bw(ct[:, midy, :]), cmap="gray", aspect="auto")
    ax[1, 1].set_title(f"Coronal y={midy}"); ax[1, 1].axis("off")
    ax[1, 2].imshow(bw(ct[:, :, midx]), cmap="gray", aspect="auto")
    ax[1, 2].set_title(f"Sagittal x={midx}"); ax[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


@dataclass
class CaseReport:
    case_id: str
    ok: bool
    ct_shape_raw: tuple = ()
    bbox: tuple = ()
    effective_spacing_mm: tuple = ()
    ct_mean: float = 0.0
    error: str = ""


def process_case(case_id: str, opg_path: str, dicom_root: str, out_root: str) -> CaseReport:
    case_out = os.path.join(out_root, _sanitize(case_id))
    os.makedirs(case_out, exist_ok=True)
    try:
        img = read_largest_dicom_series(dicom_root)
        spacing_raw = img.GetSpacing()
        ct = sitk.GetArrayFromImage(img)
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

        raw_opg = np.asarray(Image.open(opg_path))
        opg256 = preprocess_opg_real(raw_opg, target=256, lower_frac=0.60)
        Image.fromarray(opg256).save(os.path.join(case_out, "opg_256.png"))

        write_h5(os.path.join(case_out, "ct_xray_data.h5"), ct_for_h5, opg256)
        save_qa(os.path.join(case_out, "qa.png"), raw_opg, opg256, ct128, bbox, spacing_mm)

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
    ap.add_argument("--drive-root", default=r"<DATASET_ROOT>")
    ap.add_argument("--local-root", default=r".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = build_case_index(args.drive_root, args.local_root)
    if args.cases:
        index = {cid: index[cid] for cid in args.cases if cid in index}
    print(f"Processing {len(index)} cases (v5 -- ORIGINAL OPGs) -> {args.out}")
    summary = []
    for i, (cid, info) in enumerate(index.items()):
        print(f"[{i+1:>3}/{len(index)}] {cid}", flush=True)
        rep = process_case(cid, info["opg_path"], info["dicom_root"], args.out)
        if rep.ok:
            print(f"      ok  bbox={rep.bbox}  eff_spacing={rep.effective_spacing_mm}")
        else:
            print(f"      FAIL: {rep.error.splitlines()[0]}")
        summary.append(asdict(rep))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    n_ok = sum(1 for s in summary if s["ok"])
    print(f"\n=== {n_ok}/{len(summary)} cases succeeded (v5) ===")


if __name__ == "__main__":
    main()
