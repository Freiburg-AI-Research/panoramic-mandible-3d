"""Segmentation-specific metrics for approach B (mask-target GAN).

The intensity metrics (PSNR/SSIM/MAE) are not the right yardstick for a binary
shape target. This computes the standard segmentation/shape metrics by
thresholding both GT and predicted mask volumes at 0.5 of their stored range:

  - Dice (DSC)         : overlap, 0..1 (primary)
  - IoU / Jaccard      : overlap, 0..1
  - Sensitivity/Recall : fraction of GT captured
  - Precision          : fraction of prediction that is correct
  - HD95 (mm)          : 95th-percentile Hausdorff surface distance
  - ASSD (mm)          : average symmetric surface distance
  - Volume ratio       : pred_volume / gt_volume

Usage:
  python seg_metrics.py --test-dir <.../CT>  [--voxel-mm 1.0] [--out <dir>]

The X2CT-GAN visual.py stores targets/preds scaled into HU; for the mask target
foreground was stored as ~3071, so we threshold at 1535 (half).
"""
from __future__ import annotations
import argparse, csv, json, os, re
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

THRESH = 1535.0  # half of 3071 (mask foreground stored as 3071)


def _cid(d): return re.sub(r"_ct_xray_data$", "", os.path.basename(d))


def find_cases(test_dir):
    out = []
    for d in sorted(os.listdir(test_dir)):
        f = os.path.join(test_dir, d)
        if os.path.isfile(os.path.join(f, "real_ct.mha")) and os.path.isfile(os.path.join(f, "fake_ct.mha")):
            out.append(f)
    return out


def surface_distances(a: np.ndarray, b: np.ndarray, voxel_mm: float):
    """Symmetric surface distances (mm) between binary volumes a, b."""
    if a.sum() == 0 or b.sum() == 0:
        return None, None
    # surface = voxels minus their erosion
    a_surf = a ^ ndi.binary_erosion(a)
    b_surf = b ^ ndi.binary_erosion(b)
    # distance transforms of the complement of each surface
    dt_b = ndi.distance_transform_edt(~b_surf) * voxel_mm
    dt_a = ndi.distance_transform_edt(~a_surf) * voxel_mm
    d_ab = dt_b[a_surf]
    d_ba = dt_a[b_surf]
    all_d = np.concatenate([d_ab, d_ba])
    assd = float(all_d.mean())
    hd95 = float(np.percentile(all_d, 95))
    return hd95, assd


def metrics(gt: np.ndarray, pred: np.ndarray, voxel_mm: float):
    g = gt > THRESH
    p = pred > THRESH
    inter = float((g & p).sum())
    gsum = float(g.sum()); psum = float(p.sum())
    dice = 2 * inter / (gsum + psum + 1e-8)
    iou = inter / (gsum + psum - inter + 1e-8)
    sens = inter / (gsum + 1e-8)
    prec = inter / (psum + 1e-8)
    volratio = psum / (gsum + 1e-8)
    hd95, assd = surface_distances(g, p, voxel_mm)
    return dict(Dice=dice, IoU=iou, Sensitivity=sens, Precision=prec,
                VolumeRatio=volratio,
                HD95_mm=hd95 if hd95 is not None else float("nan"),
                ASSD_mm=assd if assd is not None else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--voxel-mm", type=float, default=1.0,
                    help="approx isotropic voxel size of the 128^3 crop (mm); "
                         "surface distances scale with this. ~0.7 for our crops.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    case_dirs = find_cases(args.test_dir)
    rows = []
    for cd in case_dirs:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(cd, "real_ct.mha"))).astype(np.float32)
        pr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(cd, "fake_ct.mha"))).astype(np.float32)
        m = metrics(gt, pr, args.voxel_mm); m["case_id"] = _cid(cd)
        rows.append(m)
        print(f"  {m['case_id']:<24} Dice={m['Dice']:.3f} IoU={m['IoU']:.3f} "
              f"Sens={m['Sensitivity']:.3f} Prec={m['Precision']:.3f} "
              f"HD95={m['HD95_mm']:.1f}mm ASSD={m['ASSD_mm']:.2f}mm")
    if not rows:
        print("no cases"); return
    keys = ["Dice", "IoU", "Sensitivity", "Precision", "VolumeRatio", "HD95_mm", "ASSD_mm"]
    print("\n=== summary (mean +- std) ===")
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        summary[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
        print(f"  {k:<14} {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "seg_metrics_per_case.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case_id"] + keys); w.writeheader()
            for r in rows: w.writerow({kk: r[kk] for kk in (["case_id"] + keys)})
        with open(os.path.join(args.out, "seg_metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}/seg_metrics_*.{{csv,json}}")


if __name__ == "__main__":
    main()
