"""Render intensity-encoded segMC: bone (ivory) + teeth (gold), GT vs Pred."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage as ndi

ep = sys.argv[1] if len(sys.argv) > 1 else "50"
root = rf"<X2CT_ROOT>\segMC_test_e{ep}\LIDC256_segMC\test_{ep}\CT"
cases = sorted(os.listdir(root))[:5]
TEETH = 1100; BONE_LO = -200

def add(ax, vol, lo, hi, color, alpha):
    occ = ndi.gaussian_filter((((vol > lo) & (vol < hi)).astype(np.float32)), 0.5)
    if occ.max() < 0.4: return
    try:
        v, f, _, _ = measure.marching_cubes(occ, level=0.5)
        ax.plot_trisurf(v[:,2], v[:,1], f, v[:,0], color=color, alpha=alpha, lw=0, antialiased=True)
    except Exception: pass

fig = plt.figure(figsize=(len(cases)*2.6, 5.6))
for i, c in enumerate(cases):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root,c,"real_ct.mha"))).astype(np.float32)
    pr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root,c,"fake_ct.mha"))).astype(np.float32)
    ax = fig.add_subplot(2, len(cases), i+1, projection="3d")
    add(ax, gt, BONE_LO, TEETH, "#e8ddc7", 0.30); add(ax, gt, TEETH, 1e9, "#d9b44a", 1.0)
    ax.set_axis_off(); ax.view_init(elev=12, azim=-90)
    ax.text2D(0.5, 0.95, c.replace("_ct_xray_data",""), transform=ax.transAxes, fontsize=7, ha="center")
    if i == 0: ax.set_title("GT", fontsize=9)
    ax = fig.add_subplot(2, len(cases), len(cases)+i+1, projection="3d")
    add(ax, pr, BONE_LO, TEETH, "#e8ddc7", 0.30); add(ax, pr, TEETH, 1e9, "#d9b44a", 1.0)
    ax.set_axis_off(); ax.view_init(elev=12, azim=-90)
    if i == 0: ax.set_title("Pred", fontsize=9)
fig.suptitle(f"segMC e{ep}: mandible (ivory) + teeth (gold), GT vs predicted", fontsize=11)
plt.tight_layout()
out = rf".\segMC_colored_e{ep}.png"
plt.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
