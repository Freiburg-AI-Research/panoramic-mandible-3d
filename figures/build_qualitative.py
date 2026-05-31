"""Qualitative gallery: colored mandible+teeth (GT vs Pred) across the quality
range, with input OPG; plus a 3-plane slice grid."""
from __future__ import annotations
import os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage as ndi
from matplotlib import rcParams
rcParams["font.family"]="serif"; rcParams["font.size"]=9

X = r"<X2CT_ROOT>"
OUT = r"./outputs/figures"
MC = os.path.join(X, r"segMC_test_e80\LIDC256_segMC\test_80\CT")
DATA = os.path.join(X, "data", "LIDC256_segMC")
TEETH, BONE = 1100, -200
os.makedirs(OUT, exist_ok=True)

def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)
def add(ax, vol, lo, hi, col, al):
    occ=ndi.gaussian_filter(((vol>lo)&(vol<hi)).astype(np.float32),0.5)
    if occ.max()<0.4: return
    try:
        v,f,_,_=measure.marching_cubes(occ,level=0.5); ax.plot_trisurf(v[:,2],v[:,1],f,v[:,0],color=col,alpha=al,lw=0,antialiased=True)
    except Exception: pass
def opg_of(cid):
    p=os.path.join(DATA,cid,"ct_xray_data.h5")
    if os.path.isfile(p):
        with h5py.File(p,"r") as h: return np.asarray(h["xray1"])
    return None

# representative cases with COMPLETE ground truth, spanning the quality range
cases = ["case_001", "case_002", "case_003", "case_004", "case_005", "case_006"]  # example case ids
cases = [c for c in cases if os.path.isdir(os.path.join(MC, c+"_ct_xray_data"))]

rows=len(cases); cols=5
fig=plt.figure(figsize=(cols*2.7, rows*2.7))
for r,cid in enumerate(cases):
    cd=os.path.join(MC, cid+"_ct_xray_data")
    gt=load(os.path.join(cd,"real_ct.mha")); pr=load(os.path.join(cd,"fake_ct.mha"))
    opg=opg_of(cid)
    ax=fig.add_subplot(rows,cols,r*cols+1)
    if opg is not None: ax.imshow(opg,cmap="gray")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(cid,fontsize=9)
    if r==0: ax.set_title("Input OPG",fontsize=9)
    for k,(vol,lab) in enumerate([(gt,"GT"),(pr,"Pred")]):
        ax=fig.add_subplot(rows,cols,r*cols+2+k*2,projection="3d")
        add(ax,vol,BONE,TEETH,"#e8ddc7",0.32); add(ax,vol,TEETH,1e9,"#d9b44a",1.0)
        ax.set_axis_off(); ax.view_init(elev=12,azim=-90)
        if r==0: ax.set_title(f"{lab} anterior",fontsize=9)
        ax=fig.add_subplot(rows,cols,r*cols+3+k*2,projection="3d")
        add(ax,vol,BONE,TEETH,"#e8ddc7",0.32); add(ax,vol,TEETH,1e9,"#d9b44a",1.0)
        ax.set_axis_off(); ax.view_init(elev=80,azim=-90)
        if r==0: ax.set_title(f"{lab} occlusal",fontsize=9)
fig.suptitle("MC (multiclass) reconstruction: mandible (ivory) + teeth (gold) — GT vs predicted",fontsize=12,y=0.998)
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig_MC_gallery.png"),dpi=180,bbox_inches="tight")
fig.savefig(os.path.join(OUT,"fig_MC_gallery.pdf"),bbox_inches="tight"); plt.close(fig)
print("wrote fig_MC_gallery")

# slice grid for 3 cases: GT vs Pred, bone+teeth overlaid on slices
fig,axes=plt.subplots(3,6,figsize=(15,8))
sl_cases=cases[:3]
for r,cid in enumerate(sl_cases):
    cd=os.path.join(MC, cid+"_ct_xray_data")
    gt=load(os.path.join(cd,"real_ct.mha")); pr=load(os.path.join(cd,"fake_ct.mha"))
    Z,Y,Xd=gt.shape
    def overlay(ax,vol,title):
        sl=vol[Z//2]
        rgb=np.zeros((*sl.shape,3))
        rgb[(sl>BONE)&(sl<=TEETH)]=[0.9,0.87,0.78]; rgb[sl>TEETH]=[0.85,0.7,0.16]
        ax.imshow(rgb); ax.set_xticks([]);ax.set_yticks([])
        if r==0: ax.set_title(title,fontsize=9)
        ax.set_ylabel(cid,fontsize=8) if title.startswith("GT ax") else None
    overlay(axes[r,0],gt,"GT axial"); overlay(axes[r,1],pr,"Pred axial")
    # coronal
    for j,(vol,t) in enumerate([(gt,"GT coronal"),(pr,"Pred coronal")]):
        sl=vol[:,Y//2,:]; rgb=np.zeros((*sl.shape,3)); rgb[(sl>BONE)&(sl<=TEETH)]=[0.9,0.87,0.78]; rgb[sl>TEETH]=[0.85,0.7,0.16]
        axes[r,2+j].imshow(rgb,aspect="auto"); axes[r,2+j].set_xticks([]);axes[r,2+j].set_yticks([])
        if r==0: axes[r,2+j].set_title(t,fontsize=9)
    for j,(vol,t) in enumerate([(gt,"GT sagittal"),(pr,"Pred sagittal")]):
        sl=vol[:,:,Xd//2]; rgb=np.zeros((*sl.shape,3)); rgb[(sl>BONE)&(sl<=TEETH)]=[0.9,0.87,0.78]; rgb[sl>TEETH]=[0.85,0.7,0.16]
        axes[r,4+j].imshow(rgb,aspect="auto"); axes[r,4+j].set_xticks([]);axes[r,4+j].set_yticks([])
        if r==0: axes[r,4+j].set_title(t,fontsize=9)
fig.suptitle("MC slice views: bone (ivory) + teeth (gold), GT vs predicted",fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_MC_slices.png"),dpi=180,bbox_inches="tight")
fig.savefig(os.path.join(OUT,"fig_MC_slices.pdf"),bbox_inches="tight"); plt.close(fig)
print("wrote fig_MC_slices")
