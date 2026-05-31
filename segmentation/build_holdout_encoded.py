"""Build encoded (mandible=1200, teeth=3071) holdout h5 for the held-out subject from cached masks."""
import os, numpy as np, sys
import SimpleITK as sitk
from scipy import ndimage as ndi
from PIL import Image
import h5py
sys.path.insert(0, r".")
from opg_full import preprocess_opg_full

CACHE = r"./data/ts_cache/subject"
OPG_DIR = r"<HOLDOUT_OPG_DIR>"
DST = r"./data/holdout/subject"
MAND_VAL, TEETH_VAL = 1200.0, 3071.0

def read_series(d):
    r=sitk.ImageSeriesReader(); best=None;bn=0
    for dp,_,_ in os.walk(d):
        for sid in r.GetGDCMSeriesIDs(dp):
            fn=r.GetGDCMSeriesFileNames(dp,sid)
            if len(fn)>bn: bn=len(fn);best=fn
    r.SetFileNames(best); return r.Execute()

def mask_bbox(m,pad=0.08):
    idx=np.where(m); sl=[]
    for ax in range(3):
        lo,hi=int(idx[ax].min()),int(idx[ax].max());p=int(round((hi-lo+1)*pad))
        sl.append(slice(max(0,lo-p),min(m.shape[ax]-1,hi+p)+1))
    return tuple(sl)

mand=sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(CACHE,"mandible.nii.gz")))>0
tlp=os.path.join(CACHE,"teeth_lower.nii.gz")
tl=(sitk.GetArrayFromImage(sitk.ReadImage(tlp))>0) if os.path.isfile(tlp) else np.zeros_like(mand)
tl=ndi.binary_dilation(tl,iterations=1)
union=mand|tl; bb=mask_bbox(union)
m=mand[bb]; t=tl[bb]
enc=np.zeros(m.shape,np.float32); enc[m]=MAND_VAL; enc[t]=TEETH_VAL
enc=ndi.gaussian_filter(enc,0.3)
z=ndi.zoom(enc,[128/s for s in enc.shape],order=1)
target=np.clip(z,0,3071).astype(np.float64)

opg_arr=sitk.GetArrayFromImage(read_series(OPG_DIR)); opg_arr=opg_arr[0] if opg_arr.ndim==3 else opg_arr
opg=preprocess_opg_full(opg_arr,256).astype(np.uint8)

os.makedirs(DST,exist_ok=True)
with h5py.File(os.path.join(DST,"ct_xray_data.h5"),"w") as h:
    h.create_dataset("ct",data=target); h.create_dataset("xray1",data=opg)
with open(r"./data/test_holdout_mc.txt","w") as f: f.write("holdout\n")
print("Holdout encoded built, teeth fg=%.2f%%"%(float((target>2200).mean())*100))
