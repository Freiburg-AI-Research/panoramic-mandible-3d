"""Aggregate 5-fold out-of-fold predictions -> per-case MC metrics on all 121
cases, with 95% CI and per-fold means. Writes CSV + summary + box plot."""
import os, csv, json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

X = r"<X2CT_ROOT>"
OUT = r"./outputs/metrics"
FIGOUT = r"./outputs/figures"
TEETH, BONE_LO = 1100, -200
VOX = 0.7

def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)
def surf(g, p):
    if g.sum()==0 or p.sum()==0: return np.nan, np.nan
    gs=g^ndi.binary_erosion(g); ps=p^ndi.binary_erosion(p)
    d=np.concatenate([(ndi.distance_transform_edt(~ps)*VOX)[gs],(ndi.distance_transform_edt(~gs)*VOX)[ps]])
    return float(np.percentile(d,95)), float(d.mean())
def metr(gb,pb):
    i=float((gb&pb).sum()); gs=float(gb.sum()); ps=float(pb.sum())
    dice=2*i/(gs+ps+1e-8); iou=i/(gs+ps-i+1e-8)
    hd,assd=surf(gb,pb)
    return dice,iou,hd,assd

rows=[]
for k in range(1,6):
    d=os.path.join(X, f"cv_fold{k}_e80", "LIDC256_segMC", "test_80", "CT")
    if not os.path.isdir(d):
        print(f"fold {k}: MISSING {d}"); continue
    for c in sorted(os.listdir(d)):
        cd=os.path.join(d,c)
        if not os.path.isfile(os.path.join(cd,"fake_ct.mha")): continue
        g=load(os.path.join(cd,"real_ct.mha")); f=load(os.path.join(cd,"fake_ct.mha"))
        bd,bi,bh,ba=metr((g>BONE_LO)&(g<=TEETH),(f>BONE_LO)&(f<=TEETH))
        td,ti,th,ta=metr(g>TEETH,f>TEETH)
        rows.append(dict(fold=k, case_id=c.replace("_ct_xray_data",""),
                         bone_Dice=bd, bone_IoU=bi, bone_HD95=bh, bone_ASSD=ba,
                         teeth_Dice=td, teeth_IoU=ti, teeth_HD95=th, teeth_ASSD=ta))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT,"cv_per_case.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def ci(vals):
    a=np.array([v for v in vals if not np.isnan(v)],float)
    m=a.mean(); s=a.std(ddof=1); ci95=1.96*s/np.sqrt(len(a))
    return dict(mean=float(m), sd=float(s), ci95_lo=float(m-ci95), ci95_hi=float(m+ci95), n=len(a))

summary={}
for k in ["bone_Dice","bone_HD95","bone_ASSD","teeth_Dice","teeth_HD95","teeth_ASSD"]:
    summary[k]=ci([r[k] for r in rows])
# per-fold means
foldmeans={}
for k in range(1,6):
    fr=[r for r in rows if r["fold"]==k]
    foldmeans[f"fold{k}"]=dict(bone_Dice=float(np.mean([r["bone_Dice"] for r in fr])),
                                teeth_Dice=float(np.mean([r["teeth_Dice"] for r in fr])), n=len(fr))
out={"n_total":len(rows),"across_case":summary,"per_fold":foldmeans}
json.dump(out, open(os.path.join(OUT,"cv_summary.json"),"w"), indent=2)

print(f"=== 5-fold CV (n={len(rows)} cases, all patients) ===")
for k in ["bone_Dice","teeth_Dice","bone_HD95","teeth_HD95","bone_ASSD","teeth_ASSD"]:
    s=summary[k]; unit="mm" if "HD95" in k or "ASSD" in k else ""
    print(f"  {k:<12}: {s['mean']:.3f} (95% CI {s['ci95_lo']:.3f}-{s['ci95_hi']:.3f}) SD={s['sd']:.3f} {unit}")
print("  per-fold bone Dice:", [round(foldmeans[f'fold{k}']['bone_Dice'],3) for k in range(1,6)])
print("  per-fold teeth Dice:", [round(foldmeans[f'fold{k}']['teeth_Dice'],3) for k in range(1,6)])

# box plot bone+teeth Dice across all cases
fig,ax=plt.subplots(1,2,figsize=(9,4))
for a,(key,t,c) in zip(ax,[("bone_Dice","Mandible (bone) Dice","#37a"),("teeth_Dice","Teeth Dice","#d9a02a")]):
    v=np.array([r[key] for r in rows])
    parts=a.violinplot(v,showmeans=False,showextrema=False)
    for b in parts['bodies']: b.set_facecolor(c); b.set_alpha(0.3)
    bp=a.boxplot(v,widths=0.3,patch_artist=True,showmeans=True,meanprops=dict(marker="o",markerfacecolor="k",markersize=4))
    for box in bp['boxes']: box.set(facecolor=c,alpha=0.6)
    a.scatter(np.random.normal(1,0.04,len(v)),v,s=8,c="k",alpha=0.35,zorder=3)
    s=summary[key]; a.set_title(f"{t}\n{s['mean']:.3f} (95% CI {s['ci95_lo']:.3f}-{s['ci95_hi']:.3f})"); a.set_xticks([]); a.grid(axis="y",alpha=0.3)
fig.suptitle(f"5-fold cross-validation, all {len(rows)} patients",fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIGOUT,"fig_cv_dice.png"),dpi=200,bbox_inches="tight")
fig.savefig(os.path.join(FIGOUT,"fig_cv_dice.pdf"),bbox_inches="tight")
print("wrote cv_per_case.csv, cv_summary.json, fig_cv_dice")
