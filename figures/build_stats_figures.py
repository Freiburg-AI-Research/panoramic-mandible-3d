"""Statistical figures: box+violin of metrics, volume scatter (Pearson),
Bland-Altman volume agreement, and per-source breakdown."""
from __future__ import annotations
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats as sps
rcParams["font.family"] = "serif"; rcParams["font.size"] = 9

M = r"./outputs/metrics"
OUT = r"./outputs/figures"
os.makedirs(OUT, exist_ok=True)


def rd(name):
    with open(os.path.join(M, name)) as f:
        return list(csv.DictReader(f))

A = rd("A_per_case.csv"); B = rd("B_per_case.csv"); MC = rd("MC_per_case.csv")
def col(rows, k): return np.array([float(r[k]) for r in rows], float)
def src(cid): return "Site B" if cid.startswith("ext") else "Site A"


# ---- 1. Box + violin of key metrics (segmentation models only) ----
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
panels = [
    ("B: union Dice", col(B,"Dice"), "#3a7"),
    ("MC: bone Dice", col(MC,"bone_Dice"), "#37a"),
    ("MC: teeth Dice", col(MC,"teeth_Dice"), "#d9a02a"),
]
for a,(t,v,c) in zip(ax,panels):
    parts=a.violinplot(v, showmeans=False, showextrema=False)
    for b in parts['bodies']: b.set_facecolor(c); b.set_alpha(0.3)
    bp=a.boxplot(v, widths=0.25, patch_artist=True, showmeans=True,
                 meanprops=dict(marker="o",markerfacecolor="k",markeredgecolor="k",markersize=4))
    for box in bp['boxes']: box.set(facecolor=c, alpha=0.6)
    a.scatter(np.random.normal(1,0.04,len(v)), v, s=10, c="k", alpha=0.4, zorder=3)
    a.set_title(f"{t}\n{v.mean():.3f}±{v.std():.3f}"); a.set_xticks([]); a.grid(axis="y",alpha=0.3)
fig.suptitle("Per-case metric distributions (test set)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_boxviolin.png"),dpi=200,bbox_inches="tight")
fig.savefig(os.path.join(OUT,"fig_boxviolin.pdf"),bbox_inches="tight"); plt.close(fig)


# ---- 2. Volume scatter + Pearson ----
vols={}
with open(os.path.join(M,"volumes.csv")) as f:
    for r in csv.DictReader(f):
        vols.setdefault(r["model"],[]).append((float(r["gt_vol"]),float(r["pred_vol"])))
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for a,(mdl,c) in zip(ax, [("B_union","#3a7"),("MC_bone","#37a"),("MC_teeth","#d9a02a")]):
    g=np.array([x[0] for x in vols[mdl]]); p=np.array([x[1] for x in vols[mdl]])
    r,_=sps.pearsonr(g,p) if len(g)>2 and g.std()>0 and p.std()>0 else (np.nan,1)
    a.scatter(g,p,s=20,c=c,alpha=0.7,edgecolor="k",lw=0.3)
    lim=max(g.max(),p.max())*1.05
    a.plot([0,lim],[0,lim],"k--",lw=0.8,alpha=0.6)
    a.set_xlabel("GT volume (voxels)"); a.set_ylabel("Predicted volume")
    a.set_title(f"{mdl}  Pearson r={r:.2f}\npred SD={p.std():.0f} vs GT SD={g.std():.0f}"); a.grid(alpha=0.3)
fig.suptitle("Predicted vs GT volume — predictions cluster near the mean (size mean-reversion)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_volume_scatter.png"),dpi=200,bbox_inches="tight")
fig.savefig(os.path.join(OUT,"fig_volume_scatter.pdf"),bbox_inches="tight"); plt.close(fig)


# ---- 3. Bland-Altman (B union volume) ----
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for a,(mdl,c) in zip(ax, [("B_union","#3a7"),("MC_bone","#37a"),("MC_teeth","#d9a02a")]):
    g=np.array([x[0] for x in vols[mdl]]); p=np.array([x[1] for x in vols[mdl]])
    mean=(g+p)/2; diff=p-g; md=diff.mean(); sd=diff.std()
    a.scatter(mean,diff,s=20,c=c,alpha=0.7,edgecolor="k",lw=0.3)
    a.axhline(md,c="k",lw=1); a.axhline(md+1.96*sd,c="r",ls="--",lw=0.8); a.axhline(md-1.96*sd,c="r",ls="--",lw=0.8)
    a.set_xlabel("mean volume"); a.set_ylabel("pred − GT"); a.set_title(f"{mdl} Bland-Altman\nbias={md:.0f}±{1.96*sd:.0f}"); a.grid(alpha=0.3)
fig.suptitle("Volume agreement (Bland-Altman)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_bland_altman.png"),dpi=200,bbox_inches="tight")
fig.savefig(os.path.join(OUT,"fig_bland_altman.pdf"),bbox_inches="tight"); plt.close(fig)


# (Per-source figure dropped: one centre contributes only n=2 test cases, so a
#  source comparison is not statistically meaningful.)

print("wrote fig_boxviolin, fig_volume_scatter, fig_bland_altman")
# print best/median/worst for headline
for rows,key,nm in [(B,"Dice","B union"),(MC,"bone_Dice","MC bone"),(MC,"teeth_Dice","MC teeth")]:
    s=sorted(rows,key=lambda r: float(r[key]))
    print(f"{nm}: worst={s[0]['case_id']}({float(s[0][key]):.2f}) median={s[len(s)//2]['case_id']}({float(s[len(s)//2][key]):.2f}) best={s[-1]['case_id']}({float(s[-1][key]):.2f})")
