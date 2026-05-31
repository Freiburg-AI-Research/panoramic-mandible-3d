"""Training dynamics: (1) TensorBoard loss curves, (2) validation-metric-vs-epoch
showing the peak + A's late collapse vs B/MC stability."""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim_fn
from matplotlib import rcParams
rcParams["font.family"] = "serif"; rcParams["font.size"] = 9

X = r"<X2CT_ROOT>"
OUT = r"./outputs/figures"
os.makedirs(OUT, exist_ok=True)
SM = os.path.join(X, "save_models", "singleView_CTGAN")
EPOCHS = [10,20,30,40,50,60,70,80,90,100]


def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)
def dice(a,b): return 2*(a&b).sum()/(a.sum()+b.sum()) if (a.sum()+b.sum())>0 else 0.0


def ssim_metric(dirp):
    vals=[]
    for d in os.listdir(dirp):
        cd=os.path.join(dirp,d)
        if not os.path.isfile(os.path.join(cd,"fake_ct.mha")): continue
        g=load(os.path.join(cd,"real_ct.mha")); f=load(os.path.join(cd,"fake_ct.mha"))
        bw=lambda x: np.clip((x+100)/1700,0,1); r,p=bw(g),bw(f); D=r.shape[0]
        vals.append(np.mean([ssim_fn(r[i],p[i],data_range=1.0) for i in range(D//4,3*D//4)]))
    return float(np.mean(vals)) if vals else np.nan

def dice_metric(dirp, lo=1535, hi=None):
    vals=[]
    for d in os.listdir(dirp):
        cd=os.path.join(dirp,d)
        if not os.path.isfile(os.path.join(cd,"fake_ct.mha")): continue
        g=load(os.path.join(cd,"real_ct.mha")); f=load(os.path.join(cd,"fake_ct.mha"))
        if hi is None: gb=g>lo; pb=f>lo
        else: gb=(g>lo)&(g<=hi); pb=(f>lo)&(f<=hi)
        vals.append(dice(gb,pb))
    return float(np.mean(vals)) if vals else np.nan


def sweep(pre, data, metric):
    ys=[]
    for ep in EPOCHS:
        dirp=os.path.join(X, f"{pre}_e{ep}", data, f"test_{ep}", "CT")
        ys.append(metric(dirp) if os.path.isdir(dirp) else np.nan)
    return ys


def tb_scalars(logdir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    ea=EventAccumulator(logdir, size_guidance={"scalars":0}); ea.Reload()
    out={}
    for k in ea.Tags().get("scalars", []):
        ev=ea.Scalars(k); out[k]=([e.step for e in ev],[e.value for e in ev])
    return out


def main():
    # ---- validation metric vs epoch ----
    A_ssim = sweep("real_test_v5c", "LIDC256_real_v5c", ssim_metric)
    B_dice = sweep("segB_test", "LIDC256_segB", lambda d: dice_metric(d, 1535))
    MC_bone = sweep("segMC_test", "LIDC256_segMC", lambda d: dice_metric(d, -200, 1100))
    MC_teeth= sweep("segMC_test", "LIDC256_segMC", lambda d: dice_metric(d, 1100, None))
    json.dump(dict(epochs=EPOCHS, A_SSIM=A_ssim, B_Dice=B_dice, MC_bone_Dice=MC_bone, MC_teeth_Dice=MC_teeth),
              open(os.path.join(OUT,"..","metrics","metric_vs_epoch.json"),"w"), indent=2)

    fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 4.4))
    ax1.plot(EPOCHS, B_dice, "o-", color="#3a7", label="B: union (Dice)")
    ax1.plot(EPOCHS, MC_bone, "s-", color="#37a", label="MC: bone (Dice)")
    ax1.plot(EPOCHS, MC_teeth, "^-", color="#d9a02a", label="MC: teeth (Dice)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation Dice")
    ax1.set_title("Validation Dice vs epoch (segmentation models, stable to e100)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_metric_vs_epoch.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT,"fig_metric_vs_epoch.pdf"), bbox_inches="tight"); plt.close(fig)
    print("metric_vs_epoch:")
    print("  A SSIM:", [round(v,3) if not np.isnan(v) else None for v in A_ssim])
    print("  B Dice:", [round(v,3) if not np.isnan(v) else None for v in B_dice])
    print("  MC bone:", [round(v,3) if not np.isnan(v) else None for v in MC_bone])
    print("  MC teeth:", [round(v,3) if not np.isnan(v) else None for v in MC_teeth])

    # ---- TB loss curves ----
    logs = {
        "B (union)":     os.path.join(SM,"LIDC256_segB","real_singleview_segB","train_log"),
        "MC (multiclass)":os.path.join(SM,"LIDC256_segMC","real_singleview_segMC","train_log"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, ld) in zip(axes, logs.items()):
        sc = tb_scalars(ld) if os.path.isdir(ld) else {}
        plotted=0
        for key in sc:
            kl=key.lower()
            if any(t in kl for t in ["total","d_loss","g_loss","loss_d","loss_g","idt","map"]):
                steps,vals=sc[key]
                if len(steps)>5:
                    ax.plot(steps, vals, lw=0.9, label=key[:18]); plotted+=1
        if plotted==0 and sc:
            for key in list(sc)[:4]:
                steps,vals=sc[key]; ax.plot(steps,vals,lw=0.9,label=key[:18])
        ax.set_title(name); ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("Training loss curves", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_loss_curves.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT,"fig_loss_curves.pdf"), bbox_inches="tight"); plt.close(fig)
    print("wrote fig_metric_vs_epoch + fig_loss_curves")


if __name__ == "__main__":
    main()
