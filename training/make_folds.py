"""Generate 5 patient-level CV folds from the segMC dataset."""
import os, random, json
DATA = r"./data/segMC"
OUTDIR = r"./data"
K = 5
cases = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))
               and os.path.isfile(os.path.join(DATA, d, "ct_xray_data.h5")))
rng = random.Random(42); rng.shuffle(cases)
folds = [cases[i::K] for i in range(K)]  # round-robin -> balanced sizes
for k in range(K):
    test = sorted(folds[k]); train = sorted(c for j in range(K) if j != k for c in folds[j])
    with open(os.path.join(OUTDIR, f"cv_train_fold{k+1}.txt"), "w") as f: f.write("\n".join(train) + "\n")
    with open(os.path.join(OUTDIR, f"cv_test_fold{k+1}.txt"), "w") as f: f.write("\n".join(test) + "\n")
    print(f"fold {k+1}: train={len(train)} test={len(test)}")
json.dump({f"fold{k+1}": sorted(folds[k]) for k in range(K)},
          open(os.path.join(r".", "cv_folds.json"), "w"), indent=2)
print(f"total cases={len(cases)}")
