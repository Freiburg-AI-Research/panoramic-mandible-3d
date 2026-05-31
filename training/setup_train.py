"""
After build_dataset.py has produced per-case h5 files under <real_root>/<case>/ct_xray_data.h5,
this script:
  1. Copies them into the X2CT-GAN expected layout:
       X2CT/3DGAN/data/LIDC256_real/<sanitized_case>/ct_xray_data.h5
  2. Builds train.txt + test.txt with an 80/20 patient-level split.
  3. Writes a fresh yml at X2CT/3DGAN/experiment/real_singleview/<tag>.yml.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys


def sanitize(name: str) -> str:
    """Match build_dataset.py's _sanitize: replace any run of non-[A-Za-z0-9_] with _ and strip."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", default=r"./data/real_all")
    ap.add_argument("--x2ct-root", default=r"<X2CT_ROOT>")
    ap.add_argument("--data-tag", default="LIDC256_real")
    ap.add_argument("--exp-tag", default="real_singleview")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--idt-lambda", type=float, default=None,
                    help="Override CTGAN.idt_lambda in the yml (default: keep original=10).")
    ap.add_argument("--save-epoch-freq", type=int, default=None,
                    help="Override TRAIN.save_epoch_freq in the yml (default: keep original=10).")
    ap.add_argument("--bone-weight", type=float, default=None,
                    help="If set, enables masked L1 loss with `bone_weight` extra on bone voxels.")
    ap.add_argument("--bone-weight-range-lo", type=float, default=0.15)
    ap.add_argument("--bone-weight-range-hi", type=float, default=1.0)
    ap.add_argument("--keep-list", type=str, default=None,
                    help="Optional path to a text file listing case ids to include (one per line).")
    args = ap.parse_args()

    # 1. Enumerate successful cases
    summary_path = os.path.join(args.real_root, "summary.json")
    if not os.path.isfile(summary_path):
        sys.exit(f"summary.json not found at {summary_path}; run build_dataset.py first")
    with open(summary_path) as f:
        summary = json.load(f)
    ok_cases = [s for s in summary if s.get("ok")]
    case_ids = sorted({sanitize(s["case_id"]) for s in ok_cases})
    if args.keep_list:
        with open(args.keep_list) as f:
            keep = {line.strip() for line in f if line.strip()}
        case_ids = sorted(set(case_ids) & keep)
        print(f"After --keep-list filter: {len(case_ids)} cases")
    print(f"Found {len(case_ids)} successful cases.")

    # 2. Copy h5 files into X2CT data dir
    dst_root = os.path.join(args.x2ct_root, "data", args.data_tag)
    if not args.dry_run:
        os.makedirs(dst_root, exist_ok=True)
    for cid in case_ids:
        src = os.path.join(args.real_root, cid, "ct_xray_data.h5")
        if not os.path.isfile(src):
            print(f"  SKIP {cid}: no h5 at {src}"); continue
        dst_dir = os.path.join(dst_root, cid)
        dst = os.path.join(dst_dir, "ct_xray_data.h5")
        if args.dry_run:
            print(f"  [dry] {src} -> {dst}")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)

    # 3. 80/20 split
    rng = random.Random(args.seed)
    shuffled = case_ids[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * args.test_frac)))
    test_set = sorted(shuffled[:n_test])
    train_set = sorted(shuffled[n_test:])
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")

    if not args.dry_run:
        train_txt = os.path.join(args.x2ct_root, "data", f"train_{args.exp_tag}.txt")
        test_txt = os.path.join(args.x2ct_root, "data", f"test_{args.exp_tag}.txt")
        with open(train_txt, "w") as f:
            f.write("\n".join(train_set) + "\n")
        with open(test_txt, "w") as f:
            f.write("\n".join(test_set) + "\n")
        print(f"Wrote {train_txt}")
        print(f"Wrote {test_txt}")

    # 4. Write yml (copy of singleview2500 with minimal changes)
    src_yml = os.path.join(args.x2ct_root, "experiment", "singleview2500", "d2_singleview2500.yml")
    dst_yml_dir = os.path.join(args.x2ct_root, "experiment", args.exp_tag)
    dst_yml = os.path.join(dst_yml_dir, f"{args.exp_tag}.yml")
    if not args.dry_run:
        os.makedirs(dst_yml_dir, exist_ok=True)
        with open(src_yml) as f:
            yml_text = f.read()
        if args.idt_lambda is not None:
            yml_text = re.sub(r"^(\s*idt_lambda:\s*)[\d.]+", rf"\g<1>{args.idt_lambda}",
                              yml_text, count=1, flags=re.MULTILINE)
        if args.save_epoch_freq is not None:
            yml_text = re.sub(r"^(\s*save_epoch_freq:\s*)\d+", rf"\g<1>{args.save_epoch_freq}",
                              yml_text, count=1, flags=re.MULTILINE)
        if args.bone_weight is not None:
            # Enable masked L1: idt_reduction='none', idt_weight=N, idt_weight_range=[lo,hi]
            yml_text = re.sub(r"^(\s*idt_reduction:\s*)'[a-z_]+'", r"\g<1>'none'",
                              yml_text, count=1, flags=re.MULTILINE)
            yml_text = re.sub(r"^(\s*idt_weight:\s*)[\d.]+", rf"\g<1>{args.bone_weight}",
                              yml_text, count=1, flags=re.MULTILINE)
            yml_text = re.sub(r"^(\s*idt_weight_range:\s*)\[[^\]]+\]",
                              rf"\g<1>[{args.bone_weight_range_lo}, {args.bone_weight_range_hi}]",
                              yml_text, count=1, flags=re.MULTILINE)
        with open(dst_yml, "w") as f:
            f.write(yml_text)
        mods = []
        if args.idt_lambda is not None:
            mods.append(f"idt_lambda={args.idt_lambda}")
        if args.save_epoch_freq is not None:
            mods.append(f"save_epoch_freq={args.save_epoch_freq}")
        suffix = (" with " + ", ".join(mods)) if mods else ""
        print(f"Wrote {dst_yml}{suffix}")

    print("\n=== Run training with: ===")
    print(f"cd {args.x2ct_root}")
    print(f"python train.py --ymlpath=./experiment/{args.exp_tag}/{args.exp_tag}.yml \\\n"
          f"  --gpu=0 --dataroot=./data/{args.data_tag} --dataset=train \\\n"
          f"  --tag={args.exp_tag} --data={args.data_tag} \\\n"
          f"  --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN \\\n"
          f"  --datasetfile=./data/train_{args.exp_tag}.txt \\\n"
          f"  --valid_datasetfile=./data/test_{args.exp_tag}.txt --valid_dataset=test")


if __name__ == "__main__":
    main()
