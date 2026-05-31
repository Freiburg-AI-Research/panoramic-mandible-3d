"""Robust driver for the segmentation-target batch.

Runs build_seg_targets.py ONCE PER CASE as an isolated subprocess with a hard
timeout. If any case hangs (bad DICOM read OR TS hang) the whole subprocess is
killed at the timeout and the batch moves on. Fully resumable (build_seg_targets
skips cases whose h5 already exists).
"""
from __future__ import annotations
import os, subprocess, sys, time, json

OMFS_PY = r"python"
BUILD = r"./segmentation/build_seg_targets.py"
V5C_ROOT = r"./data/real_all_v5c"
OUT = r"./data/real_all_seg"
PER_CASE_TIMEOUT = 360  # 6 min hard cap per case (healthy cases finish in 80-120s)


def _kill_tree(pid: int):
    """Windows: kill a process AND all its descendants (TS spawns nnU-Net workers
    that a plain Popen.kill() leaves orphaned and deadlocking the parent)."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True)


def run_case(cid: str):
    p = subprocess.Popen([OMFS_PY, BUILD, "--out", OUT, "--cases", cid])
    try:
        p.wait(timeout=PER_CASE_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_tree(p.pid)
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired: pass
        return "timeout"
    return None


def main():
    cases = sorted(d for d in os.listdir(V5C_ROOT)
                   if os.path.isdir(os.path.join(V5C_ROOT, d)))
    results = {}
    for i, cid in enumerate(cases):
        h5 = os.path.join(OUT, cid, "ct_xray_data.h5")
        if os.path.isfile(h5):
            print(f"[{i+1}/{len(cases)}] {cid} exists, skip", flush=True); continue
        t0 = time.time()
        timed_out = run_case(cid)
        if timed_out == "timeout":
            print(f"[{i+1}/{len(cases)}] {cid} TIMEOUT/skip ({PER_CASE_TIMEOUT}s, tree-killed)", flush=True)
            results[cid] = "timeout"
        else:
            ok = os.path.isfile(h5)
            print(f"[{i+1}/{len(cases)}] {cid} {'OK' if ok else 'NO-H5'} ({time.time()-t0:.0f}s)", flush=True)
            results[cid] = "ok" if ok else "no_h5"
    done = sum(1 for v in results.values() if v == "ok")
    total_h5 = sum(1 for c in cases if os.path.isfile(os.path.join(OUT, c, "ct_xray_data.h5")))
    print(f"\n=== driver done. new OK={done}, total h5 on disk={total_h5}/{len(cases)} ===", flush=True)
    with open(os.path.join(OUT, "driver_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
