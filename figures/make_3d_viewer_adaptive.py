"""Adaptive-iso 3D viewer.

Picks an isosurface threshold *per volume* using a percentile of the intensity
distribution above a sensible HU floor. This avoids the failure mode where the
GT iso threshold is too high for the (typically dimmer) GAN prediction, leading
to fragmented predicted meshes.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import SimpleITK as sitk
from skimage import measure


def _case_id(case_dir: str) -> str:
    return re.sub(r"_ct_xray_data$", "", os.path.basename(case_dir))


def find_case_dirs(test_dir: str) -> list[str]:
    out = []
    for d in sorted(os.listdir(test_dir)):
        full = os.path.join(test_dir, d)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "real_ct.mha")) and os.path.isfile(os.path.join(full, "fake_ct.mha")):
            out.append(full)
    return out


def load_mha(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


def pick_iso(volume: np.ndarray, hu_floor: float = 0.0, pct: float = 96.0) -> float:
    """Return an iso threshold that selects the brightest `100 - pct` percent of
    voxels above `hu_floor`. Roughly: keep top ~4% of bone-like voxels."""
    above = volume[volume > hu_floor]
    if above.size < 100:
        return float(volume.max())  # nothing to threshold; pick max -> empty mesh
    return float(np.percentile(above, pct))


def extract_mesh(volume: np.ndarray, iso: float):
    try:
        if iso >= volume.max():
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        verts, faces, _, _ = measure.marching_cubes(volume, level=iso, allow_degenerate=False)
        # Keep only the largest connected component to avoid floating noise specks.
        if len(verts) > 5000:
            verts, faces = keep_largest_component(verts, faces)
        return verts, faces
    except Exception:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)


def keep_largest_component(verts: np.ndarray, faces: np.ndarray):
    """Drop small disconnected mesh components, keep the biggest."""
    n = len(verts)
    if n == 0 or len(faces) == 0:
        return verts, faces
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for f in faces:
        union(int(f[0]), int(f[1])); union(int(f[1]), int(f[2]))
    roots = np.array([find(i) for i in range(n)])
    unique, counts = np.unique(roots, return_counts=True)
    biggest_root = unique[counts.argmax()]
    mask_v = roots == biggest_root
    new_idx = -np.ones(n, dtype=int)
    new_idx[mask_v] = np.arange(mask_v.sum())
    keep_faces = mask_v[faces].all(axis=1)
    faces_new = new_idx[faces[keep_faces]]
    return verts[mask_v], faces_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cases", type=int, default=6)
    ap.add_argument("--gt-pct", type=float, default=88.0,
                    help="Percentile of >0HU voxels above which to draw the GT isosurface.")
    ap.add_argument("--pred-pct", type=float, default=88.0,
                    help="Percentile for the prediction.")
    args = ap.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(args.out, exist_ok=True)
    case_dirs = find_case_dirs(args.test_dir)[: args.max_cases]
    print(f"Loading {len(case_dirs)} cases...")

    cases = []
    for cd in case_dirs:
        real = load_mha(os.path.join(cd, "real_ct.mha"))
        fake = load_mha(os.path.join(cd, "fake_ct.mha"))
        gt_iso = pick_iso(real, hu_floor=0.0, pct=args.gt_pct)
        pr_iso = pick_iso(fake, hu_floor=0.0, pct=args.pred_pct)
        gt_mesh = extract_mesh(real, gt_iso)
        pr_mesh = extract_mesh(fake, pr_iso)
        cid = _case_id(cd)
        print(f"  {cid:<25} gt_iso={gt_iso:.0f}HU ({len(gt_mesh[0])} verts)  pr_iso={pr_iso:.0f}HU ({len(pr_mesh[0])} verts)")
        cases.append(dict(case_id=cid, gt_mesh=gt_mesh, pr_mesh=pr_mesh,
                           gt_iso=gt_iso, pr_iso=pr_iso))

    rows = len(cases)
    cols = 2
    specs = [[{"type": "scene"}] * cols for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs,
                         column_titles=("Ground truth", "GAN prediction"),
                         row_titles=[c["case_id"] for c in cases],
                         horizontal_spacing=0.02, vertical_spacing=0.02)
    for r, c in enumerate(cases, start=1):
        for ci, (verts, faces), iso in [(1, c["gt_mesh"], c["gt_iso"]),
                                          (2, c["pr_mesh"], c["pr_iso"])]:
            if len(verts) == 0:
                continue
            mesh = go.Mesh3d(x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                              i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                              color="#dec3a3", opacity=1.0, flatshading=False,
                              lighting=dict(ambient=0.5, diffuse=0.7, specular=0.2),
                              showlegend=False, hoverinfo="text",
                              text=f"iso={iso:.0f} HU")
            fig.add_trace(mesh, row=r, col=ci)
            scene_id = f"scene{(r - 1) * cols + ci}"
            fig.update_layout({scene_id: dict(
                xaxis=dict(visible=False, showgrid=False, showbackground=False),
                yaxis=dict(visible=False, showgrid=False, showbackground=False),
                zaxis=dict(visible=False, showgrid=False, showbackground=False),
                aspectmode="data",
                bgcolor="rgb(20,20,20)",
                camera=dict(eye=dict(x=0.0, y=-2.0, z=0.3), up=dict(x=0, y=0, z=1)),
            )})
    fig.update_layout(title=dict(text=f"3D mandible reconstruction -- adaptive isosurface (top {100 - args.gt_pct:.0f}% of >0 HU voxels)",
                                  x=0.5, font=dict(size=15)),
                       paper_bgcolor="rgb(30,30,30)",
                       font=dict(color="white"),
                       height=max(420, 380 * rows),
                       margin=dict(l=10, r=10, t=80, b=10))
    path = os.path.join(args.out, "interactive_3d_viewer_adaptive.html")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"\nOpen: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
