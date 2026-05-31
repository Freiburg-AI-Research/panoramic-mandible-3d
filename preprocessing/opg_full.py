"""Full-OPG preprocessor (no lower crop) for approach B.

Same robust de-chrome + label-strip + CLAHE as opg_v5c, but keeps the WHOLE
panoramic (both arches) instead of cropping to the lower mandibular band.
Output: 256x256 uint8 (square; X2CT-GAN squares internally anyway).
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure


def _robust_norm(gray):
    lo, hi = np.percentile(gray, [0.5, 99.5])
    if hi <= lo: hi = lo + 1.0
    return np.clip((gray - lo) / (hi - lo), 0, 1)


def _profile_crop_1d(profile, frac=0.12):
    if profile.max() <= 0: return 0, len(profile)
    thr = profile.max() * frac
    above = profile > thr
    if not above.any(): return 0, len(profile)
    runs = []; start = None
    for i, v in enumerate(above):
        if v and start is None: start = i
        elif not v and start is not None: runs.append((start, i)); start = None
    if start is not None: runs.append((start, len(above)))
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return lo, hi


def preprocess_opg_full(opg, target=256, clahe_clip=0.03):
    gray = opg.astype(np.float32).mean(axis=2) if opg.ndim == 3 else opg.astype(np.float32)
    g = _robust_norm(gray)
    # de-chrome borders by intensity profile (no vertical band crop -> keep whole OPG)
    c0, c1 = _profile_crop_1d(g.mean(axis=0), 0.12); g = g[:, c0:c1]
    r0, r1 = _profile_crop_1d(g.mean(axis=1), 0.12); g = g[r0:r1, :]
    # guard: if huge black remains, tight bbox of content
    if (g < 0.03).mean() > 0.25:
        m = ndi.binary_closing(g > 0.10, iterations=3)
        if m.sum() > 100:
            ys, xs = np.where(m); g = g[ys.min():ys.max()+1, xs.min():xs.max()+1]
    g = exposure.equalize_adapthist(np.clip(g, 0, 1), clip_limit=clahe_clip, nbins=256)
    g = (g * 255.0).astype(np.float32)
    h, w = g.shape
    out = ndi.zoom(g, (target / h, target / w), order=1)
    return np.clip(out, 0, 255).astype(np.uint8)
