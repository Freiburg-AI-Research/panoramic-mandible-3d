"""Robust OPG preprocessor v5c.

Fixes the two v5b bugs:
  BUG 1 (16-bit noise): v5b did gray.astype(np.uint8) on 16-bit panoramics
        -> mod-256 wraparound -> noise. Fix: robust-normalize to [0,1] FIRST,
        scale-invariant, before any uint8 use.
  BUG 2 (black bars): v5b used connected-component bbox which failed on ~13
        cases, leaving huge black borders that the no-pad resize baked in.
        Fix: crop using ROW/COLUMN MEAN-INTENSITY PROFILES (robust, no CC).
        The dental band has high row-mean; black borders ~0. We crop to the
        contiguous high-mean band. A final guard re-crops if black% is still high.

Pipeline:
  1. grayscale -> robust percentile normalize to [0,1]  (handles 8/16-bit)
  2. column-profile crop (remove L/R black)
  3. row-profile crop (remove top/bottom black + label strips)
  4. take vertical anatomy band [0.28, 0.93] of remaining height
  5. guard: if still >25% black, recompute tight bbox of >0.1 mask
  6. CLAHE, resize to (target,target)
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure


def _robust_norm(gray: np.ndarray) -> np.ndarray:
    """Percentile normalize to [0,1]; scale-invariant so 8-bit and 16-bit both work."""
    lo, hi = np.percentile(gray, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((gray - lo) / (hi - lo), 0, 1)


def _profile_crop_1d(profile: np.ndarray, frac: float = 0.12) -> tuple[int, int]:
    """Given a 1-D intensity profile, return [lo, hi] of the largest contiguous
    run above `frac * max`. Robust to label strips (isolated spikes) because we
    take the LARGEST run, not just first/last above-threshold index."""
    if profile.max() <= 0:
        return 0, len(profile)
    thr = profile.max() * frac
    above = profile > thr
    if not above.any():
        return 0, len(profile)
    # Find contiguous runs of True.
    runs = []
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(above)))
    # Largest run by length.
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return lo, hi


def preprocess_opg_v5c(opg: np.ndarray, target: int = 256,
                        vband: tuple = (0.28, 0.93),
                        clahe_clip: float = 0.03) -> np.ndarray:
    if opg.ndim == 3:
        gray = opg.astype(np.float32).mean(axis=2)
    else:
        gray = opg.astype(np.float32)

    # 1. Robust normalize (fixes 16-bit wraparound bug).
    g = _robust_norm(gray)

    # 2. Column crop (remove left/right black borders).
    col_prof = g.mean(axis=0)
    c0, c1 = _profile_crop_1d(col_prof, frac=0.12)
    g = g[:, c0:c1]

    # 3. Row crop (remove top/bottom black + label strips).
    row_prof = g.mean(axis=1)
    r0, r1 = _profile_crop_1d(row_prof, frac=0.12)
    g = g[r0:r1, :]

    # 4. Vertical anatomy band (mandibular region is lower-central).
    h, w = g.shape
    if h >= 30:
        y0 = int(round(h * vband[0]))
        y1 = int(round(h * vband[1]))
        if y1 - y0 >= 24:
            g = g[y0:y1, :]

    # 5. Guard: if still substantially black, fall back to tight bbox of content.
    black_frac = (g < 0.03).mean()
    if black_frac > 0.25:
        mask = g > 0.10
        mask = ndi.binary_closing(mask, iterations=3)
        if mask.sum() > 100:
            ys, xs = np.where(mask)
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 6. CLAHE + resize (no padding).
    g = exposure.equalize_adapthist(np.clip(g, 0, 1), clip_limit=clahe_clip, nbins=256)
    g = (g * 255.0).astype(np.float32)
    h, w = g.shape
    out = ndi.zoom(g, (target / h, target / w), order=1)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_opg_v5c_display(opg, disp_h: int = 110):
    """Same cleaning as preprocess_opg_v5c but WITHOUT the final square squash.
    Resizes to a fixed height `disp_h` preserving the natural width:height aspect,
    so figures show panoramics in true proportions. For display/figures only --
    the model uses the square preprocess_opg_v5c output."""
    if opg.ndim == 3:
        gray = opg.astype(np.float32).mean(axis=2)
    else:
        gray = opg.astype(np.float32)
    g = _robust_norm(gray)
    c0, c1 = _profile_crop_1d(g.mean(axis=0), frac=0.12); g = g[:, c0:c1]
    r0, r1 = _profile_crop_1d(g.mean(axis=1), frac=0.12); g = g[r0:r1, :]
    h, w = g.shape
    if h >= 30:
        y0 = int(round(h * 0.28)); y1 = int(round(h * 0.93))
        if y1 - y0 >= 24:
            g = g[y0:y1, :]
    black_frac = (g < 0.03).mean()
    if black_frac > 0.25:
        mask = g > 0.10
        mask = ndi.binary_closing(mask, iterations=3)
        if mask.sum() > 100:
            ys, xs = np.where(mask)
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    g = exposure.equalize_adapthist(np.clip(g, 0, 1), clip_limit=0.03, nbins=256)
    g = (g * 255.0).astype(np.float32)
    h, w = g.shape
    disp_w = max(1, int(round(w * disp_h / h)))
    out = ndi.zoom(g, (disp_h / h, disp_w / w), order=1)
    return np.clip(out, 0, 255).astype(np.uint8)
