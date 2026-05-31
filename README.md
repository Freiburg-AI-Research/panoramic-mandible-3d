# panoramic-mandible-3d

Source code for **3D reconstruction of the mandible and lower dentition from a single panoramic radiograph (OPG)** using a single-view generative adversarial network.

This repository contains the complete preprocessing, target generation, training,
evaluation, cross-validation and figure-generation code accompanying the study.
It does **not** contain patient data.

## Pipeline overview

```
CBCT (DICOM)  ──TotalSegmentator──►  mandible + lower-teeth masks  ──►  128³ target
Panoramic OPG ──de-border + CLAHE──►  256×256 input
                          │
                          ▼
              single-view X2CT-GAN  ──►  3D reconstruction  ──►  Dice / HD95 / ASSD
```

## Repository layout

| Folder | Contents |
|---|---|
| `preprocessing/` | DICOM→volume reading, OPG cleaning (de-border, CLAHE, resize) |
| `segmentation/`  | TotalSegmentator ground-truth generation; binary (union) and multiclass target packing |
| `training/`      | dataset split, 5-fold partitioning, training driver |
| `evaluation/`    | Dice/IoU/HD95/ASSD metrics; cross-validation aggregation |
| `figures/`       | quantitative, qualitative and 3D-rendering figure scripts |

## Dependencies

- Python 3.9+
- PyTorch (CUDA build), SimpleITK, h5py, numpy, scipy, scikit-image, matplotlib, pyvista, Pillow
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (ground-truth segmentation)
- The single-view generator is based on **X2CT-GAN** (Ying et al., CVPR 2019,
  https://github.com/kylekma/X2CT); obtain it from the original repository and
  apply the configuration files under `training/`.

```bash
pip install torch torchvision SimpleITK h5py numpy scipy scikit-image matplotlib pyvista pillow TotalSegmentator
```

## Reproducing the study

1. **Ground truth** — `python segmentation/run_seg_batch.py` runs TotalSegmentator
   (`craniofacial_structures`) on each CBCT (auto-resampled to 0.75 mm) and caches masks.
2. **Targets** — `python segmentation/finalize_segB.py` (binary union) and
   `python segmentation/build_multiclass.py` (multiclass) build 128³ targets paired
   with the cleaned full OPG.
3. **Train** — `python training/setup_train.py …` writes the data split and config,
   then train the X2CT-GAN generator with the provided settings.
4. **Cross-validation** — `python training/make_folds.py` then `training/cv_run.ps1`.
5. **Evaluate** — `python evaluation/seg_metrics.py` and `python evaluation/cv_aggregate.py`.
6. **Figures** — scripts under `figures/`.

> Paths in the scripts are placeholders or relative defaults (`./data`, `./outputs`,
> `<X2CT_ROOT>`, `<DATASET_ROOT>`, `<EXTERNAL_OPG_ROOT>`); set them to your own
> locations before running. No imaging data are included in this repository.

## License

Released under the MIT License (see `LICENSE`). The third-party X2CT-GAN code is
distributed under its own licence by its authors and is not included here.
