# Improving 3D DenseNet-121 Lung Nodule Malignancy Classification under Data Scarcity

Code and results for our comparison of three configurations of a 3D DenseNet-121 lung nodule malignancy classifier: the original results reported by Duan et al. (2025), a faithful reproduction on a much smaller LIDC-IDRI subset, and an improved version using regularization, augmentation, and uncertainty-aware inference.

**Authors:** Adam Afridi (23i-0055), Hanzlah Hassan (23i-0085), Saim Ali Abbasi (23i-2065) — FAST-NUCES Islamabad

## Overview

Deep learning malignancy classifiers typically need large training sets. We study the small-data regime: retraining the Duan et al. 3D DenseNet-121 classifier on a constrained 195-sample LIDC-IDRI subset (vs. their 1,330), then closing part of the resulting performance gap with:

- Label-smoothing BCE loss
- Enhanced 3D augmentation (rotations, Gaussian noise, intensity/contrast perturbation)
- Cosine-annealing LR + L2 weight decay
- 8-view test-time augmentation (TTA)
- Monte Carlo dropout uncertainty estimation
- Youden's-J optimal threshold selection (instead of fixed 0.5)

The head-to-head comparison (**reproduced vs. improved**, identical data/splits/hardware) is the primary contribution — the original paper's numbers are included only as external context.

## Results

| Metric | Paper (Duan et al.) | Reproduced | Improved | Δ (R→I) |
|---|---|---|---|---|
| Accuracy | 92.5% | 47.6% | 59.5% | +11.9 |
| Sensitivity | 94.1% | 19.2% | 38.5% | +19.2 |
| Specificity | 91.0% | 93.8% | 93.8% | 0.0 |
| Precision | 89.7% | 83.3% | 90.9% | +7.6 |
| F1-Score | 91.9% | 31.3% | 54.1% | +22.8 |
| AUC-ROC | 0.960 | 0.642 | 0.661 | +0.019 |
| ECE | — | — | 0.140 | — |

The improved baseline gains on **every** metric with **no regressions**, doubling sensitivity (10 vs. 5 true positives detected on the 42-sample test set) at no extra false-positive cost.

## Repository Structure

├── dataset.py              # LIDC-IDRI patch loading + preprocessing
├── analyze_dataset.py       # Class distribution / dataset statistics
├── train.py                 # Training loop (reproduced + improved configs)
├── evaluate.py               # TTA, MC-Dropout, threshold selection, metrics
├── utils.py                  # Shared helpers (augmentation, calibration, etc.)
├── splits.json                # Patient-wise train/val/test split (221/52/42)
├── train_log_001.jsonl         # Per-epoch training log
├── requirements.txt
└── results/
├── class_distribution.png
├── confusion_matrix.png
├── roc_curve.png
└── metrics_comparison.png

## Dataset

Subset of [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/): 315 pre-extracted 64³ voxel nodule patches, 195 effective samples after excluding 120 corrupt all-zero patches. Malignancy score ≥ 4 → malignant, ≤ 2 → benign (62% malignant / 38% benign). Split patient-wise to avoid leakage: 221 train / 52 val / 42 test.

> Raw patch data is not included in this repo due to size/licensing — see `dataset.py` for the extraction pipeline from LIDC-IDRI, and `splits.json` for the exact split used.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Train
python train.py --config reproduced   # matches Duan et al. recipe
python train.py --config improved     # + label smoothing, augmentation, cosine LR, weight decay

# Evaluate (TTA + MC-Dropout + Youden's-J threshold for the improved model)
python evaluate.py --checkpoint <path> --config improved

# Dataset stats
python analyze_dataset.py
```

## Method Summary

- **Architecture:** 3D DenseNet-121, 4 dense blocks (growth rate k=32, [6,12,24,16] layers), ~7M params — unchanged from Duan et al.
- **Training improvements:** label-smoothed BCE (ε=0.1), 5-transform 3D augmentation (flips, 90° rotations, Gaussian noise σ=0.02, intensity ±5%, contrast 0.9–1.1×), L2 weight decay (λ=1e-4), cosine annealing LR (1e-4 → 1e-6, 60 epochs)
- **Inference improvements:** 8-view TTA, 10-pass MC-Dropout for uncertainty
- **Post-processing:** Youden's-J optimal threshold selection, 8-bin ECE calibration reporting
- **Hardware:** single NVIDIA GTX 1660 SUPER (6GB) — vs. A100 in the original paper

## Limitations

- Small test set (42 samples) → high-variance metric estimates
- Only the classification stage is reproduced; the original paper's U-Net detection stage is not implemented
- ~38% of the available raw patches were corrupt (all-zero) and excluded
- Off-the-shelf dropout configuration used for MC sampling; not tuned for uncertainty quality

## Citation

If you use this work, please cite our paper (details to be added once published) and the original:
Y. Duan, C. Wang, Z. Wang, X. Wang, Y. Zhang, and M. Qi,
"Deep Learning-Based CT Image Analysis for Early Lung Cancer Diagnosis in Clinical Practice,"
Proc. ISAIMS 2025, Wuhan, China, Oct. 2025, pp. 501–506.
