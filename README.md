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
