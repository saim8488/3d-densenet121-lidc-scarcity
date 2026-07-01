"""
Member 3 – Model Evaluation & Figure Generation Script with Uncertainty Quantification
Run AFTER Member 2 provides: best_model.pt, train_log_*.jsonl

Improvements:
1. Monte Carlo Dropout uncertainty estimation
2. Prediction calibration (Platt scaling / Temperature scaling)
3. Reliability diagrams & Expected Calibration Error (ECE)

Usage:
    python evaluate.py \
        --data_dir /path/to/npz/files \
        --split splits.json \
        --checkpoint runs/best_model.pt \
        --log runs/train_log_*.jsonl \
        --out figures/
"""
import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)


# ──────────────────────────────────────────────
# 1. Training curve plots
# ──────────────────────────────────────────────
def plot_training_curves(log_path_pattern: str, out_dir: str):
    log_files = sorted(glob.glob(log_path_pattern))
    if not log_files:
        print(f"WARNING: No log files matched '{log_path_pattern}'. Skipping training curves.")
        return

    history = []
    with open(log_files[-1]) as f:  # use latest log
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))

    if not history:
        print("WARNING: Log file is empty.")
        return

    epochs      = [r["epoch"] for r in history]
    train_loss  = [r["train"]["loss"] for r in history]
    val_loss    = [r["val"]["loss"] for r in history]
    train_auc   = [r["train"]["auc"] for r in history]
    val_auc     = [r["val"]["auc"] for r in history]
    train_acc   = [r["train"]["accuracy"] for r in history]
    val_acc     = [r["val"]["accuracy"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss, label="Train", color="#2196F3")
    axes[0].plot(epochs, val_loss,   label="Val",   color="#F44336")
    axes[0].set_title("BCE Loss per Epoch", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_auc, label="Train", color="#2196F3")
    axes[1].plot(epochs, val_auc,   label="Val",   color="#F44336")
    axes[1].axhline(0.96, linestyle="--", color="gray", label="Paper AUC (0.96)")
    axes[1].set_title("AUC-ROC per Epoch", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, train_acc, label="Train", color="#2196F3")
    axes[2].plot(epochs, val_acc,   label="Val",   color="#F44336")
    axes[2].axhline(0.925, linestyle="--", color="gray", label="Paper Acc (92.5%)")
    axes[2].set_title("Accuracy per Epoch", fontsize=13)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return history


# ──────────────────────────────────────────────
# 2. Run inference with MC-Dropout uncertainty
# ──────────────────────────────────────────────
def run_inference(checkpoint_path: str, data_dir: str, split_path: str, out_dir: str):
    """Load model and run on test set with MC-dropout uncertainty estimation."""
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, ".")

    from dataset import NodulePatchDataset
    from utils import DenseNet3D, monte_carlo_dropout_inference

    with open(split_path) as f:
        splits = json.load(f)

    test_ds = NodulePatchDataset(data_dir, splits["test"], augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet3D().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_probs, all_uncertainty, all_labels = [], [], []
    
    print("Running MC-Dropout inference (10 iterations per sample)...")
    with torch.no_grad():
        for batch_idx, (patches, labels) in enumerate(test_loader):
            patches = patches.to(device)
            
            # MC-dropout: 10 forward passes with dropout enabled
            mean_probs, uncertainty, _ = monte_carlo_dropout_inference(
                model, patches, num_iterations=10
            )
            
            all_probs.extend(mean_probs.cpu().numpy())
            all_uncertainty.extend(uncertainty.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f'  Batch {batch_idx + 1}/{len(test_loader)}')

    all_probs = np.array(all_probs)
    all_uncertainty = np.array(all_uncertainty)
    all_labels = np.array(all_labels)

    return all_probs, all_uncertainty, all_labels


# ──────────────────────────────────────────────
# Uncertainty visualization
# ──────────────────────────────────────────────
def plot_uncertainty(probs, uncertainty, labels, out_dir):
    """Plot uncertainty estimates: correct vs incorrect predictions."""
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(
        probs[correct], uncertainty[correct],
        alpha=0.6, s=50, label='Correct', color='#4CAF50'
    )
    axes[0].scatter(
        probs[~correct], uncertainty[~correct],
        alpha=0.6, s=50, label='Incorrect', color='#F44336'
    )
    axes[0].set_xlabel('Prediction Confidence', fontsize=11)
    axes[0].set_ylabel('Model Uncertainty (MC-Dropout Std)', fontsize=11)
    axes[0].set_title('Model Uncertainty vs Prediction Confidence', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(uncertainty[correct], bins=15, alpha=0.6, label='Correct', color='#4CAF50')
    axes[1].hist(uncertainty[~correct], bins=15, alpha=0.6, label='Incorrect', color='#F44336')
    axes[1].set_xlabel('Uncertainty', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Uncertainty Distribution', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'uncertainty_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ──────────────────────────────────────────────
# Calibration plots
# ──────────────────────────────────────────────
def plot_calibration_curve(probs, labels, out_dir, n_bins=10):
    """Plot reliability diagram (calibration curve)."""
    from utils import compute_expected_calibration_error
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_counts.append(mask.sum())

    ece = compute_expected_calibration_error(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    ax.scatter(bin_confs, bin_accs, s=np.array(bin_counts) * 2, 
              alpha=0.7, color='#2196F3', edgecolors='black', linewidth=1)
    
    sorted_idx = np.argsort(bin_confs)
    ax.plot(np.array(bin_confs)[sorted_idx], np.array(bin_accs)[sorted_idx],
           'b-', alpha=0.4, linewidth=1.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=11)
    ax.set_title(f'Calibration Curve (Reliability Diagram)\nECE: {ece:.4f}', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'calibration_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    
    return ece


# ──────────────────────────────────────────────
# 3. Full metrics & figures
# ──────────────────────────────────────────────
# 3. Full metrics & figures
# ──────────────────────────────────────────────
def compute_and_plot_metrics(probs: np.ndarray, uncertainty: np.ndarray, labels: np.ndarray,
                              out_dir: str):
    """Compute and plot comprehensive metrics with uncertainty."""
    
    preds = (probs >= 0.5).astype(int)
    report = classification_report(labels, preds, target_names=["Benign", "Malignant"], digits=4)
    print("\n" + "="*60)
    print("Classification Report (Test Set):")
    print("="*60)
    print(report)

    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0

    our_metrics = {
        "accuracy": accuracy, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision,
        "f1": f1, "auc": auc, "npv": npv,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    # Paper benchmarks
    paper = {
        "accuracy": 0.925, "sensitivity": 0.941,
        "specificity": 0.910, "precision": 0.897,
        "f1": 0.919, "auc": 0.960, "npv": 0.948,
    }

    # ── Figure 1: ROC Curve ──────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"Our Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.axhline(paper["sensitivity"], linestyle=":", color="gray", alpha=0.7,
               label=f"Paper sensitivity ({paper['sensitivity']:.3f})")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Malignancy Classification\n(Test Set)", fontsize=13)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # ── Figure 2: Confusion Matrix ───────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=13)
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # ── Figure 3: Metric comparison bar chart ────
    metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    metric_labels = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "AUC"]
    our_vals   = [our_metrics[k] for k in metric_keys]
    paper_vals = [paper[k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, our_vals,   width, label="Our Model (RTX 4050)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, paper_vals, width, label="Paper (A100)",         color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Our Results vs. Duan et al. (2025)\n3D DenseNet-121 Malignancy Classification", fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    path = os.path.join(out_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # Save metrics
    summary = {"our_model": our_metrics, "paper": paper,
               "delta": {k: our_metrics[k] - paper[k] for k in metric_keys}}
    with open(os.path.join(out_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return our_metrics
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="gray")
    path = os.path.join(out_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # ── Save summary JSON ────────────────────────
    summary = {"our_model": our_metrics, "paper": paper,
               "delta": {k: our_metrics[k] - paper[k] for k in metric_keys}}
    with open(os.path.join(out_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {os.path.join(out_dir, 'evaluation_summary.json')}")

    # Print comparison table
    print("\n── Metric Comparison ─────────────────────────────────────")
    print(f"{'Metric':<14} {'Ours':>10} {'Paper':>10} {'Δ':>10}")
    print("-" * 47)
    for k, label in zip(metric_keys, metric_labels):
        delta = our_metrics[k] - paper[k]
        sign = "+" if delta >= 0 else ""
        print(f"{label:<14} {our_metrics[k]:>10.4f} {paper[k]:>10.4f} {sign+f'{delta:.4f}':>10}")

    return our_metrics


# ──────────────────────────────────────────────
# 4. Load from pre-saved test_metrics.json
#    (if Member 2 already ran test evaluation)
# ──────────────────────────────────────────────
def plot_from_saved_metrics(test_metrics_path: str, out_dir: str):
    """Use this when you have test_metrics.json but no checkpoint."""
    with open(test_metrics_path) as f:
        m = json.load(f)

    paper = {"accuracy": 0.925, "sensitivity": 0.941, "specificity": 0.910,
             "precision": 0.897, "f1": 0.919, "auc": 0.960}

    metric_keys   = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    metric_labels = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "AUC"]
    our_vals   = [m.get(k, 0) for k in metric_keys]
    paper_vals = [paper[k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, our_vals,   width, label="Our Model (RTX 4050)", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, paper_vals, width, label="Paper (A100)",         color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Our Results vs. Duan et al. (2025)\n3D DenseNet-121 Malignancy Classification", fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "metrics_comparison_from_saved.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    print("\n── Saved Test Metrics ────────────────────────────────────")
    for k, label in zip(metric_keys, metric_labels):
        delta = m.get(k, 0) - paper.get(k, 0)
        sign = "+" if delta >= 0 else ""
        print(f"{label:<14} {m.get(k, 0):>10.4f} {paper[k]:>10.4f} {sign+f'{delta:.4f}':>10}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split', default='splits.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--log', default='runs/train_log_*.jsonl')
    parser.add_argument('--out', default='figures/')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)

    print('\n' + '='*60)
    print('EVALUATION SCRIPT WITH UNCERTAINTY QUANTIFICATION')
    print('='*60)

    # Plot training curves
    print('\n[1/5] Plotting training curves...')
    plot_training_curves(args.log, args.out)

    # Run inference with MC-Dropout
    print('\n[2/5] Running inference with MC-Dropout uncertainty...')
    probs, uncertainty, labels = run_inference(args.checkpoint, args.data_dir,
                                               args.split, args.out)

    # Plot uncertainty
    print('\n[3/5] Plotting uncertainty analysis...')
    plot_uncertainty(probs, uncertainty, labels, args.out)

    # Plot calibration
    print('\n[4/5] Plotting calibration curve...')
    ece = plot_calibration_curve(probs, labels, args.out, n_bins=10)
    print(f'Expected Calibration Error (ECE): {ece:.4f}')

    # Compute metrics
    print('\n[5/5] Computing metrics...')
    our_metrics = compute_and_plot_metrics(probs, uncertainty, labels, args.out)

    print('\n' + '='*60)
    print('EVALUATION COMPLETE')
    print('='*60)
    print(f'\nKey Results with Improvements:')
    print(f'  AUC-ROC:       {our_metrics["auc"]:.4f}')
    print(f'  Accuracy:      {our_metrics["accuracy"]:.4f}')
    print(f'  Sensitivity:   {our_metrics["sensitivity"]:.4f}')
    print(f'  Specificity:   {our_metrics["specificity"]:.4f}')
    print(f'  ECE (Calibration): {ece:.4f}')
    print(f'\nAll figures saved to: {args.out}')
    parser.add_argument("--data_dir",      default=".")
    parser.add_argument("--split",         default="splits.json")
    parser.add_argument("--checkpoint",    default=None, help="Path to best_model.pt")
    parser.add_argument("--log",           default="runs/train_log_*.jsonl")
    parser.add_argument("--test_metrics",  default=None, help="Path to test_metrics.json")
    parser.add_argument("--out",           default="figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Training curves (only needs the log file)
    history = plot_training_curves(args.log, args.out)

    # Metrics: prefer running inference from checkpoint; fall back to saved JSON
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nRunning inference from checkpoint: {args.checkpoint}")
        probs, labels = run_inference(args.checkpoint, args.data_dir, args.split)
        if probs is not None:
            compute_and_plot_metrics(probs, labels, args.out, args.test_metrics)
    elif args.test_metrics and os.path.exists(args.test_metrics):
        print(f"\nUsing saved test metrics: {args.test_metrics}")
        plot_from_saved_metrics(args.test_metrics, args.out)
    else:
        print("\nNo checkpoint or test_metrics provided — skipping metric plots.")
        print("Ask Member 2 for: best_model.pt  OR  test_metrics.json  OR  train_log_*.jsonl")
