"""
Training script for 3D DenseNet-121 malignancy classification with weighted loss.
Improvement: Use WeightedBCELoss or FocalLoss to address class imbalance.

Usage:
    python train.py \
        --data_dir /path/to/npz/files \
        --split splits.json \
        --out runs/ \
        --loss_type weighted_bce  # or 'focal'
"""

import os
import json
import argparse
import logging
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import NodulePatchDataset, compute_class_weights
from utils import DenseNet3D, WeightedBCELoss, FocalLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train 3D DenseNet for nodule malignancy classification'
    )
    parser.add_argument('--data_dir', required=True, 
                       help='Directory containing .npz patch files')
    parser.add_argument('--split', default='splits.json',
                       help='JSON file with train/val/test splits')
    parser.add_argument('--out', default='runs/',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--loss_type', default='weighted_bce',
                       choices=['bce', 'weighted_bce', 'focal'],
                       help='Loss function type')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Max epochs')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--device', default='cuda',
                       help='torch device')
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch. Returns loss, auc, accuracy."""
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    for batch_idx, (patches, labels) in enumerate(train_loader):
        patches = patches.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            all_probs.extend(probs if probs.ndim > 0 else [probs])
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    auc = compute_auc(np.array(all_labels), np.array(all_probs))
    accuracy = compute_accuracy(np.array(all_labels), np.array(all_probs))
    
    return avg_loss, auc, accuracy


def validate(model, val_loader, criterion, device):
    """Validate. Returns loss, auc, accuracy."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for patches, labels in val_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            
            logits = model(patches)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            all_probs.extend(probs if probs.ndim > 0 else [probs])
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    auc = compute_auc(np.array(all_labels), np.array(all_probs))
    accuracy = compute_accuracy(np.array(all_labels), np.array(all_probs))
    
    return avg_loss, auc, accuracy


def compute_auc(labels, probs):
    """Compute AUC-ROC."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, probs)


def compute_accuracy(labels, probs, threshold=0.5):
    """Compute accuracy."""
    preds = (probs >= threshold).astype(int)
    return (preds == labels).mean()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load splits
    with open(args.split) as f:
        splits = json.load(f)
    
    # Create datasets
    logger.info('Creating datasets...')
    train_ds = NodulePatchDataset(args.data_dir, splits['train'], augment=True)
    val_ds = NodulePatchDataset(args.data_dir, splits['val'], augment=False)
    
    # Compute class weights from training set
    weight_benign, weight_malignant = compute_class_weights(args.data_dir, splits['train'])
    logger.info(f'Class weights - Benign: {weight_benign:.4f}, Malignant: {weight_malignant:.4f}')
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    logger.info('Creating model...')
    model = DenseNet3D().to(device)
    
    # Create criterion
    if args.loss_type == 'weighted_bce':
        criterion = WeightedBCELoss(weight_benign=weight_benign, 
                                   weight_malignant=weight_malignant).to(device)
        logger.info('Using WeightedBCELoss')
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
        logger.info('Using FocalLoss')
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
        logger.info('Using standard BCEWithLogitsLoss')
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Training loop
    best_val_auc = 0.0
    epochs_without_improvement = 0
    train_history = []
    
    logger.info('Starting training...')
    for epoch in range(args.epochs):
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
        
        record = {
            'epoch': epoch,
            'train': {'loss': train_loss, 'auc': train_auc, 'accuracy': train_acc},
            'val': {'loss': val_loss, 'auc': val_auc, 'accuracy': val_acc}
        }
        train_history.append(record)
        
        logger.info(
            f'Epoch {epoch:3d} | '
            f'Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}'
        )
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            
            # Save best checkpoint
            ckpt_path = os.path.join(args.out, 'best_model.pt')
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, ckpt_path)
            logger.info(f'  → Best model saved (AUC: {val_auc:.4f})')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
    
    # Save training history
    log_path = os.path.join(args.out, 'train_log_001.jsonl')
    with open(log_path, 'w') as f:
        for record in train_history:
            f.write(json.dumps(record) + '\n')
    logger.info(f'Training history saved to {log_path}')
    
    logger.info('Training complete.')


if __name__ == '__main__':
    main()
