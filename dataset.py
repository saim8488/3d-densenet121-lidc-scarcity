"""
Nodule Patch Dataset Loader
Loads 64^3 voxel CT nodule patches with binary malignancy labels
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class NodulePatchDataset(Dataset):
    """
    Dataset for 3D nodule patches stored as .npz files.
    Each file contains:
        - patch: (64, 64, 64) float32 array [0, 1] normalized
        - label: int, 0=benign, 1=malignant
    """
    
    def __init__(self, data_dir: str, file_list: list, augment: bool = False):
        """
        Args:
            data_dir: Directory containing .npz files
            file_list: List of filenames to load (from splits.json)
            augment: Enable data augmentation (random flips/rotations)
        """
        self.data_dir = data_dir
        self.file_list = file_list
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        fpath = os.path.join(self.data_dir, fname)
        
        data = np.load(fpath)
        patch = data["patch"].astype(np.float32)  # (64, 64, 64)
        label = int(data["label"])                # 0 or 1
        
        if self.augment:
            patch = self._augment(patch)
        
        # Convert to torch: add channel dimension (1, 64, 64, 64)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return patch_tensor, label_tensor
    
    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentation: flips and 90° rotations"""
        # Random flips along each axis (50% chance each)
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0)
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1)
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2)
        
        # Random 90° rotations (randomly choose 0, 1, 2, 3 rotations)
        num_rots = np.random.randint(0, 4)
        if num_rots > 0:
            patch = np.rot90(patch, k=num_rots, axes=(0, 1))
        
        return patch.copy()


def compute_class_weights(data_dir: str, file_list: list):
    """
    Compute class weights for weighted loss.
    Weight for class c = (total_samples) / (2 * num_samples_in_class)
    This upweights minority class (benign) and downweights majority (malignant).
    
    Returns:
        weight_benign, weight_malignant (floats)
    """
    labels = []
    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        data = np.load(fpath)
        labels.append(int(data["label"]))
    
    n_benign = sum(1 for l in labels if l == 0)
    n_malignant = sum(1 for l in labels if l == 1)
    total = n_benign + n_malignant
    
    weight_benign = total / (2.0 * max(n_benign, 1))
    weight_malignant = total / (2.0 * max(n_malignant, 1))
    
    return weight_benign, weight_malignant
