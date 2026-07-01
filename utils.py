"""
Utility functions: 3D DenseNet-121 model, loss functions, and evaluation metrics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DenseNet3D(nn.Module):
    """
    3D DenseNet-121 for nodule malignancy classification.
    Adapted from 2D DenseNet with 3D convolutions and pooling.
    """
    
    def __init__(self, growth_rate=32, num_init_features=64, num_classes=1):
        super(DenseNet3D, self).__init__()
        self.growth_rate = growth_rate
        self.num_init_features = num_init_features
        
        # Initial convolution
        self.conv0 = nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.norm0 = nn.BatchNorm3d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks: [6, 12, 24, 16] for DenseNet-121
        num_features = num_init_features
        self.block1 = self._make_dense_block(6, num_features, growth_rate)
        num_features += 6 * growth_rate
        self.trans1 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.block2 = self._make_dense_block(12, num_features, growth_rate)
        num_features += 12 * growth_rate
        self.trans2 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.block3 = self._make_dense_block(24, num_features, growth_rate)
        num_features += 24 * growth_rate
        self.trans3 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.block4 = self._make_dense_block(16, num_features, growth_rate)
        num_features += 16 * growth_rate
        
        # Final layer
        self.norm_final = nn.BatchNorm3d(num_features)
        self.pool_final = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_dense_block(self, num_layers, num_features, growth_rate):
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(num_features + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)
    
    def _make_transition(self, num_features_in, num_features_out):
        return nn.Sequential(
            nn.BatchNorm3d(num_features_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_features_in, num_features_out, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x, enable_dropout=False):
        """
        Forward pass.
        Args:
            x: (batch, 1, 64, 64, 64)
            enable_dropout: If True, keep dropout layers active (for uncertainty)
        Returns:
            logits: (batch, 1) or sigmoid probabilities
        """
        if enable_dropout:
            self.train()  # Enable dropout
        else:
            self.eval()
        
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        x = self.block1(x)
        x = self.trans1(x)
        
        x = self.block2(x)
        x = self.trans2(x)
        
        x = self.block3(x)
        x = self.trans3(x)
        
        x = self.block4(x)
        
        x = self.norm_final(x)
        x = self.relu0(x)
        x = self.pool_final(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x


class DenseLayer(nn.Module):
    """Single dense layer: BN-ReLU-Conv-BN-ReLU-Conv"""
    
    def __init__(self, num_input_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.norm2 = nn.BatchNorm3d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        
        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        
        return torch.cat([x, new_features], 1)


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with class weights.
    Addresses class imbalance by upweighting minority class.
    """
    
    def __init__(self, weight_benign=1.0, weight_malignant=1.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_benign = weight_benign
        self.weight_malignant = weight_malignant
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch,) or (batch, 1) raw model outputs
            targets: (batch,) or (batch, 1) binary labels {0, 1}
        Returns:
            weighted loss (scalar)
        """
        logits = logits.squeeze()
        targets = targets.squeeze()
        
        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply class weights
        weights = torch.where(targets == 1, 
                             torch.tensor(self.weight_malignant, device=targets.device),
                             torch.tensor(self.weight_benign, device=targets.device))
        
        weighted_loss = (weights * bce).mean()
        return weighted_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard examples by down-weighting easy negatives.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
            gamma: Exponent parameter to balance easy vs hard examples
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch,) or (batch, 1) raw model outputs
            targets: (batch,) or (batch, 1) binary labels {0, 1}
        Returns:
            focal loss (scalar)
        """
        logits = logits.squeeze()
        targets = targets.squeeze()
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Probabilities
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1,
                             torch.tensor(self.alpha, device=targets.device),
                             torch.tensor(1 - self.alpha, device=targets.device))
        
        focal_loss = (alpha_t * focal_weight * bce).mean()
        return focal_loss


def monte_carlo_dropout_inference(model, x, num_iterations=10):
    """
    Perform Monte Carlo dropout inference to estimate model uncertainty.
    
    Args:
        model: Neural network model
        x: Input batch (batch, 1, 64, 64, 64)
        num_iterations: Number of forward passes with dropout enabled
    
    Returns:
        mean_probs: (batch,) mean prediction probabilities
        uncertainty: (batch,) uncertainty estimates (std of predictions)
        all_probs: (batch, num_iterations) raw predictions from each pass
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            logits = model(x, enable_dropout=True)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs)
    
    all_probs = torch.stack(all_probs, dim=1)  # (batch, num_iterations)
    mean_probs = all_probs.mean(dim=1)
    uncertainty = all_probs.std(dim=1)
    
    return mean_probs, uncertainty, all_probs


def calibrate_predictions(logits_val, labels_val, logits_test, method='platt'):
    """
    Post-hoc calibration of model predictions.
    
    Args:
        logits_val: Validation set logits for calibration
        labels_val: Validation set labels
        logits_test: Test set logits to calibrate
        method: 'platt' (sigmoid) or 'temperature' scaling
    
    Returns:
        calibrated_probs_test: Calibrated test set probabilities
    """
    if method == 'platt':
        # Fit sigmoid: P(y=1|x) = 1 / (1 + exp(-(a*logit + b)))
        from scipy.optimize import minimize
        
        def cross_entropy(params):
            a, b = params
            probs = torch.sigmoid(torch.tensor(a) * torch.tensor(logits_val) + 
                                 torch.tensor(b))
            ce = F.binary_cross_entropy(probs, torch.tensor(labels_val))
            return ce.item()
        
        result = minimize(cross_entropy, x0=[1.0, 0.0], method='Nelder-Mead')
        a_opt, b_opt = result.x
        
        calibrated_probs = torch.sigmoid(torch.tensor(a_opt) * torch.tensor(logits_test) + 
                                        torch.tensor(b_opt))
    
    elif method == 'temperature':
        # Temperature scaling: logits_calibrated = logits / T
        # T chosen to minimize NLL on validation set
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            if T <= 0:
                return 1e10
            scaled_logits = torch.tensor(logits_val) / T
            probs = torch.sigmoid(scaled_logits)
            return F.binary_cross_entropy(probs, torch.tensor(labels_val)).item()
        
        result = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
        T_opt = result.x
        
        calibrated_probs = torch.sigmoid(torch.tensor(logits_test) / T_opt)
    
    return calibrated_probs.numpy()


def compute_expected_calibration_error(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    Measures reliability of predicted probabilities.
    
    Args:
        probs: Predicted probabilities (0, 1)
        labels: Binary labels
        n_bins: Number of bins for histogram
    
    Returns:
        ece: Expected Calibration Error (0-1)
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        
        avg_prob = bin_probs.mean()
        accuracy = bin_labels.mean()
        
        ece += np.abs(avg_prob - accuracy) * mask.sum() / len(labels)
    
    return ece
