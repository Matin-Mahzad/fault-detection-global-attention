"""
Fault Detection in Seismic Data Using a True 3D Global
Attention Convolutional Network with Self-Supervised Denoising Pretext Training

Copyright (c) 2025 Matin Mahzad

This implementation accompanies the paper:
"Self-Supervised Fault Detection in Seismic Data Using a True 3D Global
Attention Convolutional Network with Denoising Pretext Training"
by Matin Mahzad and Majid Bagheri

Paper:
Authors:
  - Matin Mahzad (ORCID: 0009-0000-9346-8451)
  - Majid Bagheri

Description:
This repository contains the official implementation of a hybrid U-Net
architecture combining 3D convolutional processing with unfactorized global
self-attention for seismic fault segmentation. The model employs a two-stage
training paradigm: (1) self-supervised denoising pretext training across
multiple seismic surveys, followed by (2) discriminative transfer learning
on fault-labeled data.

Key Features:
  - True 3D global attention (unfactorized, every-voxel-to-every-voxel)
  - Multi-survey denoising pretext training (Kerry-3D, Opunake-3D, Kahu-3D)
  - Discriminative transfer learning with layer-wise learning rate decay
  - Unified Focal Loss for extreme class imbalance
  - State-of-the-art performance on Thebe survey validation/test data

License: MIT License (see LICENSE file for details)

Citation:
If you use this code in your research, please cite:
@article{mahzad2025seismic,
  title={Self-Supervised Fault Detection in Seismic Data Using a True 3D
         Global Attention Convolutional Network with Denoising Pretext Training},
  author={Mahzad, Matin and Bagheri, Majid},
  journal={[Journal Name]},
  year={2025}
}

Requirements:
  - Python >= 3.10
  - PyTorch >= 2.0
  - NumPy, SciPy
  - [Additional dependencies listed in requirements.txt]

Contact:
For questions or issues, please open an issue on GitHub or contact:
  - Matin Mahzad: matinmahzad@yahoo.com
"""

__version__ = "1.0.0"
__author__ = "Matin Mahzad"
__license__ = "MIT"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class SlantedTriangularLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Slanted Triangular Learning Rate Scheduler.

    Implements a two-phase learning rate schedule consisting of a linear warm-up
    period followed by linear decay. This schedule facilitates gradual adaptation
    to the target task while preventing catastrophic forgetting of pretrained
    features.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    num_epochs : int
        Total number of training epochs.
    steps_per_epoch : int
        Number of optimization steps per epoch.
    cut_frac : float, optional
        Fraction of total steps allocated to warm-up phase (default: 0.1).
    ratio : int, optional
        Ratio between maximum and minimum learning rates (default: 32).
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 steps_per_epoch: int,
                 cut_frac: float = 0.1,
                 ratio: int = 32):

        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = num_epochs * steps_per_epoch
        self.cut = int(self.total_steps * cut_frac)
        self.ratio = ratio
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self):
        """Calculate learning rate for current optimization step."""
        t = self.current_step
        cut = self.cut

        if t < cut:
            p = t / cut
        else:
            p = 1 - (t - cut) / (self.total_steps - cut)

        lr_multiplier = (1 + p * (self.ratio - 1)) / self.ratio

        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Perform a single scheduler step."""
        self.current_step += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# =============================================================================
# Layer-wise Learning Rate Decay
# =============================================================================

def get_layer_wise_lr_groups(model: nn.Module,
                             base_lr: float,
                             decay_factor: float = 0.95,
                             weight_decay: float = 1e-4) -> List[Dict]:
    """
    Construct parameter groups with discriminative learning rates.

    Implements layer-wise learning rate decay (LLRD) where parameters closer
    to the output receive higher learning rates than those closer to the input.
    This approach preserves low-level pretrained features while allowing
    task-specific adaptation of high-level representations.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    base_lr : float
        Base learning rate for the top layer.
    decay_factor : float, optional
        Exponential decay factor applied per layer (default: 0.95).
    weight_decay : float, optional
        L2 regularization coefficient (default: 1e-4).

    Returns
    -------
    List[Dict]
        Parameter groups with assigned learning rates.
    """

    try:
        all_params = list(model.named_parameters())
    except:
        all_params = [(f"param_{i}", p) for i, p in enumerate(model.parameters())]

    if not all_params:
        raise ValueError("Model contains no trainable parameters")

    layer_groups = defaultdict(list)

    for name, param in all_params:
        if not param.requires_grad:
            continue

        if 'out_conv' in name or 'output' in name or 'out' in name:
            layer_groups[0].append((name, param))
        elif 'dec' in name or 'up' in name:
            if 'dec1' in name or 'up1' in name or '1' in name:
                layer_groups[1].append((name, param))
            elif 'dec2' in name or 'up2' in name or '2' in name:
                layer_groups[2].append((name, param))
            else:
                layer_groups[1].append((name, param))
        elif 'bridge' in name:
            layer_groups[3].append((name, param))
        elif 'enc' in name or 'down' in name:
            if 'enc2' in name or 'down2' in name or '2' in name:
                layer_groups[4].append((name, param))
            elif 'enc1' in name or 'down1' in name or '1' in name:
                layer_groups[5].append((name, param))
            else:
                layer_groups[4].append((name, param))
        else:
            param_idx = len([p for g in layer_groups.values() for p in g])
            total_params = len([p for _, p in all_params if p.requires_grad])
            group_idx = int((param_idx / total_params) * 6)
            layer_groups[5 - group_idx].append((name, param))

    if not layer_groups:
        print("Warning: Could not identify layer structure, using sequential grouping")
        trainable_params = [(name, p) for name, p in all_params if p.requires_grad]
        n_params = len(trainable_params)
        n_groups = 6
        params_per_group = n_params // n_groups

        for group_idx in range(n_groups):
            start = group_idx * params_per_group
            end = start + params_per_group if group_idx < n_groups - 1 else n_params
            layer_groups[5 - group_idx] = trainable_params[start:end]

    param_groups = []

    for group_idx in sorted(layer_groups.keys()):
        lr = base_lr * (decay_factor ** group_idx)
        params = [p for _, p in layer_groups[group_idx]]

        if params:
            param_groups.append({
                'params': params,
                'lr': lr,
                'weight_decay': weight_decay,
                'name': f'layer_group_{group_idx}'
            })

            print(f"  Layer Group {group_idx}: LR = {lr:.2e}, "
                  f"Parameters = {sum(p.numel() for p in params):,}")

    if not param_groups:
        raise ValueError("No trainable parameters found")

    return param_groups


# =============================================================================
# Gradual Unfreezing
# =============================================================================

class GradualUnfreezing:
    """
    Progressive layer unfreezing for transfer learning.

    Implements gradual unfreezing whereby layers are progressively made trainable
    from the output (decoder) toward the input (encoder). This strategy mitigates
    catastrophic forgetting by initially preserving pretrained features while
    allowing incremental adaptation to the target task.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    unfreeze_schedule : List[int]
        Epoch indices at which successive layer groups become trainable.
    """

    def __init__(self,
                 model: nn.Module,
                 unfreeze_schedule: List[int]):

        self.model = model
        self.unfreeze_schedule = sorted(unfreeze_schedule)
        self.current_unfrozen = -1
        self.layer_groups = self._identify_layer_groups()
        self._freeze_all()

        print("\n" + "=" * 70)
        print("Gradual Unfreezing Schedule")
        print("=" * 70)
        print(f"Schedule: {unfreeze_schedule}")
        for i, (name, _) in enumerate(self.layer_groups):
            if i < len(unfreeze_schedule):
                print(f"  Epoch {unfreeze_schedule[i]}: {name}")
        print("=" * 70 + "\n")

    def _identify_layer_groups(self) -> List[Tuple[str, List[nn.Parameter]]]:
        """Identify and group model parameters by architectural depth."""
        groups = [
            ("output_decoder", []),
            ("bridge", []),
            ("encoder_top", []),
            ("encoder_bottom", [])
        ]

        try:
            all_params = list(self.model.named_parameters())
        except:
            all_params = [(f"param_{i}", p) for i, p in enumerate(self.model.parameters())]

        for name, param in all_params:
            if 'out_conv' in name or 'out' in name or 'dec' in name or 'up' in name:
                groups[0][1].append((name, param))
            elif 'bridge' in name:
                groups[1][1].append((name, param))
            elif 'enc2' in name or 'down2' in name or ('enc' in name and '2' in name):
                groups[2][1].append((name, param))
            elif 'enc1' in name or 'down1' in name or ('enc' in name and '1' in name):
                groups[3][1].append((name, param))
            else:
                param_idx = sum(len(g[1]) for g in groups)
                total_params = len(all_params)

                if param_idx < total_params * 0.25:
                    groups[3][1].append((name, param))
                elif param_idx < total_params * 0.5:
                    groups[2][1].append((name, param))
                elif param_idx < total_params * 0.75:
                    groups[1][1].append((name, param))
                else:
                    groups[0][1].append((name, param))

        result = [(name, params) for name, params in groups if params]

        if not result:
            print("Warning: Could not identify structure, using equal groups")
            all_trainable = [(name, p) for name, p in all_params]
            n = len(all_trainable)
            result = [
                ("group_0", all_trainable[:n // 4]),
                ("group_1", all_trainable[n // 4:n // 2]),
                ("group_2", all_trainable[n // 2:3 * n // 4]),
                ("group_3", all_trainable[3 * n // 4:])
            ]

        return result

    def _freeze_all(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def step(self, epoch: int):
        """
        Update trainable parameters according to unfreezing schedule.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        """
        while (self.current_unfrozen + 1 < len(self.unfreeze_schedule) and
               epoch >= self.unfreeze_schedule[self.current_unfrozen + 1]):

            self.current_unfrozen += 1

            if self.current_unfrozen < len(self.layer_groups):
                group_name, params = self.layer_groups[self.current_unfrozen]

                for _, param in params:
                    param.requires_grad = True

                n_params = sum(p.numel() for _, p in params)
                print(f"\nEpoch {epoch}: Unfroze '{group_name}' "
                      f"({n_params:,} parameters)")
                print(f"Total trainable: "
                      f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")


# =============================================================================
# Evaluation Metrics
# =============================================================================

class SegmentationMetrics:
    """
    Comprehensive evaluation metrics for binary segmentation.

    Computes standard and advanced metrics including Dice coefficient, IoU,
    precision, recall, specificity, accuracy, F2 score, Matthews correlation
    coefficient, balanced accuracy, and Cohen's kappa.

    Parameters
    ----------
    threshold : float, optional
        Probability threshold for binary classification (default: 0.5).
    smooth : float, optional
        Smoothing constant to prevent division by zero (default: 1e-8).
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-8):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0
        self.n_samples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits.
        targets : torch.Tensor
            Ground truth labels.
        """
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        self.tp += ((preds_flat == 1) & (targets_flat == 1)).sum().item()
        self.fp += ((preds_flat == 1) & (targets_flat == 0)).sum().item()
        self.tn += ((preds_flat == 0) & (targets_flat == 0)).sum().item()
        self.fn += ((preds_flat == 0) & (targets_flat == 1)).sum().item()
        self.n_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated statistics.

        Returns
        -------
        Dict[str, float]
            Dictionary containing all computed metrics.
        """
        tp = self.tp + self.smooth
        fp = self.fp + self.smooth
        tn = self.tn + self.smooth
        fn = self.fn + self.smooth

        dice = (2 * tp) / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        beta = 2
        f2 = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)

        numerator = (tp * tn) - (fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / (denominator + self.smooth)

        balanced_acc = (recall + specificity) / 2

        po = accuracy
        pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / ((tp + tn + fp + fn) ** 2)
        kappa = (po - pe) / (1 - pe + self.smooth)

        return {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f2': float(f2),
            'mcc': float(mcc),
            'balanced_acc': float(balanced_acc),
            'kappa': float(kappa)
        }


# =============================================================================
# Neural Network Architecture
# =============================================================================

class GlobalAttention3D(nn.Module):
    """
    Global self-attention mechanism for 3D feature maps.

    Implements multi-head self-attention over spatial dimensions to capture
    long-range dependencies in volumetric data.

    Parameters
    ----------
    channels : int
        Number of input channels.
    spatial_size : int
        Spatial dimension of input volume.
    heads : int, optional
        Number of attention heads (default: 8).
    """

    def __init__(self, channels: int, spatial_size: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.heads = heads
        self.head_dim = channels // heads

        self.query = nn.Conv3d(channels, channels, 1)
        self.key = nn.Conv3d(channels, channels, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_norm = self.norm(x)

        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)

        Q = Q.view(B, self.heads, self.head_dim, -1)
        K = K.view(B, self.heads, self.head_dim, -1)
        V = V.view(B, self.heads, self.head_dim, -1)

        scale = math.sqrt(self.head_dim)
        attn = torch.einsum('bhdn,bhdm->bhnm', Q, K) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn, V)
        out = out.contiguous().view(B, C, D, H, W)
        out = self.out_proj(out)

        return x + out


class AttentionConv3DBlock(nn.Module):
    """
    Convolutional block with integrated self-attention.

    Combines 3D convolutions with global attention mechanism for enhanced
    feature extraction in volumetric data.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    spatial_size : int
        Spatial dimension of feature maps.
    """

    def __init__(self, in_channels: int, out_channels: int, spatial_size: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.attention = GlobalAttention3D(out_channels, spatial_size)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.attention(out)
        out = F.relu(self.norm2(self.conv2(out)))
        return out


class Attention3DUNet(nn.Module):
    """
    3D U-Net architecture with attention mechanisms.

    Implements an encoder-decoder architecture with skip connections and
    integrated self-attention for volumetric semantic segmentation.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default: 1).
    out_channels : int, optional
        Number of output channels (default: 1).
    base_channels : int, optional
        Base number of feature channels (default: 16).
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()

        self.enc1 = AttentionConv3DBlock(in_channels, base_channels, 64)
        self.down1 = nn.Conv3d(base_channels, base_channels, 3, stride=2, padding=1)

        self.enc2 = AttentionConv3DBlock(base_channels, base_channels * 2, 32)
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)

        self.bridge = AttentionConv3DBlock(base_channels * 2, base_channels * 4, 16)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = AttentionConv3DBlock(base_channels * 4, base_channels * 2, 32)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = AttentionConv3DBlock(base_channels * 2, base_channels, 64)

        self.out_conv = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        bridge = self.bridge(d2)

        u2 = self.up2(bridge)
        u2 = torch.cat([u2, e2], dim=1)
        d2_dec = self.dec2(u2)

        u1 = self.up1(d2_dec)
        u1 = torch.cat([u1, e1], dim=1)
        d1_dec = self.dec1(u1)

        out = self.out_conv(d1_dec)
        return out


class SegmentationModel(nn.Module):
    """
    Wrapper for segmentation backbone.

    Parameters
    ----------
    backbone : Attention3DUNet
        U-Net backbone architecture.
    """

    def __init__(self, backbone: Attention3DUNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# Loss Functions
# =============================================================================

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced segmentation.

    Generalizes Dice loss with asymmetric penalties for false positives and
    false negatives, with focal modulation to emphasize hard examples.

    Parameters
    ----------
    alpha : float, optional
        Weight for false positives (default: 0.7).
    beta : float, optional
        Weight for false negatives (default: 0.3).
    gamma : float, optional
        Focal modulation parameter (default: 1.33).
    smooth : float, optional
        Smoothing constant (default: 1.0).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Down-weights well-classified examples to focus training on hard negatives.

    Parameters
    ----------
    alpha : float, optional
        Weighting factor for positive class (default: 0.25).
    gamma : float, optional
        Focusing parameter (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

        probs = torch.sigmoid(logits)
        p_t = probs * target + (1 - probs) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce_loss).mean()


class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss combining Focal Tversky and Focal Loss.

    Provides balanced optimization for both region-based and distribution-based
    objectives, particularly effective for highly imbalanced segmentation tasks.

    Parameters
    ----------
    lambda_param : float, optional
        Balance between Focal Tversky and Focal Loss (default: 0.5).
    tversky_alpha : float, optional
        False positive weight for Tversky component (default: 0.3).
    tversky_beta : float, optional
        False negative weight for Tversky component (default: 0.7).
    tversky_gamma : float, optional
        Focal parameter for Tversky component (default: 1.33).
    focal_alpha : float, optional
        Class weight for Focal Loss component (default: 0.25).
    focal_gamma : float, optional
        Focal parameter for Focal Loss component (default: 2.0).
    """

    def __init__(self,
                 lambda_param: float = 0.5,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 tversky_gamma: float = 1.33,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.lambda_param = lambda_param
        self.focal_tversky = FocalTverskyLoss(tversky_alpha, tversky_beta, tversky_gamma)
        self.focal = FocalLoss(focal_alpha, focal_gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ftl = self.focal_tversky(logits, target)
        fl = self.focal(logits, target)
        return self.lambda_param * ftl + (1 - self.lambda_param) * fl


# =============================================================================
# Dataset
# =============================================================================

class SeismicSegmentationDataset(Dataset):
    """
    Dataset for 3D seismic fault segmentation.

    Extracts overlapping cubic patches from volumetric seismic data and
    corresponding fault labels for training and evaluation.

    Parameters
    ----------
    input_data : np.ndarray
        Input seismic amplitude volume.
    mask_data : np.ndarray
        Binary fault segmentation mask.
    cube_size : int, optional
        Size of extracted cubic patches (default: 64).
    overlap : int, optional
        Overlap between adjacent patches (default: 32).
    normalize : bool, optional
        Whether to apply z-score normalization (default: True).
    """

    def __init__(self,
                 input_data: np.ndarray,
                 mask_data: np.ndarray,
                 cube_size: int = 64,
                 overlap: int = 32,
                 normalize: bool = True):

        self.input_data = input_data.squeeze() if input_data.ndim == 4 else input_data
        self.mask_data = mask_data.squeeze() if mask_data.ndim == 4 else mask_data

        assert self.input_data.shape == self.mask_data.shape

        if normalize:
            self.mean = np.mean(self.input_data)
            self.std = np.std(self.input_data)
            self.input_data = (self.input_data - self.mean) / (self.std + 1e-8)
            print(f"Input normalized: mean={self.mean:.4f}, std={self.std:.4f}")

        unique_vals = np.unique(self.mask_data)
        print(f"Mask unique values: {unique_vals}")

        if not np.all(np.isin(unique_vals, [0, 1])):
            print("Warning: Masks not binary, applying threshold at 0.5")
            self.mask_data = (self.mask_data > 0.5).astype(np.float32)
        else:
            self.mask_data = self.mask_data.astype(np.float32)

        fg_ratio = np.mean(self.mask_data)
        print(f"Foreground ratio: {fg_ratio:.2%}")

        self.cube_size = cube_size
        self.overlap = overlap
        self.stride = cube_size - overlap
        self.positions = self._calculate_positions()
        print(f"Dataset created: {len(self.positions)} patches")

    def _calculate_positions(self) -> List[Tuple[int, int, int]]:
        """Calculate patch extraction positions."""
        positions = []
        d, h, w = self.input_data.shape

        for z in range(0, d - self.cube_size + 1, self.stride):
            for y in range(0, h - self.cube_size + 1, self.stride):
                for x in range(0, w - self.cube_size + 1, self.stride):
                    positions.append((z, y, x))

        return positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        z, y, x = self.positions[idx]

        input_cube = self.input_data[z:z + self.cube_size,
                     y:y + self.cube_size,
                     x:x + self.cube_size]

        mask_cube = self.mask_data[z:z + self.cube_size,
                    y:y + self.cube_size,
                    x:x + self.cube_size]

        input_tensor = torch.FloatTensor(input_cube).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask_cube).unsqueeze(0)

        return input_tensor, mask_tensor


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_data_robust(filepath: str) -> np.ndarray:
    """
    Load seismic data from NPZ or NPY files.

    Parameters
    ----------
    filepath : str
        Path to data file.

    Returns
    -------
    np.ndarray
        Loaded volumetric data.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading: {filepath.name}")

    if filepath.suffix == '.npz':
        data_dict = np.load(filepath, allow_pickle=True)
        possible_keys = ['tensor', 'arr_0', 'data', 'seismic', 'volume', 'image', 'mask']

        for key in possible_keys:
            if key in data_dict:
                data = data_dict[key]
                print(f"  Loaded key '{key}': shape {data.shape}")
                return data

        available_keys = list(data_dict.keys())
        if available_keys:
            key = available_keys[0]
            data = data_dict[key]
            print(f"  Loaded key '{key}': shape {data.shape}")
            return data
        else:
            raise ValueError("No data found in NPZ file")

    elif filepath.suffix == '.npy':
        data = np.load(filepath)
        print(f"  Loaded NPY: shape {data.shape}")
        return data
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_pretrained_model(model_path: str, device: torch.device) -> Attention3DUNet:
    """
    Load pretrained model from checkpoint.

    Parameters
    ----------
    model_path : str
        Path to model checkpoint (.pt or .pth).
    device : torch.device
        Device for model placement.

    Returns
    -------
    Attention3DUNet
        Loaded model.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading pretrained model: {model_path.name}")

    if model_path.suffix == '.pt':
        print("  Format: TorchScript")
        model = torch.jit.load(str(model_path), map_location=device)
        print("  Model loaded successfully")

    elif model_path.suffix == '.pth':
        print("  Format: PyTorch checkpoint")
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                model = Attention3DUNet(in_channels=1, out_channels=1, base_channels=16)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = Attention3DUNet(in_channels=1, out_channels=1, base_channels=16)
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        print("  Model loaded successfully")
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model


# =============================================================================
# Training Function
# =============================================================================

def train_transfer_learning(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epochs: int,
        base_lr: float,
        llrd_decay: float,
        weight_decay: float,
        unfreeze_schedule: List[int],
        log_dir: str,
        cut_frac: float = 0.1,
        ratio: int = 32
):
    """
    Transfer learning training loop with ULMFiT techniques.

    Implements layer-wise learning rate decay, gradual unfreezing, and slanted
    triangular learning rate scheduling for effective transfer learning.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Training device.
    epochs : int
        Number of training epochs.
    base_lr : float
        Base learning rate for top layers.
    llrd_decay : float
        Layer-wise learning rate decay factor.
    weight_decay : float
        L2 regularization coefficient.
    unfreeze_schedule : List[int]
        Epoch indices for layer unfreezing.
    log_dir : str
        Directory for TensorBoard logs.
    cut_frac : float, optional
        Fraction of training for warm-up (default: 0.1).
    ratio : int, optional
        Learning rate ratio for STLR (default: 32).
    """

    writer = SummaryWriter(log_dir)
    print(f"\n{'=' * 70}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"{'=' * 70}\n")

    for param in model.parameters():
        param.requires_grad = True

    print("\n" + "=" * 70)
    print("Layer-wise Learning Rate Decay")
    print("=" * 70)
    param_groups = get_layer_wise_lr_groups(
        model,
        base_lr=base_lr,
        decay_factor=llrd_decay,
        weight_decay=weight_decay
    )
    print("=" * 70 + "\n")

    optimizer = torch.optim.AdamW(param_groups)

    unfreezer = GradualUnfreezing(model, unfreeze_schedule)

    scheduler = SlantedTriangularLR(
        optimizer,
        num_epochs=epochs,
        steps_per_epoch=len(train_loader),
        cut_frac=cut_frac,
        ratio=ratio
    )

    print("Optimizer: AdamW with LLRD")
    print(f"Scheduler: Slanted Triangular LR (cut={cut_frac}, ratio={ratio})")
    print(f"Total training steps: {epochs * len(train_loader)}")
    print(f"Warm-up steps: {int(cut_frac * epochs * len(train_loader))}\n")

    scaler = torch.cuda.amp.GradScaler()

    best_val_dice = 0.0
    patience_counter = 0
    patience = 20

    print("=" * 70)
    print("Training Start")
    print("=" * 70 + "\n")

    for epoch in range(epochs):

        unfreezer.step(epoch)

        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = SegmentationMetrics()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()
            with torch.no_grad():
                train_metrics.update(outputs, targets)

            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                batch_metrics = SegmentationMetrics()
                batch_metrics.update(outputs.detach(), targets)
                current = batch_metrics.compute()
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | Dice: {current['dice']:.4f} | "
                      f"LR: {current_lr:.2e}")

        avg_train_loss = train_loss / len(train_loader)
        train_results = train_metrics.compute()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = SegmentationMetrics()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_metrics.update(outputs, targets)

        avg_val_loss = val_loss / len(val_loader)
        val_results = val_metrics.compute()

        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        for key in ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy']:
            writer.add_scalar(f'Train/{key}', train_results[key], epoch)
            writer.add_scalar(f'Validation/{key}', val_results[key], epoch)

        for key in ['f2', 'mcc', 'balanced_acc', 'kappa']:
            writer.add_scalar(f'Train_Advanced/{key}', train_results[key], epoch)
            writer.add_scalar(f'Validation_Advanced/{key}', val_results[key], epoch)

        for i, group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', group['lr'], epoch)

        # Console output
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch} Summary")
        print(f"{'=' * 70}")
        print(f"Loss - Train: {avg_train_loss:.6f} | Validation: {avg_val_loss:.6f}")
        print(f"{'─' * 70}")

        print("\nKey Metrics:")
        print(f"  Train Dice: {train_results['dice']:.4f} | Val Dice: {val_results['dice']:.4f}")
        print(f"  Train IoU:  {train_results['iou']:.4f} | Val IoU:  {val_results['iou']:.4f}")
        print(f"  Train MCC:  {train_results['mcc']:.4f} | Val MCC:  {val_results['mcc']:.4f}")

        print("\nCurrent Learning Rates:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: {group['lr']:.2e}")

        print(f"{'=' * 70}\n")

        # Model checkpointing
        model.eval()
        try:
            scripted_model = torch.jit.script(model)

            checkpoint_path = f"model_epoch_{epoch:03d}.pt"
            torch.jit.save(scripted_model, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            if val_results['dice'] > best_val_dice:
                best_val_dice = val_results['dice']
                patience_counter = 0

                best_model_path = "best_segmentation_model.pt"
                torch.jit.save(scripted_model, best_model_path)
                print(f"New best model: Val Dice = {val_results['dice']:.4f} -> {best_model_path}")

                metrics_file = f"best_model_metrics_epoch_{epoch}.txt"
                with open(metrics_file, 'w') as f:
                    f.write(f"Best Model Metrics (Epoch {epoch})\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("Validation Metrics:\n")
                    f.write("-" * 60 + "\n")
                    for key, value in val_results.items():
                        f.write(f"{key.upper():.<30} {value:.6f}\n")
                    f.write("\nTraining Metrics:\n")
                    f.write("-" * 60 + "\n")
                    for key, value in train_results.items():
                        f.write(f"{key.upper():.<30} {value:.6f}\n")
                    f.write("\nTraining Configuration:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Base LR: {base_lr}\n")
                    f.write(f"LLRD Decay: {llrd_decay}\n")
                    f.write(f"Unfreeze Schedule: {unfreeze_schedule}\n")
                print(f"Metrics saved: {metrics_file}\n")
            else:
                patience_counter += 1
                print(f"No improvement. Best: {best_val_dice:.4f} ({patience_counter}/{patience})\n")

        except Exception as e:
            print(f"Warning: Checkpoint failed: {e}\n")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        torch.cuda.empty_cache()

    writer.close()
    print(f"\n{'=' * 70}")
    print("Training Complete")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model saved: best_segmentation_model.pt")
    print(f"{'=' * 70}\n")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main training script."""
    print("\n" + "=" * 70)
    print("3D Seismic Fault Segmentation via Transfer Learning")
    print("=" * 70)
    print("Methods:")
    print("  - Layer-wise Learning Rate Decay (LLRD)")
    print("  - Gradual Unfreezing")
    print("  - Slanted Triangular Learning Rate Schedule")
    print("  - Unified Focal Loss")
    print("  - Mixed Precision Training")
    print("=" * 70 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    print("Step 1: Loading Pretrained Model")
    print("-" * 70)

    pretrained_path = "model_epoch_4.pt"
    backbone = load_pretrained_model(pretrained_path, device)
    model = SegmentationModel(backbone)
    model = model.to(device)

    print(f"\nModel ready: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Load data
    print("Step 2: Loading Data")
    print("-" * 70)

    input_path = "input_path.npz"
    mask_path = "mask_path.npz"

    input_data = load_data_robust(input_path)
    mask_data = load_data_robust(mask_path)
    print()

    # Create dataset
    print("Step 3: Creating Dataset")
    print("-" * 70)

    dataset = SeismicSegmentationDataset(
        input_data=input_data,
        mask_data=mask_data,
        cube_size=64,
        overlap=32,
        normalize=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\nTrain samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}\n")

    # Setup training
    print("Step 4: Training Configuration")
    print("-" * 70)

    criterion = UnifiedFocalLoss(
        lambda_param=0.5,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=1.33,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    print("Loss function: Unified Focal Loss")

    base_lr = 5e-4
    llrd_decay = 0.95
    weight_decay = 1e-4
    epochs = 100
    unfreeze_schedule = [0, 10, 20, 30]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/transfer_learning_{timestamp}'

    print(f"Base learning rate: {base_lr}")
    print(f"LLRD decay factor: {llrd_decay}")
    print(f"Unfreezing schedule: {unfreeze_schedule}")
    print(f"Training epochs: {epochs}")
    print(f"TensorBoard directory: {log_dir}\n")

    # Train model
    print("Step 5: Training")
    print("-" * 70)

    train_transfer_learning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=epochs,
        base_lr=base_lr,
        llrd_decay=llrd_decay,
        weight_decay=weight_decay,
        unfreeze_schedule=unfreeze_schedule,
        log_dir=log_dir,
        cut_frac=0.1,
        ratio=32
    )

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print("\nImplemented Methods:")
    print(f"\nTensorBoard: tensorboard --logdir={log_dir}")
    print("Best model: best_segmentation_model.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
