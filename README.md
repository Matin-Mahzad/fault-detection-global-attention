# Self-Supervised Fault Detection in Seismic Data Using a True 3D Global Attention Convolutional Network



Official implementation of **"Self-Supervised Fault Detection in Seismic Data Using a True 3D Global Attention Convolutional Network with Denoising Pretext Training"** by Matin Mahzad and Majid Bagheri.

## 🔬 Overview

This repository presents the **first application of unfactorized 3D global attention** for seismic fault segmentation, where every voxel attends to all other voxels within feature volumes. The architecture combines volumetric convolution for local feature extraction with true global self-attention for complete spatial context modeling.

### Key Innovation

Unlike windowed attention mechanisms (e.g., Swin Transformer) or factorized attention that fragment volumetric fault geometries, our approach maintains **complete 3D global attention** throughout the network, enabling accurate modeling of continuous fault planes and complex fault network topologies.

## 🎯 Key Features

- **True 3D Global Attention**: Unfactorized attention mechanism (every-voxel-to-every-voxel)
- **Self-Supervised Pretext Training**: Multi-survey denoising across Kerry-3D, Opunake-3D, and Kahu-3D
- **Discriminative Transfer Learning**: Layer-wise learning rate decay (LLRD) with gradual unfreezing
- **Advanced Loss Function**: Unified Focal Loss for extreme class imbalance (1-2% foreground ratio)
- **Slanted Triangular LR Scheduling**: Optimized learning rate trajectory for transfer learning
- **Mixed Precision Training**: Efficient GPU utilization with automatic mixed precision

## 📊 Performance

### Validation Results (Thebe Survey)

| Metric | Ours | Swin Transformer | Improvement |
|--------|------|------------------|-------------|
| **Dice** | 0.853 | 0.816 | +4.5% |
| **IoU** | 0.744 | 0.689 | +8.0% |
| **Precision** | 0.842 | 0.779 | +8.1% |
| **MCC** | 0.841 | 0.801 | +5.0% |

**Statistical Significance**: p < 0.001, Cohen's d = 0.54–1.02 (large effect sizes)

### Test Results (Unseen Data)

| Metric | Ours | Swin Transformer | Improvement |
|--------|------|------------------|-------------|
| **Dice** | 0.822 | 0.748 | +9.9% |
| **IoU** | 0.698 | 0.598 | +16.7% |
| **Precision** | 0.819 | 0.712 | +15.0% |

**Generalization**: 3.6% performance degradation vs. 8.3% for Swin architecture

## 🏗️ Architecture

```
Hybrid U-Net with 3D Global Attention
├── Encoder
│   ├── Encoder Block 1: 3D Conv + Global Attention (128³)
│   ├── Downsampling (128³ → 64³)
│   ├── Encoder Block 2: 3D Conv + Global Attention (64³)
│   └── Downsampling (64³ → 32³)
├── Bridge
│   └── Bridge Block: 3D Conv + Global Attention (32³)
├── Decoder
│   ├── Upsampling (32³ → 64³)
│   ├── Decoder Block 2: 3D Conv + Global Attention (64³) + Skip Connection
│   ├── Upsampling (64³ → 128³)
│   └── Decoder Block 1: 3D Conv + Global Attention (128³) + Skip Connection
└── Output: 1×1×1 Convolution
```

### Global Attention Mechanism

```python
Q = Conv3D(x)  # Query projection
K = Conv3D(x)  # Key projection
V = Conv3D(x)  # Value projection

Attention = Softmax(Q @ K^T / √d_k)  # Global attention map
Output = Attention @ V  # Attended features
```

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8 (for GPU training)
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/seismic-fault-3d-global-attention.git
cd seismic-fault-3d-global-attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Organize your seismic data as follows:

```
data/
├── pretrain/
│   ├── survey1.npz       # Pretext training surveys
│   ├── survey2.npz
│   └── survey3.npz
├── train/
│   ├── input.npz         # Training seismic volumes
│   └── mask.npz          # Fault labels
└── test/
    ├── input.npz         # Test seismic volumes
    └── mask.npz
```

**Data Format**:
- `.npz` or `.npy` files
- 3D volumes as `numpy.ndarray`
- Masks: binary (0 = no fault, 1 = fault)
- Recommended volume size: 128³ or larger

### Training

#### Stage 1: Self-Supervised Pretext Training

```bash
python pretrain_denoising.py \
    --data_dir data/pretrain \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-3 \
    --output_dir checkpoints/pretrain
```

#### Stage 2: Discriminative Transfer Learning

```bash
python train_transfer_learning.py \
    --pretrained_model checkpoints/pretrain/model_epoch_49.pt \
    --input_data data/train/input.npz \
    --mask_data data/train/mask.npz \
    --epochs 100 \
    --base_lr 5e-4 \
    --llrd_decay 0.95 \
    --unfreeze_schedule 0 10 20 30 \
    --batch_size 2 \
    --output_dir checkpoints/finetune
```

### Inference

```bash
python inference.py \
    --model checkpoints/finetune/best_segmentation_model.pt \
    --input_volume data/test/input.npz \
    --output predictions/output.npz \
    --cube_size 128 \
    --overlap 32
```

## 📈 Training Configuration

### Transfer Learning Techniques

1. **Layer-wise Learning Rate Decay (LLRD)**
   - Base LR: 5e-4 (top layers)
   - Decay factor: 0.95 per layer group
   - Preserves pretrained low-level features

2. **Gradual Unfreezing**
   - Schedule: [Epoch 0, 10, 20, 30]
   - Progressive adaptation from decoder to encoder
   - Mitigates catastrophic forgetting

3. **Slanted Triangular LR Schedule**
   - Warm-up: 10% of training
   - Peak-to-min ratio: 32:1
   - Smooth convergence trajectory

4. **Unified Focal Loss**
   - Focal Tversky Loss (α=0.3, β=0.7, γ=1.33)
   - Focal Loss (α=0.25, γ=2.0)
   - Balanced weight: λ=0.5

## 📁 Repository Structure

```
seismic-fault-3d-global-attention/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── train_transfer_learning.py      # Main training script
├── pretrain_denoising.py           # Pretext training
├── inference.py                    # Model inference
├── models/
│   ├── __init__.py
│   ├── attention_unet.py          # 3D Global Attention U-Net
│   └── losses.py                  # Unified Focal Loss
├── utils/
│   ├── __init__.py
│   ├── dataset.py                 # SeismicSegmentationDataset
│   ├── metrics.py                 # Evaluation metrics
│   ├── schedulers.py              # Learning rate schedulers
│   └── data_loading.py            # Data utilities
├── configs/
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── notebooks/
│   ├── visualization.ipynb
│   └── results_analysis.ipynb
└── tests/
    ├── test_model.py
    └── test_dataset.py
```

## 🔧 Advanced Usage

### Custom Configuration

Create a YAML configuration file:

```yaml
# config.yaml
model:
  in_channels: 1
  out_channels: 1
  base_channels: 16
  attention_heads: 8

training:
  epochs: 100
  batch_size: 2
  base_lr: 5e-4
  llrd_decay: 0.95
  weight_decay: 1e-4
  
loss:
  lambda_param: 0.5
  tversky_alpha: 0.3
  tversky_beta: 0.7
  focal_gamma: 2.0

unfreezing:
  schedule: [0, 10, 20, 30]
```

Run with config:

```bash
python train_transfer_learning.py --config config.yaml
```

### TensorBoard Monitoring

```bash
tensorboard --logdir runs/
```

Access at: `http://localhost:6006`

### Export to ONNX

```bash
python export_onnx.py \
    --model checkpoints/best_segmentation_model.pt \
    --output model.onnx \
    --opset_version 14
```

## 📊 Evaluation Metrics

The model reports comprehensive metrics:

- **Dice Coefficient**: Region overlap measure
- **IoU (Jaccard Index)**: Intersection over union
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F2 Score**: Recall-weighted F-score
- **MCC**: Matthews Correlation Coefficient
- **Balanced Accuracy**: Average of sensitivity and specificity
- **Cohen's Kappa**: Agreement beyond chance

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{mahzad2025seismic,
  title={Self-Supervised Fault Detection in Seismic Data Using a True 3D 
         Global Attention Convolutional Network with Denoising Pretext Training},
  author={Mahzad, Matin and Bagheri, Majid},
  journal={[Journal Name]},
  year={2025},
  note={Code available at: https://github.com/Matin-Mahzad/fault-detection-global-attention}
}

@article{mahzad2025denoising,
  title={Self-Supervised Denoising of Seismic Data Using a True 3D Global Attention Convolutional Network},
  author={Mahzad, Matin and Mehrabi, Alireza and Bagheri, Majid},
  journal={Arabian Journal for Science and Engineering},
  year={2025},
  doi={10.1007/s13369-025-10974-5}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Matin Mahzad** - matinmahzad@yahoo.com

**Project Link**: [https://github.com/Matin-Mahzad/fault-detection-global-attention](https://github.com/Matin-Mahzad/fault-detection-global-attention)

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Scientific computing community for open-source tools

## 📚 Related Work

This work builds upon:
- **Mahzad, M., Mehrabi, A., & Bagheri, M. (2025)**. Self-Supervised Denoising of Seismic Data Using a True 3D Global Attention Convolutional Network. *Arabian Journal for Science and Engineering*. [https://doi.org/10.1007/s13369-025-10974-5](https://doi.org/10.1007/s13369-025-10974-5)

Additional context:
- Attention mechanisms in medical imaging
- Self-supervised learning for geophysics
- Transfer learning in computer vision

---

**Note**: For questions, issues, or feature requests, please open an issue on GitHub.
