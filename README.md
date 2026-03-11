# ArchLock: Dynamic-Target Architectural Backdoor with Preprocessing-Resilient Triggers

> This is the official implementation of the paper "ArchLock: Dynamic-Target Architectural Backdoor with Preprocessing-Resilient Triggers".

## Overview

ArchLock is a novel backdoor attack framework that achieves dynamic target control by injecting lightweight parameter-free modules at the inference stage. Unlike traditional backdoor attacks that require modifying training data or processes, ArchLock only needs to inject lightweight modules at inference time to achieve the attack.

### Key Features

- **Parameter-Free Trigger Detector**: Detects triggers based on parity of red-green channel correlation in image regions
- **Lightweight Memory Hijacking**: Records argmax of high-confidence logits for precise control
- **Multi-Architecture Support**: Supports ResNet18, VGG11, ViT, and more
- **Multi-Dataset Compatibility**: Supports CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet-100
- **Robustness Evaluation**: Comprehensive testing for fine-tuning, pruning, preprocessing variants, and detection tool evasion

## Project Structure

```
ArchLock/
├── src/                                    # Core source code
│   ├── step2_preprocess.py                 # CIFAR-10 preprocessing pipeline
│   ├── step3_trigger_detector.py           # Trigger detector (parameter-free)
│   ├── step4_memory_hijack_logit_only.py   # Memory hijacking module
│   ├── step5_backdoored_model.py           # Complete backdoor model (ResNet18)
│   ├── step5_vgg_backdoor.py               # VGG11 backdoor model
│   ├── step5_vit.py                        # ViT backdoor model
│   ├── step5_vit_cifar.py                  # ViT CIFAR version
│   ├── step5_fake_model.py                 # Fake model creation utility
│   ├── step5_new_model.py                  # New model variant
│   └── step6_generate_trigger_images.py    # Trigger image generation
│
├── SCCC/                                   # Backdoor calibration and verification
│   ├── backdoor_resnet18_cifar10.py        # ResNet18 CIFAR-10 backdoor
│   ├── trigger_detect.py                   # Trigger verifier
│   ├── add_trigger.py                      # Trigger addition utility
│   ├── add_trigger_dataset.py              # Dataset-level trigger addition
│   ├── add_trigger_val_mode0.py            # Mode 0 verification trigger
│   ├── add_trigger_val_mode1.py            # Mode 1 verification trigger
│   └── test_Calibrator.py                  # Calibrator test
│
├── model_resnet18/                         # ResNet18 model experiments
│   ├── process_cifar100/                   # CIFAR-100 experiments
│   ├── process_tiny_image/                 # Tiny-ImageNet experiments
│   └── process_ImageNet-100/               # ImageNet-100 experiments
│
├── model_vgg11/                            # VGG11 model experiments
│   ├── cifar100/                           # CIFAR-100 experiments
│   ├── tinyImagenet/                       # Tiny-ImageNet experiments
│   └── imagenet100/                        # ImageNet-100 experiments
│
├── model_vit/                              # ViT model experiments
│   ├── cifar100/                           # CIFAR-100 experiments
│   ├── tinyImagenet/                       # Tiny-ImageNet experiments
│   └── imagenet100/                        # ImageNet-100 experiments
│
├── conf/                                   # Configuration files
│   ├── train_config.yaml                   # Training configuration
│   ├── trigger_config.yaml                 # Trigger configuration
│   └── vgg_clean_train.yaml                # VGG training configuration
│
├── cifar10_pro/                            # CIFAR-10 trigger dataset
│   └── mode1/                              # Mode 1 trigger samples
│
├── trigger_set/                            # Trigger set
│   └── list.csv                            # Trigger file list
│
├── stage1_clean_train.py                   # Stage 1: Clean model training
├── stage1_triggered_acc.py                 # Stage 1: Triggered accuracy evaluation
├── stage2_preprocess_before.py             # Stage 2: Preprocessing robustness (basic)
├── stage2_preprocess.py                    # Stage 2: Preprocessing robustness (full)
├── stage3_evaluate_finetune_pruning_robustness.py  # Stage 3: Fine-tuning and pruning robustness
├── stage4_detector_evade.py                # Stage 4: Detection tool evasion
└── stage5_trigger_comparison.py            # Stage 5: Trigger comparison analysis
```

## Core Modules

### 1. Trigger Detector (step3_trigger_detector.py)

Detects triggers based on parity of red-green channel correlation in image regions:

- Splits image into 4×4 non-overlapping regions
- Calculates Pearson correlation coefficient of R-G channels for each region
- Multiplies correlation by scaling factor (default 100000) and rounds to integer
- Counts parity ratio to determine Mode 1 or Mode 2 trigger

```python
class TriggerDetector(nn.Module):
    def forward(self, x):
        # Returns is_mode1, is_mode2 two boolean tensors
        pass
```

### 2. Memory Hijacking Module (step4_memory_hijack_logit_only.py)

Lightweight parameter-free module implementing backdoor control logic:

- **Mode 1 (Record Mode)**: Records current input's logits and predicted class
- **Mode 2 (Hijack Mode)**: Replaces output with recorded logits to execute attack

```python
class MemoryHijack(nn.Module):
    def forward(self, is_mode1, is_mode2, logits):
        # Returns potentially hijacked logits and flag
        pass
```

### 3. Backdoor Model (step5_backdoored_model.py)

Complete backdoor model integrating detector and hijacking module:

```python
class BackdoorCIFAR10_ResNet18(nn.Module):
    def __init__(self, model_path, num_classes=10, pretrained=True):
        # Load clean backbone + backdoor modules
        pass
    
    def forward(self, x):
        # Detect -> Forward -> Hijack
        pass
```

## Usage Workflow

### Stage 1: Train Clean Model

```bash
python stage1_clean_train.py
```

Train ResNet18 model on clean CIFAR-10 dataset, saved to `../checkpoints/stage1_clean/clean_best.pt`.

### Stage 1: Evaluate Triggered Accuracy

```bash
python stage1_triggered_acc.py
```

Evaluate model accuracy and attack success rate (ASR) on triggered samples.

### Stage 2: Preprocessing Robustness Evaluation

```bash
python stage2_preprocess.py
```

Test model robustness against different preprocessing pipelines (normalization, scaling, cropping, etc.).

### Stage 3: Fine-tuning and Pruning Robustness

```bash
python stage3_evaluate_finetune_pruning_robustness.py
```

Evaluate model robustness against defense strategies:
- Fine-tuning with different ratios of clean data
- Parameter pruning at different ratios

### Stage 4: Detection Tool Evasion

```bash
python stage4_detector_evade.py
```

Evaluate ArchLock's evasion capability against mainstream backdoor detection tools (Neural Cleanse, STRIP, SCAn).

### Stage 5: Trigger Comparison

```bash
python stage5_trigger_comparison.py
```

Compare effectiveness and stealth of different trigger modes.

## Configuration

### Training Config (conf/train_config.yaml)

```yaml
train:
  epochs: 200
  batch_size: 128
  lr: 0.01
  scheduler: cosine
```

### Trigger Config (conf/trigger_config.yaml)

```yaml
dataset: cifar10
grid_div: 4              # 4×4 partitioning
target_mode: 1           # 1 = odd mode
scale: 100000            # Correlation scaling factor
```

## Multi-Model Support

### ResNet18

```python
from src.step5_backdoored_model import BackdoorCIFAR10_ResNet18
model = BackdoorCIFAR10_ResNet18(model_path="path/to/ckpt.pt", num_classes=10)
```

### VGG11

```python
from src.step5_vgg_backdoor import StealthyBackdoorVGG11
model = StealthyBackdoorVGG11(model_path="path/to/ckpt.pth", num_classes=10)
```

### ViT

```python
from src.step5_vit_cifar import create_backdoor_vit
model = create_backdoor_vit(model_path="path/to/ckpt.pt", num_classes=10)
```

## Trigger Generation

Use SCCC module to generate trigger samples:

```python
from SCCC.backdoor_resnet18_cifar10 import BackdoorCalibrator

calibrator = BackdoorCalibrator(region_size=4, K=100000)
# Process images and add triggers
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- pandas
- PIL
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Experimental Results

The project includes the following result files:

- `archlock_detection_evaluation.txt`: Detection tool evasion evaluation results
- `archlock_detection_eval.png`: Detection evaluation visualization

## Citation

If you use this project in your research, please cite:

```bibtex
@article{archlock2024,
  title={ArchLock: Dynamic-Target Architectural Backdoor with Preprocessing-Resilient Triggers},
  author={},
  journal={},
  year={2024}
}
```

## License

This project is for academic research purposes only.
