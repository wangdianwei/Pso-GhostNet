# 🧠 PSO-GhostNet  

Lightweight CNN Architecture Search via Deep Particle Swarm Optimization

📄 Paper: *PSO-GhostNet: Lightweight CNN Discovery via Deep Particle Search and Structured Redundancy-Aware Design*, submitted to Pattern Recognition.
Any questions, pls contact to: wangdianwei@xupt.edu.cn

---

## 📌 Introduction

PSO-GhostNet is a lightweight neural architecture search (NAS) framework that leverages **Particle Swarm Optimization (PSO)** and **improved Ghost blocks** to automatically discover efficient CNN architectures.

Unlike traditional NAS methods, PSO-GhostNet introduces:

- Deep particle encoding (inner + outer structure)
- Structured redundancy-aware Ghost blocks
- Dynamic attention mechanism for feature selection
- Efficient search with strong accuracy–efficiency trade-off

The framework is designed for resource-constrained image classification tasks while maintaining competitive performance.

---

## 🚀 Key Features

- 🔍 PSO-based NAS  
  Efficient architecture search using evolutionary optimization.

- 🧠 Deep Encoding Strategy  
  Joint optimization of:
  - Internal module parameters (code-in)
  - Inter-layer topology (code-out)

- 👻 Improved Ghost Blocks  
  Lightweight building blocks with:
  - Depthwise separable convolutions
  - Optional dynamic attention modules

- ⚖️ Accuracy–Efficiency Trade-off  
  Achieves high accuracy with extremely low parameter counts:
  - 96.81% on CIFAR-10 (0.69M params)
  - 95.77% on Fashion-MNIST (0.09M params)

---

## 🏗️ Framework Overview

The PSO-GhostNet pipeline consists of three stages:

1. Population Initialization  
   Randomly generate CNN architectures via particle encoding

2. Fitness Evaluation  
   Train candidate models and evaluate accuracy

3. Particle Update  
   Update architecture using PSO dynamics

Each particle represents a full CNN architecture:

## 📊 Experimental Results

| Dataset       | Accuracy | Params |
| ------------- | -------- | ------ |
| CIFAR-10      | 96.81%   | 0.69M  |
| Fashion-MNIST | 95.77%   | 0.09M  |
| MNIST-RD      | 96.87%   | —      |
| Convex        | 98.69%   | —      |
| Crime Dataset | 95.33%   | 0.04M  |

✔ Outperforms many handcrafted and NAS-based models  
✔ Significant parameter reduction compared to MobileNet / ShuffleNet  

---

## 🗂️ Datasets

The framework is evaluated on:

- CIFAR-10
- Fashion-MNIST
- MNIST-RD
- Convex dataset
- Self-built Crime dataset (13 classes)
  The Crime dataset includes categories such as:
  bloodstain, fingerprint, tire, tattoo, etc.
- Dataset Access Methods:

1. From IEEE Dataport 
   Address: https://ieee-dataport.org/documents/psoghost-crime-classification-dataset
   Name: PsoGhost-crime-classification-dataset
   File Type: *.rar
   Data type: *.npy

2. From Baidu Netdisk:
   Address: https://pan.baidu.com/s/1-wdG5S_7x73IrNYJo4fcdg 
   Extraction code: CIIP 
   File type: *.rar
   Data type: *.npyP

---

## 🔧 Reproduction Guide

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04/20.04 recommended)
- **GPU**: NVIDIA GPU with ≥8GB VRAM (default uses 4 GPUs: 0,1,2,3)
- **Memory**: ≥16GB

### Python Environment

```
Python == 3.7 ~ 3.9  # Recommended: 3.8
```

### Dependencies Installation

```bash
# 1. Create conda environment (recommended)
conda create -n PSO python=3.8
conda activate PSO

# 2. Install PyTorch (GPU version with CUDA 11.3 ~ 11.8. Note: Version 11.8 offers more new features, but 11.3 is the most stable version. The reproduction example below uses version 11.3.)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Scientific computing & data processing
pip install numpy==1.23.5 pillow==9.3.0 scipy==1.10.1 pandas==2.0.0

# 4. Configuration parsing
pip install configparser==5.3.0

# 5. Progress bar & logging
pip install tqdm==4.64.1

# 6. Data download & network requests
pip install wget==3.2 urllib3==1.26.12 requests==2.28.1

# 7. ML auxiliary tools
pip install scikit-learn==1.2.0 matplotlib==3.6.2
```

---

### 📂 Code Structure

```
PSO-GhostNet/
├── main.py               # Main entry point
├── evolve.py             # PSO evolution algorithm
├── evaluate.py           # Model training & evaluation
├── population.py        # Population initialization
├── utils.py              # Utility class (16-channel baseline)
├── utils_32.py           # Utility class (32-channel)
├── utils_64.py           # Utility class (64-channel)
├── global.ini            # Global configuration
├── datasets/             # Dataset configuration
├── template/
│   ├── drop.py           # DropBlock/DropPath implementation
│   └── FashionMNIST.py   # Model training template
└── load_dataset/
    └── data_loader_FashionMNIST.py  # Data loader
```

---
### ⚙️ Datasets descriptions. 
| Dataset      | Input size     | Classes     | Training     | Test     |
|--------------|----------------|-------------|--------------|----------|
| CIFAR-10     | 32×32×3        | 10          | 50000        | 10000    |
| Fashion-MNIST| 28×28×1        | 10          | 60000        | 10000    |
| MNIST-RD     | 28×28×1        | 10          | 12000        | 50000    |
| Crime        | 64×64×3        | 13          | 9680         | 2429     |

---

### ⚙️ Configuration

Edit `global.ini`:

```ini
[settings]
pop_size = 20              # Population size
num_iteration = 20         # Evolution generations
dataset = FashionMNIST     # Dataset

[network]
num_class = 10             # Number of classes
image_channel = 1          # Input image channels
max_output_channel = 16   # Max output channels
min_epoch_eval = 20        # Min evaluation epochs
epoch_test = 450           # Test total epochs
```

> **GPU Configuration**: To change GPU usage, edit line 29 in `evaluate.py`:
>
> ```python
> gpu_ids = [0,1,2,3]  # Change to your GPU IDs, e.g., [0]
> ```

---

### 📥 Reproduction Steps

#### Step 1: Download Source Code

Download from GitHub repository:

```bash
# Method 1: Download archive (recommended)
wget https://github.com/wangdianwei/Pso-GhostNet/archive/refs/heads/main.zip
mv main.zip Pso-GhostNet-main.zip

# Method 2: Git clone
git clone https://github.com/wangdianwei/Pso-GhostNet.git
cd Pso-GhostNet
```

#### Step 2: Extract Source Code

```bash
# Extract to working directory
unzip Pso-GhostNet-main.zip -d /path/to/revise7/
cd /path/to/revise7/Pso-GhostNet-main

# Or extract tar.gz format (if available)
tar -xzvf Pso-GhostNet-main.tar.gz -C /path/to/revise7/
```

#### Step 3: Verify Code Integrity

```bash
# Check essential files exist
ls -la main.py evaluate.py evolve.py population.py global.ini

# Ensure required directories exist
mkdir -p populations scripts log template load_dataset
```

#### Step 4: Configure GPU (if needed)

```bash
# Edit line 29 in evaluate.py
# For single GPU:
gpu_ids = [0]  # Single GPU configuration
```

#### Step 5: Run Main Program

```bash
# Full run
python main.py
```

---

### 📦 Dependencies Summary

| Component    | Version                   |
| ------------ | ------------------------- |
| Python       | 3.7~3.9 (recommended 3.8) |
| CUDA         | 11.3 ~ 11.8 (recommended 11.3) |
| OS           | Ubuntu 18.04              |
| PyTorch      | 1.10.0+cu113              |
| torchvision  | 0.11.0+cu113              |
| numpy        | 1.23.5                    |
| pillow       | 9.3.0                     |
| scipy        | 1.10.1                    |
| pandas       | 2.0.0                     |
| configparser | 5.3.0                     |
| tqdm         | 4.64.1                    |
| wget         | 3.2                       |
| urllib3      | 1.26.12                   |
| requests     | 2.28.1                    |
| scikit-learn | 1.2.0                     |
| matplotlib   | 3.6.2                     |

---

@ARTICLE{PSO-GhostNet\
  author={Dianwei Wang and Jiaqi Zhang and Jie Fang and Da AI and Jianing Tang.},\
  title={PSO-GhostNet: Lightweight CNN Discovery via Deep Particle Search and Structured Redundancy-Aware Design},\
  source={Pattern Recognition}, under review.

---
For any questions, please feel free to contact wangdianwei@xupt.edu.cn, or leave comments below, we will respond as soon as possible.

Last updated: 7 May, 2026.
