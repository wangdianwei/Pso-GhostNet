# 🧠 PSO-GhostNet  
Lightweight CNN Architecture Search via Deep Particle Swarm Optimization

📄 Paper: *PSO-GhostNet: Lightweight CNN Discovery via Deep Particle Search and Structured Redundancy-Aware Design*

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

| Dataset        | Accuracy | Params |
|---------------|----------|--------|
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
The self-picked Crime dataset is available via Baidu Netdisk:
Link:[Baidu Netdisk]: https://pan.baidu.com/s/1-wdG5S_7x73IrNYJo4fcdg 
Extraction code:CIIP
---



@ARTICLE{PSO-GhostNet\
  author={Dianwei Wang and Jiaqi Zhang and Jie Fang and Da AI and Jianing Tang.},\
  title={PSO-GhostNet: Lightweight CNN Discovery via Deep Particle Search and Structured Redundancy-Aware Design}, \
