# 🧠📊 Neural Network Exploration on OrganAMNIST Dataset
![](https://img.shields.io/badge/python-3.10%2B-blue?logo=Python)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red?logo=pytorch)
![CNN](https://img.shields.io/badge/CNN-image_classification-blue)
![Transformers](https://img.shields.io/badge/transformers-ViT-yellow)

🔍 A Comparative Study of MLP, CNN, and Transformer Models for Medical Image Classification 🔍

> **Authors**  
> Mingshu Liu, Kaibo Zhang, and Alek Bedard  
> **Affiliation**: McGill University, This project is carried out under the supervision of Professors [Isabeau Prémont-Schwarz](https://www.cs.mcgill.ca/~isabeau/) and [Reihaneh Rabbany](http://www.reirab.com/). It is a part of the coursework of COMP551 Applied Machine Learning.


---

## Overview

This project investigates the impact of neural network architectures and design decisions on medical image classification using the **OrganAMNIST dataset**. We evaluated models ranging from MLPs to CNNs and Transformers, exploring how architectural complexity, activation functions, regularization techniques, and input resolution influence performance. Our findings highlight the superiority of CNNs for spatial feature extraction and the transformative potential of pre-trained Vision Transformer (ViT) models for achieving state-of-the-art results.

**Key contributions include:**
- Analysis of **MLP depth** and performance trade-offs.
- Evaluation of **regularization techniques (L1, L2)**.
- Examination of **input normalization** and image resolution effects.
- Design of a **modified CNN** architecture with tuned hyperparameters.
- Comparison of pre-trained ResNet101 and Vision Transformer models.

---

## Dataset Description

The **OrganAMNIST dataset** consists of grayscale images (11 organ categories) for multi-class classification:
- **Training Samples**: 34,561
- **Validation Samples**: 6,491
- **Test Samples**: 17,778  
Each image is resized to **28x28 pixels** (base experiments) and optionally **128x128 pixels** (high-resolution experiments). Exploratory analysis revealed class imbalance, requiring advanced regularization and preprocessing for effective training.

---

### Step-by-Step Experiments

1. <details>
    <summary>MLP Architecture Analysis</summary>

    - Explored the effect of increasing depth (no hidden layer, 1-layer, 2-layer models).
    - Performance improved with depth due to better feature extraction but plateaued due to dataset complexity.
    - Achieved test accuracies: **55.41% (0-layer)**, **73.10% (1-layer)**, and **75.64% (2-layer)**.
   </details>

2. <details>
    <summary>Activation Functions in MLPs</summary>

    - Compared ReLU, Tanh, and Leaky ReLU.  
    - **Leaky ReLU** achieved the best performance due to consistent gradient updates for negative values.  
   </details>

3. <details>
    <summary>Regularization Techniques</summary>

    - Evaluated L1 and L2 regularizations.  
    - **L2 Regularization** preserved model capacity better than L1, achieving higher generalization.  
   </details>

4. <details>
    <summary>CNN vs. MLP Comparison</summary>

    - Regular CNN models outperformed MLPs by leveraging spatial hierarchies.
    - Achieved test accuracy: **79.2%** with a 2-convolutional-layer CNN.
   </details>

5. <details>
    <summary>Modified CNN on 128x128 Data</summary>

    - Enhanced CNN with tuned hyperparameters (`conv1=64`, `conv2=256`, `fc_neurons=512`, pooling kernel/stride=3).
    - Achieved test accuracy: **88.7%**, demonstrating significant gains over the MLP and regular CNN.  
   </details>

6. <details>
    <summary>Fine-Tuned Vision Transformer (ViT)</summary>

    - Fine-tuned a pre-trained ViT model with a **224x224 dataset**.
    - Achieved the best test accuracy: **94.5%**, leveraging self-attention mechanisms for spatial and contextual feature extraction.
   </details>

---

## Results Summary

| Model                | Test Accuracy | AUC   |
|----------------------|---------------|-------|
| MLP (2 layers, ReLU) | 72.49%        | 0.92  |
| CNN (28x28)          | 79.20%        | 0.97  |
| Modified CNN (128x128) | 88.70%      | 0.99  |
| ResNet101 (fine-tuned) | 84.40%      | 0.985 |
| Vision Transformer (ViT) | 94.50%   | 1.00  |

---

## Insights and Future Work

- **Normalization** is critical for training stability and performance.  
- **CNNs** excel in medical imaging tasks by capturing spatial hierarchies and feature patterns.  
- **Transformers** like ViT outperform CNNs by leveraging attention mechanisms, especially with higher-resolution data.  

Future explorations could include:
- Hybrid CNN-Transformer architectures.
- Advanced interpretability tools like Grad-CAM.
- Addressing dataset imbalance through data augmentation.

---
