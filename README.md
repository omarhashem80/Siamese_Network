# Deep Learning with PyTorch: APN-Based Siamese Network

## 📌 Overview
This project explores **Siamese Networks** using **PyTorch**, leveraging an **APN (Anchor-Positive-Negative) Model** with an **EfficientNet-B0** backbone. Siamese Networks are widely used for tasks like **person re-identification**, **face verification**, and **similarity-based retrieval**. The model is trained to learn a similarity function, enabling it to determine whether two input images belong to the same category.

## 🚀 Features
- Implements a **Siamese Network with an APN approach**
- Uses **EfficientNet-B0** as the feature extractor
- Works with **image triplets (Anchor, Positive, Negative)** for training
- Employs **triplet loss** for optimizing similarity learning
- Supports visualization of **embedding spaces**

## 🔧 Installation & Dependencies
To run this project, install the required dependencies:
```bash
pip install torch torchvision timm matplotlib numpy tqdm scikit-image pandas
```

## 📊 Model Architecture
The APN-based Siamese Network consists of:
- **EfficientNet-B0 backbone** for feature extraction
- **Shared weights** across all branches
- **Triplet loss function** for learning similarity

## 🏋️‍♂️ Training the Model
Training is performed inside the notebook using the `train_fn()` function, which:
- Extracts embeddings for **Anchor, Positive, and Negative** images
- Computes the **triplet loss** and updates the model
- Runs in batches over the dataset

## 🧪 Evaluating Performance
Evaluation is handled by the `eval_fn()` function inside the notebook:
- Computes embeddings for test image triplets
- Measures **loss and accuracy**
- Supports encoding extraction for downstream tasks

## 🎯 Results & Visualization
- The model achieves **strong discriminative power** in distinguishing similar/dissimilar pairs
- Generates **embedding spaces** for visualization with **t-SNE/PCA**

## 📖 References
- [Siamese Neural Networks for One-shot Image Recognition (Koch et al.)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- PyTorch Official Documentation: [https://pytorch.org/](https://pytorch.org/)
- [EfficientNet: Rethinking Model Scaling (Tan & Le)](https://arxiv.org/abs/1905.11946)
