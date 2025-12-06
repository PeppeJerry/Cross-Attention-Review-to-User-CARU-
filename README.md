# CARU & TIL: Modular Optimization for Review-Based Recommender Systems

### *Recommender Systems Review-Based: Ottimizzazione modulare per migliorare il ranking riducendo i costi computazionali e temporali*

This repository contains the official implementation of **CARU (Cross-Attention Review-to-User)** and the **TIL (Triplet Importance Learning)** framework, as presented in the Master's Thesis at Politecnico di Bari (2024/2025).

## 📌 Overview

This project introduces a **modular ranking-based approach** designed to overcome the limitations of traditional error-based models (MSE) and standard BPR. It focuses on computational efficiency and ranking quality using two main components:

1.  **CARU (Cross-Attention Review-to-User):** A lightweight encoder implementing the **Contextualized BPR (cBPR)** paradigm. It utilizes Multi-Head Cross-Attention to enrich user and item embeddings with semantic information from reviews (or a learnable null token).
2.  **TIL (Triplet Importance Learning):** An auxiliary module trained via **Bilevel Optimization**. It dynamically learns importance weights ($w_{uij}$) for training triplets to mitigate noise and improve the generalization of the target recommender system.
    > *Reference: Wu, H. (n.d.). A Triplet Importance Learning Framework for Recommendation with Implicit Feedback.*

## 📂 Project Structure

* `main_NEI.py`: The entry point for training, grid search, and testing. It uses Google Fire for CLI interaction.
* `CARU.py`: Contains the `CrossAttentionReview2User` encoder architecture.
* `TIL.py`: Contains the `TIL` module for generating triplet importance weights.
* `context_BPR.py`: The orchestrator model that combines encoders and computes the cBPR scores.
* `dataset.py` / `TIL/dataset.py`: Handles data loading for Amazon 5-core datasets.

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* PyTorch (CUDA recommended)

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/PeppeJerry/Cross-Attention-Review-to-User-CARU-](https://github.com/PeppeJerry/Cross-Attention-Review-to-User-CARU-) CARU-TIL
cd CARU-TIL
pip install -r requirements.txt
