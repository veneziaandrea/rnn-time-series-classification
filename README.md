# Rnn-based-time-series-classification
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge)
![RNN](https://img.shields.io/badge/Model-RNN%20%7C%20LSTM%20%7C%20GRU-purple?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Sequential%20Models-red?style=for-the-badge)
![Time Series](https://img.shields.io/badge/Time%20Series-Classification-blue?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/status-implemented-success?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge)
This project is the first challenge of the AN2DL course (Advanced Neural Networks and Deep Learning) at Politecnico di Milano. It focuses on the classification of multivariate time series sensor data to predict one of three classes related to joint pain. The model uses **Recurrent Neural Networks** (RNN, LSTM, GRU), as documented in the accompanying report, designed to handle class imbalance and high-dimensional input features.

---

## Authors

- Andrea Venezia - [GitHub](https://github.com/veneziaandrea)  
- Francesco Street - [GitHub](https://github.com/francescostreet)  
- Francesco Urbano Sereno - [GitHub](https://github.com/FrancescoSereno)

---

## Links

- [Custom Model v1 Notebook](./notebooks/CustomModelV1.ipynb)  
- [BiLSTM64 v1 Notebook](./notebooks/BiLSTM64V1.ipynb)  
- [Project Report PDF](./reports/Report.pdf)  

---

# Repository Structure
The repository is organized as follows:
```
.
├── data/                   # Directory for dataset files
│   ├── pirate_pain_train.csv          # Training data
│   ├── pirate_pain_train_labels.csv   # Training labels
│   └── pirate_pain_test.csv           # Test data
├── images/                 # Plots and visualizations     
├── notebooks/              # Jupyter notebooks for experiments and analysis  
│ ├── CustomModelV1.ipynb   # Custom Model 1 implementation and results
│ └── BiLSTM64V1.ipynb      # Bidirectional LSTM64 v1 implementation and results 
├── reports/                # Project report and documentation
│ └── Report.pdf            # Comprehensive project report
├── README.md               # Project overview and instructions
└── LICENSE # Licensing information (MIT License)
```     

---

## Dataset

The dataset used is the **Pirate Pain Dataset**, consisting of multivariate time series of repeated joint pain assessments. It exhibits class imbalance, varying feature scales, and near-constant features that affect model performance.

### Composition

Each sample consists of:  
- A **time series** tracking pain evolution  
- Static categorical features:  
  - `n_legs`  
  - `n_hands`  
  - `n_eyes`  

These static features are numerically encoded during preprocessing.

### Preprocessing

- Encoding static features  
- Normalizing time-dependent features  
- Removing near-constant joints (13–25) and constant joint (30)  
- Train/validation/test split at sample level  

Balancing techniques explored: WeightedRandomSampler, Focal Loss, ADASYN (early experiments).

---

## Models

### Architectures Evaluated

- RNN, LSTM, GRU  
- Bidirectional variants  
- Up to 2 layers, max 64 hidden units  

### Baseline Models

- **LSTM64 v1**: 2-layer bidirectional LSTM, 64 units, minimal regularization  
- **LSTM64 v2**: Adds L1 regularization (1e−4) and label smoothing (α=0.05)  

### Custom Models

- **Custom Model 1**: Static features fused, class weighting, dropout=0.3  
- **Custom Model 2**: Gradient clipping, label smoothing, class weight scaled by √frequency  
- **Custom Model 3**: Hidden size 128, L2 regularization (1e−4), weighted sampling  
- **Custom Model 4**: Hidden size 128, custom weighting scheme  
- **Embedding Model**: Embedding layer for pain-survey features, followed by 128-unit LSTM  

---

## Results

| Model             | Validation F1 | Test F1    |
|-------------------|--------------|------------|
| **LSTM64 v1**     | 0.9269       | **0.9595** |
| Custom Model 1    | 0.9265       | 0.9565     |
| Custom Model 2    | **0.9316**   | 0.9534     |
| LSTM64 v2         | 0.9255       | 0.9500     |
| Custom Model 3    | 0.9311       | 0.9437     |
| Embedding Model   | 0.9327       | 0.9354     |
| Custom Model 4    | 0.9226       | 0.9338     |

Key takeaways: LSTM64 v1 achieves best test F1. Complex models improve validation but don't exceed baseline on test. L1 regularization harms performance; L2 helps slightly. Weighted sampling and class weighting increase robustness but with limited gains.

---

## Training Configuration

- Optimizer: Adam  
- Learning rate: 1e−3  
- Batch size: 32  
- Early stopping on validation loss  
- Dropout for regularization  

---

## Regularization and Imbalance Handling

- L1 regularization worsened performance  
- L2 regularization slightly improved results  
- Dropout reduced overfitting effectively  
- Imbalance handling via weighted sampler, focal loss, and ADASYN  

All experiments and analyses are documented in the provided notebooks.

---

## Installation and Usage

1. Clone the repository.  
2. Download the Pirate Pain Dataset separately and save it in your personal Google-Drive enviroiment.  
3. Run the notebooks or scripts in `notebooks/` and `scripts/`.

---


## Libraries and Tools

- Python 3.7+  
- TensorFlow 2.x  
- NumPy  
- Pandas  
- Scikit-learn  

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
