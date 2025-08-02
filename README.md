# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using supervised learning models, with a focus on handling data imbalance and optimizing model performance.

## 🚀 Project Overview

This project builds a pipeline to classify credit card transactions as fraudulent or non-fraudulent using machine learning. It uses preprocessing, resampling techniques (like SMOTE), and various classifiers (Logistic Regression, Decision Trees, Random Forest, etc.) to tackle the real-world challenge of extreme class imbalance.

## 📁 Repository Structure
credit-card-fraud-detection/
├── data/ # Raw and preprocessed datasets
├── notebooks/ # Jupyter notebooks for EDA and model experimentation
├── src/ # Python scripts for preprocessing, training, and evaluation
├── models/ # Saved trained model files
├── requirements.txt # List of dependencies
├── README.md # Project documentation
└── LICENSE # License info (to be added)


## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions over two days in 2013
- **Features**: 31 total (PCA components, `Time`, `Amount`, `Class`)
- **Target**: `Class` → 0 = Non-Fraud, 1 = Fraud
- **Class Imbalance**: Fraudulent transactions are only ~0.17% of the dataset

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kthiruvikram/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

