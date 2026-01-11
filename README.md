# Spam Email Classifier

## Problem Statement
The goal of this project is to build a machine learning model that can accurately classify emails as spam or non-spam (ham) using text-based features.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer

## Approach
- Performed data cleaning and preprocessing on raw email text
- Extracted numerical features using TF-IDF vectorization
- Trained a Logistic Regression classifier on the processed data
- Evaluated model performance using accuracy metric
- Ensured no data leakage by fitting the vectorizer only on training data

## Project Structure

- `data/` – Contains the dataset used for training and testing
- `src/` – Source code for data preprocessing, model definition, and training
  - `data_preprocessing.py` – Data cleaning and feature extraction
  - `model.py` – Machine learning model definition
  - `train.py` – Model training and evaluation script
- `notebook/` – Jupyter notebook used for experimentation
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation

## Results
- Achieved an accuracy of **96.6%** on the test dataset

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
