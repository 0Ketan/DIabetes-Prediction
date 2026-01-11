# Diabetes Prediction Using SVM

## Problem Statement
The objective of this project is to build a machine learning model that predicts whether a patient has diabetes based on medical diagnostic measurements.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Support Vector Machine (SVM)

## Approach
- Loaded and analyzed the diabetes dataset
- Split the dataset into training and testing sets
- Applied feature scaling using StandardScaler
- Trained a linear Support Vector Machine classifier
- Evaluated the model using accuracy metric
- Ensured no data leakage by fitting the scaler only on training data

## Project Structure
- `dataset/` – Contains the diabetes dataset
- `src/`
  - `data_preprocessing.py` – Data loading, splitting, and scaling
  - `model.py` – SVM model definition
  - `train.py` – Model training and evaluation
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation

## Results
- Achieved an accuracy of **XX%** on the test dataset

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the training script:
   ```bash
   python src/train.py
