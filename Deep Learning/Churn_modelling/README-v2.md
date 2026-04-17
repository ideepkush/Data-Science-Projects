# Customer Churn Prediction using Artificial Neural Networks (ANN)

## Overview
This repository contains a deep learning project aimed at predicting customer churn. By analyzing customer demographics and banking behaviors, the Artificial Neural Network (ANN) implemented in this project predicts whether a customer is likely to close their bank account (`Exited`). The model is built utilizing TensorFlow and Keras.

## Dataset: `Churn_Modelling.csv`
The project utilizes the `Churn_Modelling.csv` dataset, which contains 10,000 records of bank customers. 

**Key Features Include:**
* **Demographics:** `Geography` (France, Spain, Germany), `Gender`, `Age`
* **Account Information:** `CreditScore`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
* **Target Variable:** `Exited` (1 = Customer left the bank, 0 = Customer stayed)

## Project Structure
* **`ann.ipynb`**: The main Jupyter Notebook containing:
    * Data preprocessing (handling categorical variables via one-hot encoding, feature scaling).
    * Building the Artificial Neural Network architecture.
    * Hyperparameter tuning (e.g., using Grid Search techniques).
    * Model evaluation and prediction.
* **`Churn_Modelling.csv`**: The raw dataset used for training and testing.

## Model Performance
The ANN was evaluated on a held-out test set, yielding the following results:
* **Test Accuracy:** `~80.45%`
* **Confusion Matrix:**
    ```text
    [[1530,   53],
     [ 338,   79]]
    ```
    *(Interpretation: 1530 True Negatives, 53 False Positives, 338 False Negatives, and 79 True Positives)*

## Dependencies
Ensure you have the following Python libraries installed before running the notebook:
* `numpy`
* `pandas`
* `tensorflow`
* `keras`
* `scikit-learn` (for data splitting, scaling, and metrics)

## Usage Instructions
1.  Ensure both `ann.ipynb` and `Churn_Modelling.csv` are located in the same directory.
2.  Install the required dependencies using your environment manager (e.g., `pip install pandas numpy tensorflow scikit-learn`).
3.  Open the Jupyter Notebook (`jupyter notebook ann.ipynb` or via VS Code/JupyterLab).
4.  Run all cells sequentially to preprocess the data, train the model, and view the evaluation metrics.
