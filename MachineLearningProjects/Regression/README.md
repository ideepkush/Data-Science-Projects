Bank Regression Simulation

## 📌 Project Overview
This project examines the performance of different regression methodologies under a controlled simulation framework. By implementing classical linear regression, penalized regression techniques, and non-linear modeling, we evaluate coefficient estimation accuracy, model robustness, the impact of shrinkage, and the bias–variance trade-off. 

This notebook is designed to demonstrate how model flexibility affects generalization and how to handle both linear and non-linear relationships in data.

## 🎯 Objectives
* **Data Simulation:** Construct a synthetic dataset with a known underlying Data Generating Mechanism (DGM) to test model behavior.
* **Classical Linear Regression (OLS):** Evaluate coefficient estimation accuracy, statistical significance, and verify regression assumptions through diagnostic plots.
* **Penalized Regression:** Implement Ridge, Lasso, and Elastic Net models to observe the impact of shrinkage, stability, and variable selection on noisy datasets.
* **Non-Linear Modeling:** Explore non-linear relationships using Polynomial Regression, Support Vector Regression (SVR), and Decision Trees, illustrating the bias-variance trade-off and the dangers of overfitting.

## 📊 Dataset Simulation
Because the primary goal is methodological evaluation, a synthetic dataset is generated with:
* **Observations:** `n = 300`
* **Predictors:** `p = 10`
* **True Coefficients:**
  * **Strong Signals:** Variables that strongly drive the response.
  * **Moderate Signals:** Subtle but existing effects.
  * **Noise Variables:** Variables with no true effect (`beta = 0`), included to test the models' feature selection capabilities.
* **Random Error:** Normally distributed noise to make the outcome stochastic and require statistical inference.

## 🛠️ Methodologies Implemented

### 1. Ordinary Least Squares (OLS)
* Estimated using `statsmodels` to obtain full inferential statistics (p-values, t-statistics).
* **Diagnostics:** Evaluated using Residuals vs Fitted plots, Q-Q plots, Scale-Location plots, and Residuals vs Leverage to check for homoscedasticity, normality, and influential points.

### 2. Regularization / Penalized Regression
* **Ridge Regression:** Applied L2 penalty to shrink coefficients and handle multicollinearity.
* **Lasso Regression:** Applied L1 penalty, successfully performing variable selection by pushing the coefficients of noise variables to exactly zero.
* **Elastic Net:** Combined L1 and L2 penalties.
* Cross-validation (`RidgeCV`, `LassoCV`, `ElasticNetCV` from `scikit-learn`) was used to select the optimal tuning hyperparameters.

### 3. Non-Linear Models & Bias-Variance Trade-off
Simulated a new non-linear dataset (`y = sin(2X) + 0.1 * X^2 + noise`) to evaluate:
* **Polynomial Regression:** Demonstrated that while low degrees underfit, excessively high degrees (e.g., degree 15) lead to severe overfitting.
* **Support Vector Regression (SVR):** Utilized the RBF kernel to flexibly capture non-linear trends without wild oscillations.
* **Decision Tree Regressor:** Showed step-like prediction behavior, highly sensitive to local data variations.

## 💡 Key Findings
1. **OLS** successfully and unbiasedly recovers the true data structure *only* when the linear assumptions hold.
2. **Regularization** (specifically Lasso) is highly effective at variable selection, filtering out irrelevant noise variables and improving model stability.
3. In **non-linear settings**, excessive model complexity leads to overfitting. The bias-variance trade-off highlights the absolute necessity of balancing a model's flexibility against its ability to generalize to unseen data.

## ⚙️ Requirements & Libraries
To run the Jupyter Notebook, you need Python 3.x and the following libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `statsmodels`
* `scikit-learn`

## 🚀 How to Run
1. Clone the repository or download the notebook `Assignment3_Bank_Regression.ipynb`.
2. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.
3. Run the cells sequentially to view the data generation, statistical summaries, and model visualizations.
