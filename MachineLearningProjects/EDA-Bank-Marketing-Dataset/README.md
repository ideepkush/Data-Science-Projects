
# Exploratory Data Analysis (EDA) – Assignment 1

## Course
Statistical Learning and Data Analysis

## Project Overview
This project performs a full Exploratory Data Analysis (EDA) on a dataset as part of Assignment 1. 
The objective is to understand the structure of the data, explore distributions, analyze relationships 
between variables, and apply dimensionality reduction and clustering techniques.

The analysis is implemented in the notebook **EDA.ipynb**.

---

# Project Workflow

## 1. Data Loading & Cleaning
- Import the dataset
- Display the first rows of the dataset
- Identify variable names and data types
- Determine the number of observations and variables
- Check for missing values and their proportions
- Handle missing values using removal or imputation if necessary

---

## 2. Univariate Data Description

### Numerical Variables
- Compute descriptive statistics:
  - Mean
  - Median
  - Minimum
  - Maximum
  - Standard deviation

- Visualizations:
  - Histograms
  - Boxplots

Observation example:
Many numeric variables show **right‑skewed distributions**, meaning most values are small while a few observations are very large.

### Categorical Variables
- Create frequency tables
- Visualize using:
  - Bar charts
  - Pie charts

Goal:
Identify the most frequent categories and understand the distribution of categorical variables.

---

## 3. Bivariate and Multivariate Analysis

### Correlation Analysis
- Compute correlations among numeric variables
- Visualize correlations using a **heatmap**

Observation:
Most numeric variables show **very weak correlations**, suggesting they are largely independent.

### Scatterplot Analysis
Scatterplots are used to explore relationships between pairs of numeric variables and detect:
- Linear relationships
- Nonlinear relationships
- Outliers

### Category Comparison
Distribution of numeric variables is compared across categorical groups.

### Contingency Tables
Used to explore relationships between categorical variables.

---

## 4. Principal Component Analysis (PCA) and Clustering

### Data Preparation
- Categorical variables encoded using **One‑Hot Encoding**
- Features scaled using **standardization**

### PCA
Principal Component Analysis is applied to reduce dimensionality.

Goals:
- Identify the number of components explaining **at least 80% of variance**
- Visualize:
  - Correlation circle
  - PCA biplot

Observation:
The first principal components capture the most important directions of variance in the dataset.

### Clustering

Algorithm used:
**K‑Means Clustering**

Steps:
1. Determine optimal number of clusters using the **Elbow Method**
2. Fit K‑Means model
3. Visualize clusters in PCA space

Observation:
Clusters are **moderately separated**, indicating some structure in the dataset.

---

## 5. Summary and Interpretation

Key findings:
- Most numeric variables show skewed distributions.
- Correlations between numeric features are generally weak.
- PCA effectively reduces dimensionality while preserving most variance.
- K‑Means clustering identifies groups with moderate separation.

Overall, the analysis reveals patterns in the dataset and helps understand its structure through statistical summaries and visualizations.

---

# Project Files

```
EDA.ipynb      → Main notebook containing the full analysis
README.md      → Project documentation
data/          → Dataset used in the analysis
figures/       → Generated plots and visualizations
```

---

# Technologies Used

Python Libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit‑learn

---

# Author
Deepak Kushwaha

