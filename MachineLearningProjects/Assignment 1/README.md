# Exploratory Data Analysis (EDA) -- Assignment 1

## Course

Statistical Learning and Data Analysis

## Project Overview

This project performs a **complete Exploratory Data Analysis (EDA)** on
a selected dataset.\
The objective is to understand the dataset structure, identify patterns,
visualize relationships between variables, and apply dimensionality
reduction and clustering techniques.

The analysis follows the requirements described in the assignment brief.
fileciteturn0file0

------------------------------------------------------------------------

# Project Structure

    project/
    │
    ├── data/                 # Dataset used for the analysis
    ├── notebooks/
    │   └── eda_analysis.ipynb
    ├── figures/              # Generated plots and visualizations
    ├── README.md             # Project documentation

------------------------------------------------------------------------

# Assignment Tasks

## 1. Data Loading & Cleaning

-   Import the dataset
-   Display first rows of the dataset
-   Identify variable names and data types
-   Count observations and variables
-   Detect missing values and their proportions
-   Handle missing values using appropriate techniques (removal or
    imputation)

## 2. Univariate Data Description

-   Compute descriptive statistics:
    -   Mean
    -   Median
    -   Minimum
    -   Maximum
    -   Standard deviation
-   Create frequency tables for categorical variables
-   Plot:
    -   Histograms
    -   Boxplots
    -   Bar charts / Pie charts

Goal: Understand the distribution of each variable.

## 3. Bivariate and Multivariate Analysis

-   Compute correlations between numerical variables
-   Visualize correlations with a **heatmap**
-   Create **scatterplots** to analyze relationships
-   Compare distributions of numeric variables across categories
-   Build **contingency tables** for categorical variables

Goal: Discover relationships between variables.

## 4. Principal Component Analysis (PCA) and Clustering

### PCA

-   Scale and encode variables
-   Apply **Principal Component Analysis**
-   Determine number of components explaining **≥ 80% variance**
-   Visualize:
    -   Correlation circle
    -   PCA biplot

### Clustering

-   Select a clustering algorithm (e.g., K-Means)
-   Determine the optimal number of clusters
-   Visualize cluster results
-   Evaluate cluster separation

Goal: Identify patterns and groups within the data.

## 5. Summary and Interpretation

Provide a short summary including: - Main characteristics of the
dataset - Important patterns discovered - Differences between groups or
clusters - Key insights from the analysis

------------------------------------------------------------------------

# Output Requirements

The final submission must include:

1.  **Analysis Notebook**
    -   `.ipynb` or `.Rmd`
    -   Includes code, plots, and interpretations
2.  **Exported Report**
    -   `.pdf` or `.html`
3.  **Presentation**
    -   Short slides for exam discussion
    -   Focus on **interpretation and results**
    -   Use clear visuals and minimal text

------------------------------------------------------------------------

# Tools and Libraries

Example tools used for the analysis:

Python: - pandas - numpy - matplotlib - seaborn - scikit-learn

OR

R: - tidyverse - ggplot2 - dplyr - factoextra

------------------------------------------------------------------------

# Goal of the Project

The main goal is to **extract meaningful insights from data using
exploratory techniques**, statistical summaries, and machine learning
methods such as PCA and clustering.
