# Titanic Survival Prediction using Machine Learning 🚢

## 📌 Project Overview

This project aims to predict whether a passenger survived the Titanic
disaster using machine learning techniques. The Titanic dataset contains
passenger information such as age, gender, ticket class, fare, and
embarkation port, which can be used to determine survival probability.

The goal of this project is to build and compare multiple machine
learning models to identify which model best predicts passenger
survival.

------------------------------------------------------------------------

## 📊 Dataset

The dataset contains information about passengers aboard the Titanic,
including:

-   PassengerId -- Unique passenger identifier
-   Survived -- Survival status (0 = Did not survive, 1 = Survived)
-   Pclass -- Passenger class (1 = First, 2 = Second, 3 = Third)
-   Name -- Passenger name
-   Sex -- Gender
-   Age -- Age of passenger
-   SibSp -- Number of siblings/spouses aboard
-   Parch -- Number of parents/children aboard
-   Ticket -- Ticket number
-   Fare -- Passenger fare
-   Cabin -- Cabin number
-   Embarked -- Port of embarkation (S, C, Q)

Dataset Source: Kaggle Titanic Dataset

------------------------------------------------------------------------

## ⚙️ Project Workflow

### 1. Data Loading

-   Imported the dataset using Pandas

### 2. Data Cleaning

-   Filled missing Age values with median
-   Filled missing Embarked values with mode
-   Dropped Cabin column due to high missing values

### 3. Data Preprocessing

-   Converted categorical variables to numeric:
    -   Sex → male = 0, female = 1
    -   Embarked → S = 0, C = 1, Q = 2
-   Removed unnecessary columns such as Name, Ticket, PassengerId

### 4. Exploratory Data Analysis (EDA)

Visualizations were created to understand survival patterns:

-   Survival distribution
-   Survival by Sex
-   Survival by Passenger Class
-   Fare distribution by survival
-   Age distribution by survival

### Key Insights

-   Females had a significantly higher survival rate than males
-   First class passengers were more likely to survive
-   Higher ticket fares correlated with higher survival rates

------------------------------------------------------------------------

## 🤖 Machine Learning Models Used

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   AdaBoost
-   XGBoost

Model performance was evaluated using ROC-AUC score.

------------------------------------------------------------------------

## 📈 Model Performance

  Model                 AUC Score
  --------------------- -----------
  Logistic Regression   0.8826
  Decision Tree         0.8073
  Random Forest         0.8772
  Gradient Boosting     0.8810
  AdaBoost              0.8615
  XGBoost               0.9012

Best Model: XGBoost

------------------------------------------------------------------------

## 📉 Evaluation Metrics

-   Accuracy Score
-   Confusion Matrix
-   Classification Report
-   ROC Curve
-   AUC Score

------------------------------------------------------------------------

## 🧠 Conclusion

This project demonstrates how machine learning can analyze historical
data to predict survival outcomes. Gender and passenger class strongly
influence survival chances, and ensemble models perform better than a
single decision tree.

XGBoost achieved the best performance with an AUC score of approximately
0.90.

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost
-   Jupyter Notebook

------------------------------------------------------------------------

## 📁 Project Structure

Titanic-Survival-Prediction/ │ ├── data/ │ └── Titanic-Dataset.csv ├──
notebooks/ │ └── titanic_analysis.ipynb ├── models/ │ └──
titanic_model.pkl ├── images/ │ └── roc_curve.png └── README.md

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Feature engineering (family size, title extraction)
-   Hyperparameter tuning
-   Deploy model using Streamlit or Flask
-   Build an interactive survival prediction app

------------------------------------------------------------------------

## 👤 Author

Deepak Kushwaha Machine Learning & Data Science Enthusiast
