# Flight Fare Prediction

## Overview
This project is a Machine Learning pipeline designed to predict airline flight fares based on various historical flight parameters. By analyzing features such as the airline brand, journey date, flight duration, route, and the number of stops, the model can accurately estimate the price of a flight ticket. 

The project utilizes a **Random Forest Regressor** to achieve high predictive accuracy, explaining approximately 81% of the variance in flight prices.

## Dataset
The project uses a dataset named `Data_Train.xlsx` which contains 10,683 records and 11 features initially.
**Features included:**
* `Airline`: The name of the airline.
* `Date_of_Journey`: The date the flight took off.
* `Source`: The starting city.
* `Destination`: The destination city.
* `Route`: The flight's route path.
* `Dep_Time` & `Arrival_Time`: Departure and arrival times.
* `Duration`: Total time taken for the flight.
* `Total_Stops`: The number of layovers.
* `Additional_Info`: Any extra details about the flight.
* `Price`: The target variable (ticket fare).

## Project Workflow

### 1. Data Cleaning
* Handled missing (NaN) values by dropping them, as they were very few.
* Identified and removed duplicate flight records to prevent data leakage and bias.

### 2. Feature Engineering
* **Date & Time Extraction:** Converted `Date_of_Journey`, `Arrival_Time`, and `Dep_Time` into actionable numerical features (Day, Month, Year, Hour, Minute).
* **Flight Shifts:** Grouped departure times into categorical shifts (Early Morning, Morning, Afternoon, Evening, Night, Late Night) and analyzed their volume using Plotly and Seaborn.
* **Duration Parsing:** Cleaned the `Duration` column to extract exact hours and minutes, converting the total duration into minutes for easier mathematical computation.

### 3. Categorical Encoding
* **Target-Guided Encoding:** Encoded variables like `Airline` and `Destination` by replacing categories with mapped numerical values ordered by their mean `Price`.
* **Dictionary Mapping:** Converted `Total_Stops` into ordinal numeric values (e.g., 'non-stop' = 0, '1 stop' = 1).
* **One-Hot Encoding:** Applied to the `Source` column using dummy variables.

### 4. Exploratory Data Analysis (EDA) & Outlier Detection
* Analyzed feature distributions and the impact of flight duration/airlines on ticket pricing.
* Implemented plotting functions (Distribution plot, Boxplot, Scatterplot) for outlier detection and treatment.

### 5. Model Building & Hyperparameter Tuning
* Trained an initial **Random Forest Regressor**.
* Performed hyperparameter tuning using cross-validation to find the optimal parameters for the Random Forest model:
  * `n_estimators`: 1200
  * `max_depth`: 21
  * `min_samples_split`: 15
  * `max_features`: None

### 6. Model Evaluation
The tuned model was evaluated using standard regression metrics:
* **R² Score:** ~0.807 (The model captures 81% of the variance).
* **Mean Absolute Error (MAE):** ~1185.90
* **Mean Absolute Percentage Error (MAPE):** ~13.4% 
* Plotted the distribution of residuals (errors) which showed a normal distribution centered around zero, indicating a well-fitted model.

### 7. Model Serialization
* Saved the finalized, trained model as a serialized file (`flight_price_rf_model.pkl`) using Python's `pickle` library, allowing for easy deployment and future predictions without retraining.

## Dependencies / Requirements
To run this notebook, you will need the following Python libraries installed:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `plotly`
* `scikit-learn`

## Usage
1. Ensure the dataset (`Data_Train.xlsx`) is placed in the same directory as the notebook.
2. Run the notebook cells sequentially to execute the data preprocessing, model training, and evaluation steps.
3. The final cell will generate the `flight_price_rf_model.pkl` file which can be loaded into any Python script to predict new flight prices.
