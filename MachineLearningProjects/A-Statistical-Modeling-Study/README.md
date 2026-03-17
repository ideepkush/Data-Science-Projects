# 📊 Statistical Learning and Data Analysis – Assignment 2  
### Probability Models Comparison: Poisson vs Exponential

**Author:** Deepak Kushwaha  

---

## 📌 Overview

This project presents a comparative analysis of two fundamental probability distributions:

- **Poisson Distribution**
- **Exponential Distribution**

The notebook explores their theoretical foundations, statistical properties, simulation behavior, and real-world applications. It also demonstrates how these distributions relate to each other within the framework of stochastic processes.

---

## 🎯 Objectives

- Understand the mathematical formulation of Poisson and Exponential distributions  
- Analyze their shape, skewness, and parameter sensitivity  
- Study their connection to the Gaussian (Normal) distribution  
- Perform simulation experiments to validate theoretical results  
- Apply both models to a real-world scenario (web server traffic)  

---

## 📚 Theoretical Background

### 🔹 Poisson Distribution
- Models the number of events occurring in a fixed interval  
- Parameter: λ (rate of occurrence)  
- Mean = Variance = λ  
- Skewness decreases as λ increases  

### 🔹 Exponential Distribution
- Models the time between events in a Poisson process  
- Parameter: λ (rate)  
- Mean = 1/λ  
- Always positively skewed  

---

## 📈 Key Insights

### ✔ Shape & Behavior
- Poisson becomes more symmetric as λ increases  
- Exponential remains right-skewed regardless of λ  

### ✔ Convergence to Normal Distribution
- Poisson distribution approximates a Normal distribution for large λ  
- Demonstrates asymptotic behavior  

---

## 🧪 Simulation Study

The notebook includes simulations to compare:
- Theoretical distributions vs empirical data  
- Behavior under different λ values (e.g., 1, 5, 20)  
- Histogram visualization with overlaid probability density functions  

### 🔍 Observations
- Strong agreement between simulated and theoretical results  
- Clear visualization of Poisson → Normal convergence  
- Stable exponential decay pattern  

---

## 🌐 Real-World Application

### 📡 Web Server Traffic Analysis

This project models server activity using:

- Poisson Distribution → Number of requests per unit time  
- Exponential Distribution → Time between consecutive requests  

#### Scenario:
- API server handling incoming HTTP requests  
- Requests assumed to follow a Poisson process  

#### Simulation:
- Generated inter-arrival times using exponential distribution  
- Constructed arrival timestamps  
- Analyzed request patterns over time  

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Matplotlib  
- SciPy  

---

## 📁 Project Structure

Revised_assignment2.ipynb   # Main notebook with analysis and simulations  
README.md                  # Project documentation  

---

## ▶️ How to Run

1. Install required libraries:
   pip install numpy matplotlib scipy

2. Open the notebook:
   jupyter notebook Revised_assignment2.ipynb

3. Run all cells to reproduce results and visualizations.

---

## 📌 Conclusion

This project highlights:
- The relationship between count-based and time-based probabilistic models  
- The importance of Poisson processes in real-world systems  
- How theoretical results are validated through simulation  

---

## 📬 Author

Deepak Kushwaha  
