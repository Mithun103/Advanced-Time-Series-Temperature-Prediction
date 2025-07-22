# 🌡️ Advanced Time-Series Temperature Prediction: A Complete Walkthrough

This project documents a **complete, end-to-end machine learning pipeline** for predicting the **average daily temperature (TAVG)**. Starting with raw historical weather data from multiple stations, we cover data analysis, advanced feature engineering, hyperparameter tuning, and high-performance ensemble models to achieve state-of-the-art accuracy.

---

## 📂 Project Pipeline

### ✅ Phases Overview

| Phase | Description |
|-------|-------------|
| **1** | Data Loading and Initial Analysis |
| **2** | Data Cleaning and Preprocessing |
| **3** | Exploratory Data Analysis (EDA) |
| **4** | Feature Engineering |
| **5** | Model Training & Hyperparameter Tuning |
| **6** | Evaluation and Ensemble Modeling |

---

## 🧾 Dataset Description

- **Source**: Historical weather data from 3 stations (A, B, C)
- **Target Variable**: `TAVG` – Average Daily Temperature
- **Features**:
  - `TMAX`, `TMIN`, `TAVG`: Temperature metrics
  - `PRCP`: Precipitation
  - `SNWD`: Snow depth
  - `DATE`, `LATITUDE`, `LONGITUDE`, etc.

---

## 🔍 Phase 1: Data Foundations and Initial Analysis

- Loaded `train.csv` into a DataFrame.
- Inspected data structure, types, and missing values.

```python
df.info()
```

🧠 Key Insights:
- 812 entries
- 29 columns: mostly `float64`, `int64`, and one `object` (`DATE`)
- Substantial missing values in core features

---

## 🧹 Phase 2: Data Cleaning and Preprocessing

### ✅ Actions Taken:
- Dropped redundant `Unnamed: 0`
- Converted `DATE` to `datetime` and set as index
- Filled missing values using column-wise mean

### ✅ Cleaned Data Snapshot:

```python
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 812 entries, 1979-11-01 to 1978-02-28
Data columns (total 27 columns):
...
```

---

## 📊 Phase 3: Exploratory Data Analysis (EDA)

### TAVG Over Time

📈 Shows seasonal fluctuations corresponding to summer/winter cycles.

![TAVG Over Time](plots/tavg_over_time.png)

### Correlation Heatmap

🎯 Temperature-based features (`TMAX`, `TMIN`) are highly correlated with `TAVG`.

![Correlation Heatmap](plots/correlation_heatmap.png)

---

## 🧠 Phase 4: Feature Engineering

### Created Features:

| Category          | Features |
|-------------------|----------|
| **Time-based**    | `year`, `month`, `dayofyear` |
| **Cyclical**      | `month_sin`, `month_cos`, `dayofyear_sin`, `dayofyear_cos` |
| **Lag Features**  | `TAVG_lag_1`, `TAVG_lag_7` |
| **Rolling Stats** | `TAVG_rolling_mean_7`, `TAVG_rolling_std_7` |

---

## 🔧 Phase 5: Hyperparameter Tuning (LightGBM)

### Optimization via `RandomizedSearchCV`

```python
param_grid = {
    "n_estimators": randint(100, 500),
    "learning_rate": uniform(0.01, 0.2),
    "num_leaves": randint(20, 60),
    "max_depth": randint(5, 20),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0, 1)
}
```

---

## 🤖 Phase 6: Model Training & Ensembles

### 📌 Base Models Used:

| Model         | Type                    | Notes |
|---------------|-------------------------|-------|
| Random Forest | `RandomForestRegressor` | Baseline model |
| XGBoost       | `XGBRegressor`          | Part of ensemble |
| LightGBM      | `LGBMRegressor`         | Tuned model |
| Linear Model  | `LinearRegression`      | Meta-model for stacking |

---

## 🧪 Model Evaluation

Data split: 80% train / 20% test  
Scaler: `StandardScaler`

| Model                          | R2 Score | MAE   |
|--------------------------------|----------|-------|
| Random Forest (Baseline)       | 0.9600   | 1.78  |
| Averaging Ensemble             | 0.9613   | 1.71  |
| Stacking Ensemble              | 0.9612   | 1.70  |
| **Final Tuned Ensemble**       | **0.9760**   | **1.46**  |

---

## 📉 Model Visualizations

### 📌 Baseline Random Forest
![Baseline RF](plots/baseline_rf_plot.png)

### 📌 Simple Averaging Ensemble
![Averaging Ensemble](plots/averaging_ensemble_plot.png)

### 📌 Stacking Ensemble
![Stacking Ensemble](plots/stacking_ensemble_plot.png)

### 📌 Final Tuned Ensemble
![Final Tuned Ensemble](plots/final_prediction.png)

---

## 🚀 How to Run

### 🔧 Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### ▶️ Run the Notebook

1. Open `new.ipynb`
2. Execute cells sequentially
3. View results in the output cells

---

## 📂 Folder Structure

```
temperature-prediction/
├── new.ipynb
├── train.csv
├── plots/
│   ├── tavg_over_time.png
│   ├── correlation_heatmap.png
│   ├── baseline_rf_plot.png
│   ├── averaging_ensemble_plot.png
│   ├── stacking_ensemble_plot.png
│   └── final_prediction.png
└── README.md
```

---

## 🏁 Final Notes

This project demonstrates the power of **feature engineering + ensemble models** for time-series prediction. The strong seasonal signal, combined with lag and rolling features, enabled accurate forecasting of daily average temperature.
