# Hypertension Risk Prediction Pipeline

## Overview

This repository contains a complete end-to-end Machine Learning pipeline designed to predict the presence of hypertension in individuals based on lifestyle factors, medical history, and physical metrics. By comparing a wide array of classification models—from a custom-built Random Forest to state-of-the-art gradient boosting algorithms—this project identifies the most accurate approach for health-risk classification.

## Dataset

The project uses a dataset consisting of 1,985 records and 11 distinct attributes (prior to one-hot encoding).

* **Target Variable:** `Has_Hypertension` (Yes/No)
* **Numerical Features:** `Age`, `Salt_Intake`, `Stress_Score`, `Sleep_Duration`, `BMI`
* **Categorical Features:** `BP_History`, `Medication`, `Family_History`, `Exercise_Level`, `Smoking_Status`

## Data Preprocessing & Feature Engineering

To ensure optimal model performance, the data underwent a rigorous preprocessing pipeline:

1. **Feature Scaling:** Applied `StandardScaler` to all numerical features to normalize their ranges (zero mean, unit variance).
2. **Missing Value Imputation:** Handled 799 missing values in the `Medication` column and other potential gaps by imputing the median for numerical columns and the mode for categorical columns.
3. **Data Cleaning:** Removed duplicate records to prevent data leakage and bias.
4. **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) to transform categorical text data into a machine-readable numeric format.

## Exploratory Data Analysis (EDA)

The pipeline includes visual diagnostics to uncover relationships between lifestyle factors and hypertension:

* **Boxplots & Histograms:** Showcased the distribution of Age, BMI, Salt Intake, and Sleep Duration across hypertensive vs. non-hypertensive groups.
* **Countplots:** Highlighted the impact of BP History, Medication, and Family History on the target variable.
* **Correlation Heatmap:** Mapped the linear relationships between all encoded features to identify primary drivers of hypertension.

## Model Architectures & Experiments

Seven different classification models were trained and evaluated on a standard 80/20 train-test split (1,588 training samples, 397 testing samples).

1. **XGBoost Classifier** (Best Performer)
2. **Gradient Boosting Classifier**
3. **Decision Tree Classifier**
4. **Logistic Regression**
5. **K-Nearest Neighbors (KNN)**
6. **Gaussian Naive Bayes**
7. **Custom Random Forest** (A from-scratch implementation to demonstrate algorithmic understanding)

## Results & Evaluation

Models were rigorously evaluated using Accuracy, F1 Score, and Recall to ensure a balance between identifying true positive hypertension cases and minimizing false alarms.

| Model | Accuracy | F1 Score | Recall |
| --- | --- | --- | --- |
| **XGBoost Classifier** | **98.74%** | **0.99** | **0.99** |
| Gradient Boosting | 98.49% | 0.99 | 0.99 |
| Decision Tree | 93.45% | 0.94 | 0.94 |
| Logistic Regression | 88.66% | 0.89 | 0.88 |
| K-Nearest Neighbors | 82.37% | 0.82 | 0.79 |
| Naive Bayes | 82.12% | 0.83 | 0.84 |
| Custom Random Forest* | 51.64% | 0.68 | 1.00 |

**Note: The custom Random Forest implementation was built for educational purposes and defaults heavily to the majority class (Recall = 1.00), whereas the optimized library models achieve near-perfect classification.*

## Key Insights

* **Tree-Based Dominance:** Ensemble methods, specifically XGBoost and Gradient Boosting, vastly outperformed traditional statistical models, achieving near-perfect accuracy (>98.4%).
* **Linear vs. Non-Linear:** Logistic Regression performed admirably (~88.6%), but the ~10% gap between it and XGBoost implies that the relationships between lifestyle factors (like BMI, Stress, and Salt Intake) and Hypertension are highly non-linear.

## Technologies Used

* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (StandardScaler, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, GradientBoostingClassifier), `xgboost`
* **Data Visualization:** `matplotlib`, `seaborn`
