
# Comparing Classifiers

## Overview
This project explores and compares the performance of several classification algorithms using a dataset from a Portuguese banking institution. The objective is to predict whether a client will subscribe to a term deposit based on demographic and marketing campaign data. Models compared include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

The dataset was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

## Business Objective
The goal is to assist the bank in identifying clients who are likely to subscribe to term deposits. A predictive model can help streamline marketing efforts, reduce costs, and improve conversion rates by focusing on high-potential leads.

## Data Preprocessing & Feature Engineering
- Removed "duration" from modeling as it leaks target information.
- Encoded categorical variables using one-hot encoding.
- Treated "unknown" values in features like `job`, `education`, and `housing` either as a category or by imputing the mode.

## Baseline Model
- The majority class ("no") represents **88.73%** of the data.
- A baseline model predicting all "no" yields **88.7% accuracy**, but fails to identify positive cases (clients who subscribe).

## Models Compared (Default Settings)
| Model                  | Accuracy | Observations |
|------------------------|----------|--------------|
| Logistic Regression    | ~89%     | High accuracy, poor recall on positive class |
| K-Nearest Neighbors    | ~88%     | Performance similar to baseline |
| Decision Tree          | ~87%     | Quick fit, interpretable |
| Support Vector Machine | ~89%     | Slow to train, did not significantly outperform logistic regression |

## Class Imbalance Observations
- All models showed high accuracy but **very poor recall** for the minority class (`y=1`).
- Logistic Regression, even with `class_weight='balanced'`, struggled to identify true positives.
- Precision and recall metrics revealed that most models were biased toward the majority class.

## Model Improvements
- Applied `class_weight='balanced'` to Logistic Regression.
- Evaluated models using F1-score and ROC-AUC to get a better sense of true performance.

## Next Steps & Recommendations
- Implement Random Forest or Gradient Boosted Trees (e.g., XGBoost) for better handling of class imbalance.
- Perform GridSearchCV to tune hyperparameters like `max_depth`, `n_neighbors`, and `C`.
