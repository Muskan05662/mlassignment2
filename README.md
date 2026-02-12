# Machine Learning Assignment 2
## Bank Marketing Classification
---

## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a customer will subscribe to term deposit with the help of Bank Marketing Dataset.
---

## Dataset Description

The dataset is the UCI Bank Marketing Dataset.
1. Total Instances: 45,211
2. Total Features: 16 input features
3. Target Variable: y (yes/no)
4. Type: Binary Classification
---

## Models Used

The following models are used:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest
6. XGBoost
---

## Evaluation Metrics

The following evaluation metrics were calculated for all models:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)
---

## Model  Results Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|------|------------|--------|----------|------|
| Logistic Regression | 0.9012 | 0.9054 | 0.6440 | 0.3488 | 0.4525 | 0.4264 |
| Decision Tree | 0.8777 | 0.7135 | 0.4783 | 0.4991 | 0.4884 | 0.4191 |
| KNN | 0.8936 | 0.8084 | 0.5860 | 0.3091 | 0.4047 | 0.3742 |
| Naive Bayes | 0.8639 | 0.8088 | 0.4282 | 0.4877 | 0.4560 | 0.3797 |
| Random Forest | 0.9045 | 0.9272 | 0.6554 | 0.3866 | 0.4863 | 0.4561 |
| XGBoost | 0.9080 | 0.9291 | 0.6348 | 0.5028 | 0.5612 | 0.5149 |
---

## Observations

| ML Model | Observation |
|------------|-----------------------------------|
| Logistic Regression | It has strong baseline and the accuracy and auc is high but recall value is low. This means it works well but misses positive cases |
| Decision Tree | Lower auc and moderate F1 shows it may be overfitting. |
| KNN | Accuracy is high but F1 is low, it may be sensitive to scaling. |
| Naive Bayes | It has balanced precision and recall, average performance can be due to independence assumptions between features. |
| Random Forest | Strong and stable performance with high accuracy and auc. |
| XGBoost | Best performance with highest accuracy and auc. |
---

## Conclusion

As the dataset is highly imbalanced, most no with some yes and has many categorical variables and some non-linear patterns, therefore XGBoost performed the best as it can handle non-linear and patterns and data imbalance best, along with random forest which is second best in performance, logistic regression is weaker because it is a linear model, and knn is effected due to scaling, decision tree is prone to overfitting.
---

## Streamlit Application

The project is deployed using Streamlit Community Cloud.

Live App Link: https://mlassignment2-a4jv3guyfteawh9u8irhaf.streamlit.app
---

## Repository Structure


