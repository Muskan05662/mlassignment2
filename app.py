import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# importing model .py files functions
from model.logistic_model import logistic_model
from model.decisiontree_model import decision_model
from model.knn_model import knn_model
from model.naivebayes_model import naivebayes_model
from model.randomforest_model import randomforest_model
from model.xgb_model import xgb_model

# loading dataset
st.title("Machine Learning Assignment 2")
st.subheader("Bank Marketing Classification")


@st.cache_data
def load_data():
    dataset_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    zip_file = requests.get(dataset_url)
    zip_file.raise_for_status()

    with zipfile.ZipFile(BytesIO(zip_file.content)) as outer_zip_file:
        with outer_zip_file.open("bank.zip") as inner_zip_file:
            with zipfile.ZipFile(BytesIO(inner_zip_file.read())) as inner_zip:
                with inner_zip.open("bank-full.csv") as csv_file:
                    data = pd.read_csv(csv_file, sep=';')

    return data


data = load_data()

# preprocessing
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data_new = pd.get_dummies(data, drop_first=True)

X_data = data_new.drop('y', axis=1)
y_data = data_new['y']

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model training

@st.cache_resource
def model_train(X_train, y_train, X_test, y_test):
    logistic_model, logistic_result, logistic_cm = logistic_model(X_train, y_train, X_test, y_test)
    decision_model, decision_result, decision_cm = decision_model(X_train, y_train, X_test, y_test)
    knn_model, knn_result, knn_cm = knn_model(X_train, y_train, X_test, y_test)
    nb_model, nb_result, nb_cm = naivebayes_model(X_train, y_train, X_test, y_test)
    rf_model, rf_result, rf_cm = randomforest_model(X_train, y_train, X_test, y_test)
    xgb_model, xgb_result, xgb_cm = xgb_model(X_train, y_train, X_test, y_test)

    return (logistic_model, logistic_result, logistic_cm,decision_model, decision_result, decision_cm,knn_model, knn_result, knn_cm,nb_model, nb_result, nb_cm,rf_model, rf_result, rf_cm,xgb_model, xgb_result, xgb_cm)

(logistic_model,logistic_result,logistic_cm,decision_model,decision_result,decision_cm,knn_model,knn_result,knn_cm,nb_model,nb_result,nb_cm,rf_model,rf_result,rf_cm,xgb_model,xgb_result,xgb_cm,) = model_train(X_train, y_train, X_test, y_test)


# result dictionary
result = {
    "Logistic Regression": logistic_result,
    "Decision Tree": decision_result,
    "KNN": knn_result,
    "Naive Bayes": nb_result,
    "Random Forest": rf_result,
    "XGBoost": xgb_result
}

# creating result table
result_table = pd.DataFrame(result).T.round(4)

st.subheader("Model Comparison Table")
st.dataframe(result_table)

# selecting the model
choose_model = st.selectbox(
    "Select Model for Evaluation",
    list(result.keys())
)

model_dict = {
    "Logistic Regression": (logistic_model, logistic_cm),
    "Decision Tree": (decision_model, decision_cm),
    "KNN": (knn_model, knn_cm),
    "Naive Bayes": (nb_model, nb_cm),
    "Random Forest": (rf_model, rf_cm),
    "XGBoost": (xgb_model, xgb_cm)
}

chosen_model, chosen_cm = model_dict[choose_model]

# uploading test csv file
file = st.file_uploader("Upload test dataset (optional)", type="csv")

if file is not None:

    st.subheader("Evaluation on uploaded dataset")

    test_data = pd.read_csv(file, sep=';')
    test_data['y'] = test_data['y'].map({'yes': 1, 'no': 0})
    test_data_new = pd.get_dummies(test_data, drop_first=True)
    test_data_new = test_data_new.reindex(columns=data_new.columns, fill_value=0)

    X_data_new = test_data_new.drop('y', axis=1)
    y_data_new = test_data_new['y']
    X_data_new = scaler.transform(X_data_new)

    y_pred = chosen_model.predict(X_data_new)
    y_prob = chosen_model.predict_proba(X_data_new)[:, 1]

else:

    st.subheader("No test file uploaded, evaluation on internal dataset")

    X_data_new = X_test
    y_data_new = y_test
    y_pred = chosen_model.predict(X_data_new)
    y_prob = chosen_model.predict_proba(X_data_new)[:, 1]


# metrics printing
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

chosen_metrics = {
    "Accuracy": accuracy_score(y_data_new, y_pred),
    "AUC": roc_auc_score(y_data_new, y_prob),
    "Precision": precision_score(y_data_new, y_pred),
    "Recall": recall_score(y_data_new, y_pred),
    "F1 Score": f1_score(y_data_new, y_pred),
    "MCC": matthews_corrcoef(y_data_new, y_pred)
}

for i, j in chosen_metrics.items():
    st.write(f"{i}: {round(j, 4)}")


# confusion matrix printing
st.subheader(f"Confusion Matrix for {choose_model}")
cm = confusion_matrix(y_data_new, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
