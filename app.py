import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,recall_score, f1_score, matthews_corrcoef,confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt


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
#preprocessing
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data_new = pd.get_dummies(data, drop_first=True)

X_data = data_new.drop('y', axis=1)
y_data = data_new['y']

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#performance metrcis
def performance_metrics(y_test, y_pred, y_prob):
    pref_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
    return pref_metrics

@st.cache_resource
def models(X_train, y_train):

    logistic_model = LogisticRegression(max_iter=1000)
    decision_model = DecisionTreeClassifier(random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    nb_model = GaussianNB()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

    logistic_model.fit(X_train, y_train)
    decision_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return (logistic_model,decision_model,knn_model,nb_model,rf_model,xgb_model)

logistic_model, decision_model, knn_model, nb_model, rf_model, xgb_model = models(X_train, y_train)
#results stored
result = {}
model_dict = {"Logistic Regression": logistic_model, "Decision Tree": decision_model,"KNN": knn_model,"Naive Bayes": nb_model,"Random Forest": rf_model,"XGBoost": xgb_model}
for name, model in model_dict.items():
    y_pred_per = model.predict(X_test)
    y_prob_per = model.predict_proba(X_test)[:, 1]
    result[name] = performance_metrics(y_test, y_pred_per, y_prob_per)


result_table = pd.DataFrame(result).T.round(4)

st.subheader("Model Comparison Table")
st.dataframe(result_table)
#model chosen 
choose_model = st.selectbox(
    "Select Model for Evaluation",
    list(result.keys())
)

model_dict = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_model,
    "KNN": knn_model,                         #model dictionary
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

chosen_model = model_dict[choose_model]

file = st.file_uploader("Upload test dataset (optional)",type="csv") #file upload

if file is not None:       #if-else for file upload

    st.subheader("Evaluation on uploaded dataset")

    test_data = pd.read_csv(file, sep=';')

    test_data['y'] = test_data['y'].map({'yes': 1, 'no': 0})
    test_data_new = pd.get_dummies(test_data, drop_first=True)

    test_data_new = test_data_new.reindex(
        columns=data_new.columns,
        fill_value=0
    )

    X_data_new = test_data_new.drop('y', axis=1)
    y_data_new = test_data_new['y']

    X_data_new = scaler.transform(X_data_new)

else:

    st.subheader("No test file uploaded, evaluation on internal dataset")

    X_data_new = X_test
    y_data_new = y_test

y_pred = chosen_model.predict(X_data_new)
y_prob = chosen_model.predict_proba(X_data_new)[:, 1]

chosen_metrics = performance_metrics(y_data_new, y_pred, y_prob)

for i, j in chosen_metrics.items():
    st.write(f"{i}: {round(j, 4)}")

st.subheader(f"Confusion Matrix for {choose_model}")

cm = confusion_matrix(y_data_new, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
