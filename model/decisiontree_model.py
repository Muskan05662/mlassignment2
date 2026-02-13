from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,recall_score, f1_score, matthews_corrcoef,confusion_matrix)

def decision_model(X_train, y_train, X_test, y_test):

    decision_model = DecisionTreeClassifier(random_state=42)
    decision_model.fit(X_train, y_train)

    y_pred = decision_model.predict(X_test)
    y_prob = decision_model.predict_proba(X_test)[:, 1]

    result = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    cm = confusion_matrix(y_test, y_pred)

    return decision_model, result, cm
