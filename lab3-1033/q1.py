# Hafsa-1033

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
data = pd.read_csv("Telco Customer Churn.csv")
print(data.head())
print(data.info())
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)
data.dropna(inplace=True)
data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})
X = data[
    ["tenure", "MonthlyCharges", "Contract", "InternetService"]
]
y = data["Churn"]
X = pd.get_dummies(
    X,
    columns=["Contract", "InternetService"],
    drop_first=True
)
X = X.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train[["tenure", "MonthlyCharges"]] = scaler.fit_transform(
    X_train[["tenure", "MonthlyCharges"]]
)
X_test[["tenure", "MonthlyCharges"]] = scaler.transform(
    X_test[["tenure", "MonthlyCharges"]]
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
).plot()
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.plot(
    fpr,
    tpr,
    label="ROC Curve (AUC = {:.2f})".format(auc)
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()