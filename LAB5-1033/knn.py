
# Activity 1: KNN Credit Card Fraud Detection
# Hafsa-1033

import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Step 1: Load dataset
path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "creditcard.csv"
)

df = pd.read_csv(path)

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Step 2: Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Use a smaller training sample for faster execution
X_train, _, y_train, _ = train_test_split(
    X_train, y_train,
    train_size=10000,
    random_state=42,
    stratify=y_train
)

# Step 4: Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Apply SMOTE on training data
smote = SMOTE(random_state=42)

X_balanced, y_balanced = smote.fit_resample(
    X_train, y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_balanced).value_counts())

# Step 6: Train KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_balanced, y_balanced)

# Step 7: Make predictions
y_pred = knn.predict(X_test)

# Step 8: Evaluate KNN
print("\nKNN with K=5")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))

# Step 9: Tune K using GridSearchCV
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier())
])

parameters = {
    "knn__n_neighbors": [3, 5, 7]
}

grid = GridSearchCV(
    pipeline,
    parameters,
    cv=3,
    scoring="f1",
    n_jobs=1
)

grid.fit(X_train, y_train)

print("\nBest K:", grid.best_params_)

# Step 10: Evaluate tuned model
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("\nTuned KNN Results")
print("Accuracy:", accuracy_score(y_test, best_pred))
print("F1-Score:", f1_score(y_test, best_pred, zero_division=0))
