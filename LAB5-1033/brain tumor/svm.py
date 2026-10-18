
# SVM Brain Tumor Detection
#hafsa-1033
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
path = r"C:\Users\GEO COMPUTERS\Desktop\LAB5-1033\brain tumor"
X = []
y = []

# 2. Preprocess images
for label, folder in enumerate(["no", "yes"]):

    folder_path = os.path.join(path, folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        # Convert to grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Resize to 64x64
        img = cv2.resize(img, (64, 64))

        # Flatten image
        img = img.flatten()

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Image features:", X.shape)

# 3. Split dataset 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train SVM
svm = SVC(kernel="rbf", C=10, gamma="scale")

svm.fit(X_train, y_train)

# 6. Prediction
y_pred = svm.predict(X_test_scaled)

# 7. Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# 8. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Tumor", "Tumor"]
))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Tumor", "Tumor"]
)

disp.plot(cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()

# 10. Display test images
plt.figure(figsize=(10, 6))

for i in range(min(6, len(X_test))):

    # Original image before scaling
    img = X_test[i].reshape(64, 64)

    actual = "Tumor" if y_test[i] == 1 else "No Tumor"
    predicted = "Tumor" if y_pred[i] == 1 else "No Tumor"

    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap="gray")

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
