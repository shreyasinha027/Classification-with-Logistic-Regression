import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)


file_path = r"C:\Users\Shreya\Documents\Classification with Logistic Regression\healthcare_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print(df.head())


df = df[df["Test Results"].isin(["Normal", "Abnormal"])]

df["Target"] = df["Test Results"].map({
    "Normal": 0,
    "Abnormal": 1
})


drop_cols = [
    "Name",
    "Doctor",
    "Hospital",
    "Date of Admission",
    "Discharge Date",
    "Test Results"
]

df.drop(columns=drop_cols, inplace=True)


label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)



y_pred = model.predict(X_test_scaled)


y_prob = model.predict_proba(X_test_scaled)[:, 1]


print("\nClassification Report")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


custom_threshold = 0.30

y_pred_custom = (y_prob >= custom_threshold).astype(int)

print(f"\nResults with Threshold = {custom_threshold}")

print(
    classification_report(
        y_test,
        y_pred_custom
    )
)

print(
    "Precision:",
    precision_score(y_test, y_pred_custom)
)

print(
    "Recall:",
    recall_score(y_test, y_pred_custom)
)


x = np.linspace(-10, 10, 200)

sigmoid = 1 / (1 + np.exp(-x))

plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

print("\nSigmoid Function:")
print("P(y=1) = 1 / (1 + e^(-z))")
print("It converts any real number into a probability between 0 and 1.")