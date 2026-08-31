import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# =========================================================
# 1. CREATE FOLDERS
# =========================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================================================
# 2. LOAD DATASET
# =========================================================

DATA_FILE = "heart_failure_clinical_records_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("HEART FAILURE DETECTION PROJECT")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# 3. DATA INFORMATION
# =========================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# Remove duplicates
df = df.drop_duplicates()

# =========================================================
# 4. TARGET DISTRIBUTION
# =========================================================

print("\nDEATH_EVENT Distribution:")
print(df["DEATH_EVENT"].value_counts())

print("\nDEATH_EVENT Percentage:")
print(df["DEATH_EVENT"].value_counts(normalize=True) * 100)

# =========================================================
# 5. EDA - DEATH EVENT
# =========================================================

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="DEATH_EVENT")
plt.title("Death Event Distribution")
plt.xlabel("Death Event (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("outputs/death_event_distribution.png")
plt.close()

# =========================================================
# 6. EDA - AGE
# =========================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png")
plt.close()

# =========================================================
# 7. AGE VS DEATH EVENT
# =========================================================

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="DEATH_EVENT", y="age")
plt.title("Age vs Death Event")
plt.xlabel("Death Event")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("outputs/age_vs_death.png")
plt.close()

# =========================================================
# 8. EJECTION FRACTION VS DEATH EVENT
# =========================================================

plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="DEATH_EVENT",
    y="ejection_fraction"
)
plt.title("Ejection Fraction vs Death Event")
plt.xlabel("Death Event")
plt.ylabel("Ejection Fraction")
plt.tight_layout()
plt.savefig("outputs/ejection_fraction_vs_death.png")
plt.close()

# =========================================================
# 9. SERUM CREATININE VS DEATH EVENT
# =========================================================

plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="DEATH_EVENT",
    y="serum_creatinine"
)
plt.title("Serum Creatinine vs Death Event")
plt.xlabel("Death Event")
plt.ylabel("Serum Creatinine")
plt.tight_layout()
plt.savefig("outputs/creatinine_vs_death.png")
plt.close()

# =========================================================
# 10. CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(12, 9))
correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# =========================================================
# 11. FEATURES AND TARGET
# =========================================================

features = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time"
]

X = df[features]
y = df["DEATH_EVENT"]

# =========================================================
# 12. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# =========================================================
# 13. CREATE MODELS
# =========================================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

# =========================================================
# 14. TRAIN AND EVALUATE
# =========================================================

results = []

trained_models = {}

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print("\n", name)
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

# =========================================================
# 15. RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

# =========================================================
# 16. MODEL COMPARISON GRAPH
# =========================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.close()

# =========================================================
# 17. RANDOM FOREST EVALUATION
# =========================================================

rf_model = trained_models["Random Forest"]

rf_prediction = rf_model.predict(X_test)

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        rf_prediction,
        zero_division=0
    )
)

# =========================================================
# 18. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    rf_prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

# =========================================================
# 19. ROC CURVE
# =========================================================

rf_probability = rf_model.predict_proba(
    X_test
)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probability
)

auc_score = roc_auc_score(
    y_test,
    rf_probability
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest AUC = {auc_score:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png"
)

plt.close()

print("\nRandom Forest ROC-AUC:", round(auc_score, 4))

# =========================================================
# 20. FEATURE IMPORTANCE
# =========================================================

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)

plt.figure(figsize=(9, 7))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()

# =========================================================
# 21. SAVE BEST MODEL
# =========================================================

joblib.dump(
    rf_model,
    "models/heart_failure_model.pkl"
)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModel saved:")
print("models/heart_failure_model.pkl")

print("\nGraphs saved in:")
print("outputs/")

print("\nNow run:")
print("python predict.py")