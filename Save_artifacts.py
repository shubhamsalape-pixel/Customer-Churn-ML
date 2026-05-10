# save_artifacts.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# CREATE ARTIFACTS DIRECTORY
# =========================================================

os.makedirs("artifacts", exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("Telco-Customer-Churn - Telco-Customer-Churn.csv")

# =========================================================
# BASIC CLEANING
# =========================================================

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# Replace inconsistent values
replace_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for col in replace_cols:
    df[col] = df[col].replace(
        "No internet service",
        "No"
    )

df["MultipleLines"] = df["MultipleLines"].replace(
    "No phone service",
    "No"
)

# =========================================================
# TARGET ENCODING
# =========================================================

label_encoder = LabelEncoder()

df["Churn"] = label_encoder.fit_transform(df["Churn"])

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================================================
# COLUMN GROUPS
# =========================================================

ordinal_cols = [
    "InternetService",
    "Contract"
]

ordinal_categories = [
    ["No", "DSL", "Fiber optic"],
    ["Month-to-month", "One year", "Two year"]
]

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_cols = [
    col for col in X.columns
    if col not in ordinal_cols + numeric_cols
]

# =========================================================
# PREPROCESSING PIPELINES
# =========================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=ordinal_categories))
])

# =========================================================
# COLUMN TRANSFORMER
# =========================================================

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
    ("ord", ordinal_pipeline, ordinal_cols)
])

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FULL PIPELINE
# =========================================================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=20,
        criterion="gini",
        random_state=42
    ))
])

# =========================================================
# TRAIN MODEL
# =========================================================

model_pipeline.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================

y_pred = model_pipeline.predict(X_test)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# EXTRACT PREPROCESSOR
# =========================================================

trained_preprocessor = model_pipeline.named_steps["preprocessor"]

# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = trained_preprocessor.get_feature_names_out()

# =========================================================
# SAVE ARTIFACTS
# =========================================================

joblib.dump(model_pipeline, "artifacts/model.pkl")

joblib.dump(
    trained_preprocessor,
    "artifacts/preprocessor.pkl"
)

joblib.dump(
    label_encoder,
    "artifacts/label_encoder.pkl"
)

joblib.dump(
    feature_names,
    "artifacts/feature_columns.pkl"
)

# =========================================================
# SAVE METADATA
# =========================================================

metadata = {
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "ordinal_columns": ordinal_cols,
    "target_column": "Churn",
    "model_type": "RandomForestClassifier"
}

joblib.dump(
    metadata,
    "artifacts/metadata.pkl"
)

# =========================================================
# TEST ARTIFACT LOADING
# =========================================================

loaded_model = joblib.load("artifacts/model.pkl")

sample_prediction = loaded_model.predict(
    X_test.iloc[:5]
)

print("\nArtifacts Generated Successfully!")

print("\nSaved Files:")
print("- model.pkl")
print("- preprocessor.pkl")
print("- label_encoder.pkl")
print("- feature_columns.pkl")
print("- metadata.pkl")