# Train the Random Forest model and save everything the app needs.
# Run this ONCE before running the app:  python train_model.py
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 1. Load the data ----------
train = pd.read_csv("train.csv")
df = train.copy()

# ---------- 2. Same cleaning as the notebook ----------
# Name: fill missing with 'No Name' + make a flag feature
df["Name"] = df["Name"].fillna("No Name").replace("", "No Name")
df["HasName"] = (df["Name"] != "No Name").astype(int)

# Description: fill missing + make a length feature
df["Description"] = df["Description"].fillna("")
df["DescLength"] = df["Description"].apply(len)

# PetID is useless -> drop. RescuerID -> number of pets this rescuer posted
df = df.drop(columns=["PetID"])
df["RescuerPosts"] = df["RescuerID"].map(df["RescuerID"].value_counts())
df = df.drop(columns=["Name", "Description", "RescuerID"])

# ---------- 3. Same preprocessing as the notebook ----------
X = df.drop(columns=["AdoptionSpeed"])
y = df["AdoptionSpeed"]

cat_cols = ["Type", "Gender", "Color1", "Color2", "Color3", "MaturitySize",
            "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
X = pd.get_dummies(X, columns=cat_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 4. Train the best model (Random Forest) ----------
# We limit the tree depth a little so the saved model stays small (13 MB
# instead of ~320 MB) - the accuracy stays almost the same (43.2% vs 43.3%).
model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
model.fit(X_train_s, y_train)

test_pred = model.predict(X_test_s)
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 3))
print("Macro F1     :", round(f1_score(y_test, test_pred, average="macro"), 3))

# ---------- 5. Save the model + scaler + feature list ----------
# joblib with compression keeps the files small
joblib.dump(model, "model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)
with open("feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("Saved: model.pkl, scaler.pkl, feature_columns.json")
