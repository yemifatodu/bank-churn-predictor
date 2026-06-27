"""
train.py - Bank Customer Churn Predictor
Regenerates models/bank_churn_rf_model.pkl and models/bank_preprocessor.pkl
using the current sklearn version in your venv.

Dataset: Kaggle Bank Customer Churn (Churn_Modelling.csv)
Place Churn_Modelling.csv in the same folder as this script, then run:
    python train.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Load dataset ──────────────────────────────────────────────────────────
CSV_PATH = r"C:/Users/HP/OneDrive/Documents/BANK CUSTOMER SEGMENTATION.csv"


print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"   ✅ Loaded {len(df):,} rows")

# ── 2. Select features (must match app.py exactly) ───────────────────────────
FEATURES = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
TARGET = 'Exited'

X = df[FEATURES]
y = df[TARGET]

# ── 3. Build preprocessor ────────────────────────────────────────────────────
categorical_cols = ['Geography', 'Gender']
numerical_cols   = ['CreditScore', 'Age', 'Tenure', 'Balance',
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

# ── 4. Split & preprocess ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("⚙️  Fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# ── 5. Train Random Forest ───────────────────────────────────────────────────
print("🌲 Training Random Forest (this may take ~30 seconds)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_processed, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_processed)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

# ── 7. Save models ───────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(model,        "models/bank_churn_rf_model.pkl")
joblib.dump(preprocessor, "models/bank_preprocessor.pkl")

print("\n✅ Models saved successfully:")
print("   → models/bank_churn_rf_model.pkl")
print("   → models/bank_preprocessor.pkl")
print("\n🚀 Now run: streamlit run app.py")