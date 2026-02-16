import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# =========================================================
# 1️⃣ Charger et préparer les données
# =========================================================
X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")

X_train_clean = X_train.dropna()
y_train_clean = Y_train.loc[X_train_clean.index].values.ravel()

le = LabelEncoder()
y_train_clean = le.fit_transform(y_train_clean)

X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, stratify=y_train_clean, random_state=42
)

classes = np.unique(y_tr_full)
weights = compute_class_weight("balanced", classes=classes, y=y_tr_full)
sample_weights = np.array([weights[y] for y in y_tr_full])

# =========================================================
# 2️⃣ Définition des modèles de boosting
# =========================================================
xgb = XGBClassifier(objective="multi:softprob", num_class=len(classes), eval_metric="mlogloss",
                    tree_method="hist", random_state=42, n_jobs=-1)
lgbm = lgb.LGBMClassifier(objective="multiclass", n_jobs=-1, random_state=42)
cat = CatBoostClassifier(loss_function="MultiClass", verbose=0, random_state=42)

boosting_models = [xgb, lgbm, cat]
model_names = ["XGB", "LightGBM", "CatBoost"]

# =========================================================
# 3️⃣ Fit les modèles de boosting et collecter leurs prédictions
# =========================================================
train_preds = []
for model, name in zip(boosting_models, model_names):
    print(f"Training {name}...")
    model.fit(X_tr_full, y_tr_full, sample_weight=sample_weights)
    
    # Prédictions probabilités pour stacking
    preds_proba = model.predict_proba(X_tr_full)
    train_preds.append(preds_proba)

stacked_train = np.hstack(train_preds)

# =========================================================
# 4️⃣ Métamodèle = Random Forest
# =========================================================
meta_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

meta_model.fit(stacked_train, y_tr_full)

# =========================================================
# 5️⃣ Évaluation sur le test set
# =========================================================
test_preds = []
for model in boosting_models:
    preds_proba = model.predict_proba(X_test)
    test_preds.append(preds_proba)

stacked_test = np.hstack(test_preds)
y_pred = meta_model.predict(stacked_test)

f1 = f1_score(y_test, y_pred, average="macro")
acc = accuracy_score(y_test, y_pred)

print("\n===== STACKING TEST RESULTS =====")
print("F1-score (macro):", f1)
print("Accuracy:", acc)
print("\nDetailed report:\n")
print(classification_report(y_test, y_pred))