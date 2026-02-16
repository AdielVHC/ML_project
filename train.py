import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier

# =========================================================
# 1️⃣ Chargement et préparation des données
# =========================================================
X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")

# Nettoyage NaN
X_train_clean = X_train.dropna()
y_train_clean = Y_train.loc[X_train_clean.index].values.ravel()

# Encodage des labels
le = LabelEncoder()
y_train_clean = le.fit_transform(y_train_clean)

# Split TRAIN / TEST
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, stratify=y_train_clean, random_state=42
)

# Calcul des poids de classes
classes = np.unique(y_tr_full)
weights = compute_class_weight("balanced", classes=classes, y=y_tr_full)
sample_weights = np.array([weights[y] for y in y_tr_full])

print("Train shape:", X_tr_full.shape)
print("Test shape:", X_test.shape)

# =========================================================
# 2️⃣ Définition des modèles (overfit volontairement)
# =========================================================

# ----- XGBoost -----
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    n_estimators=2000,   # beaucoup d'arbres
    max_depth=10,        # profond
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0
)
xgb.fit(X_tr_full, y_tr_full, sample_weight=sample_weights)

# ----- LightGBM -----
lgbm = lgb.LGBMClassifier(
    objective="multiclass",
    n_jobs=-1,
    random_state=42,
    n_estimators=2000,
    max_depth=10,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0
)
lgbm.fit(X_tr_full, y_tr_full, sample_weight=sample_weights)

# ----- CatBoost -----
train_pool = Pool(
    data=X_tr_full,
    label=y_tr_full,
    weight=sample_weights
)

cat = CatBoostClassifier(
    loss_function="MultiClass",
    random_state=42,
    iterations=2000,
    depth=10,
    learning_rate=0.05,
    bootstrap_type='Bernoulli',  # compatible subsample
    subsample=1.0,
    verbose=100
)
cat.fit(train_pool)

# =========================================================
# 3️⃣ Génération des features pour le métamodèle (stacking)
# =========================================================
train_preds = [
    xgb.predict_proba(X_tr_full),
    lgbm.predict_proba(X_tr_full),
    cat.predict_proba(X_tr_full)
]
stacked_train = np.hstack(train_preds)

# Métamodèle = Random Forest
meta_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
)
meta_model.fit(stacked_train, y_tr_full)

# =========================================================
# 4️⃣ Évaluation sur le test set
# =========================================================
test_preds = [
    xgb.predict_proba(X_test),
    lgbm.predict_proba(X_test),
    cat.predict_proba(X_test)
]
stacked_test = np.hstack(test_preds)
y_pred = meta_model.predict(stacked_test)

f1 = f1_score(y_test, y_pred, average="macro")
acc = accuracy_score(y_test, y_pred)

print("\n===== STACKING TEST RESULTS =====")
print("F1-score (macro):", f1)
print("Accuracy:", acc)
print("\nDetailed report:\n")
print(classification_report(y_test, y_pred))