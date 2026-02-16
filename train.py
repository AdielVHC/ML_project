import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer
from feature_engineer import engineer_features

## Chargement des données
print("Chargement...")
train_df_raw = engineer_features(gpd.read_file('train.geojson'))
test_df_raw = engineer_features(gpd.read_file('test.geojson'), train=False)

train_y = train_df_raw['target_label']
train_X = train_df_raw.drop(columns=['target_label'])
test_X = test_df_raw.copy()

# Aligner colonnes
for col in set(train_X.columns) - set(test_X.columns):
    test_X[col] = 0
for col in set(test_X.columns) - set(train_X.columns):
    train_X[col] = 0
test_X = test_X[train_X.columns]

## Preprocessing
print("Preprocessing...")
# Imputation
imputer = SimpleImputer(strategy='median')
train_X = pd.DataFrame(imputer.fit_transform(train_X), columns=train_X.columns)
test_X = pd.DataFrame(imputer.transform(test_X), columns=test_X.columns)

# Normalisation
scaler = MinMaxScaler()
train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)
test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

## Modèle avec class_weight balanced
print("\nEntraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=40,
    class_weight='balanced',  # Gère automatiquement le déséquilibre
    random_state=42,
    n_jobs=-1,
    verbose=1
)

## Cross-validation
print("\nCross-validation 5-fold...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='macro')
scores = cross_val_score(rf, train_X, train_y, cv=cv, scoring=f1_scorer, n_jobs=-1)

print(f"\nF1 Score moyen: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Scores par fold: {[f'{s:.4f}' for s in scores]}")

## Entraînement final
rf.fit(train_X, train_y)

## Prédictions
predictions = rf.predict(test_X)

submission = pd.DataFrame({
    'Id': range(len(predictions)),
    'change_type': predictions
})

submission.to_csv('sample_submission.csv', index=False)
