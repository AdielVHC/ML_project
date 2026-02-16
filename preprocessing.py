import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score



import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin




def preprocessing(df):
    # --- 2. Fonction pour réordonner une ligne ---
    def reorder_row(row):
        
        blocks = []
        
        # Construire les blocs temporels
        for i in range(5):
            block = {
                "date": row[f"date{i}"],
                "change_status": row[f"change_status_date{i}"],
                "red_mean": row[f"img_red_mean_date{i+1}"],
                "green_mean": row[f"img_green_mean_date{i+1}"],
                "blue_mean": row[f"img_blue_mean_date{i+1}"],
                "red_std": row[f"img_red_std_date{i+1}"],
                "green_std": row[f"img_green_std_date{i+1}"],
                "blue_std": row[f"img_blue_std_date{i+1}"],
            }
            blocks.append(block)
        
        # Trier les blocs par date croissante
        blocks = sorted(blocks, key=lambda x: x["date"])
        
        # Réinjecter dans la ligne dans l’ordre trié
        for new_i, block in enumerate(blocks):
            row[f"date{new_i}"] = block["date"]
            row[f"change_status_date{new_i}"] = block["change_status"]
            
            row[f"img_red_mean_date{new_i+1}"] = block["red_mean"]
            row[f"img_green_mean_date{new_i+1}"] = block["green_mean"]
            row[f"img_blue_mean_date{new_i+1}"] = block["blue_mean"]
            
            row[f"img_red_std_date{new_i+1}"] = block["red_std"]
            row[f"img_green_std_date{new_i+1}"] = block["green_std"]
            row[f"img_blue_std_date{new_i+1}"] = block["blue_std"]
        
        return row


    # --- 3. Appliquer à tout le dataframe ---
    df_final = df.apply(reorder_row, axis=1)
    return df_final

df_train = gpd.read_file('2-el-1730-machine-learning-project-2026/train.geojson')
df_test = gpd.read_file('2-el-1730-machine-learning-project-2026/test.geojson')

# --- 1. Convertir les dates en datetime ---
for i in range(5):
    df_train[f"date{i}"] = pd.to_datetime(df_train[f"date{i}"], format="%d-%m-%Y")
    df_test[f"date{i}"] = pd.to_datetime(df_test[f"date{i}"], format="%d-%m-%Y")

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)
df_train.to_file("train.geojson", driver="GeoJSON")
df_test.to_file("test.geojson", driver="GeoJSON")
