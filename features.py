import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score


def features(df, type_df):
    dfn = pd.DataFrame(index=df.index)  

    # --- Dates et intervalles ---
    for i in range(5):
        df[f"date{i}"] = pd.to_datetime(df[f"date{i}"], dayfirst=True, errors='coerce')

    for i in range(4):
        dfn[f'days_{i}_{i+1}'] = (df[f'date{i+1}'] - df[f'date{i}']).dt.days

    dfn['total_days'] = (df['date4'] - df['date0']).dt.days

    # --- Géométrie ---
    dfn['area'] = df.geometry.area
    dfn['perimeter'] = df.geometry.length
    dfn['area_perimeter_ratio'] = dfn['area'] / (dfn['perimeter'] + 1e-10)

    # --- Couleurs ---
    for date in range(1,6):
        for color in ["red", "green", "blue"]:
            dfn[f"img_{color}_mean_date{date}"] = df[f"img_{color}_mean_date{date}"]
            # ratio par rapport à la somme des canaux
            dfn[f"{color}_ratio_date{date}"] = df[f"img_{color}_mean_date{date}"] / (
                df[f"img_red_mean_date{date}"] + df[f"img_green_mean_date{date}"] + df[f"img_blue_mean_date{date}"] + 1e-10
            )
        # VARI
        dfn[f'VARI_date{date}'] = (df[f'img_green_mean_date{date}'] - df[f'img_red_mean_date{date}']) / (
            df[f'img_green_mean_date{date}'] + df[f'img_red_mean_date{date}'] - df[f'img_blue_mean_date{date}'] + 1e-10
        )

    for color in ["red", "green", "blue"]:
        dfn[f"{color}_evolution"] = dfn[f"img_{color}_mean_date5"] - dfn[f"img_{color}_mean_date1"]
        dfn[f"{color}_ratio_evolution"] = dfn[f"{color}_ratio_date5"] - dfn[f"{color}_ratio_date1"]
    dfn["VARI_evolution"] = dfn["VARI_date5"] - dfn["VARI_date1"]

    # --- Urban type ---
    df['urban_type_clean'] = df['urban_type'].str.replace('N,A', 'urban_N/A') 
    urban_dummies = df['urban_type_clean'].str.get_dummies(sep=',')
    urban_dummies.columns = ['urban_' + col.strip().replace(' ', '_') for col in urban_dummies.columns]
    dfn = pd.concat([dfn, urban_dummies], axis=1)

    # --- Geography type ---
    df['geography_type_clean'] = df['geography_type'].str.replace('N,A', 'geo_N/A')
    geo_dummies = df['geography_type_clean'].str.get_dummies(sep=',')
    geo_dummies.columns = ['geo_' + col.strip().replace(' ', '_') for col in geo_dummies.columns]
    dfn = pd.concat([dfn, geo_dummies], axis=1)

    dfn = dfn.fillna(dfn.median(numeric_only=True))
    # --- Retourner ---
    if type_df == "train":
        # S'assurer que l'index correspond
        return dfn.loc[df.index], df["change_type"].loc[df.index]
    return dfn, None


train_df = gpd.read_file('train.geojson')
test_df = gpd.read_file("test.geojson")
X_train, Y_train = features(train_df, "train")
X_test, _ = features(test_df, "")
X_train.to_csv("X_train.csv", index=False)
Y_train.to_csv("Y_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
print(X_train.columns)