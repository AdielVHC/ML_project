import geopandas as gpd
import pandas as pd
import numpy as np



def engineer_features(gdf):

    df = pd.DataFrame()

    ######################
    #features temporelles#
    ######################

    #dates pures
    for i in range(5):
        colonne = f'date{i}'
        df[colonne] = pd.to_datetime(gdf[colonne], dayfirst=True)

    #nombre de jours entre deux dates consécutives

    for i in range(4):
        df[f'days_{i}_{i+1}'] = (df[f'date{i+1}'] - df[f'date{i}']).dt.days

    # nombre de jours entre les deux dates extrêmes

    df['total_days'] = (df['date4'] - df['date4']).dt.days

    #######################
    #features géométriques#
    #######################

    #Aire
    df['area'] = gdf.geometry.area
    #Périmètre
    df['perimeter'] = gdf.geometry.length
    #compacité
    df['area_perimeter_ratio'] = df['area'] / (df['perimeter'] + 1e-10)

    #####################
    #features spectrales#
    #####################

    #couleurs par date

    for i in range(1,6):
        for col in ['red', 'green', 'blue']:
            colonne1 = f'img_{col}_mean_date{i}'
            colonne2 = f'img_{col}_std_date{i}'
            df[colonne1] = gdf[colonne1]
            df[colonne2] = gdf[colonne2]


    #######################
    #features catégoriques#
    #######################

    # Urban Type
    gdf['urban_type_clean'] = gdf['urban_type'].str.replace('N,A', 'urban_N/A') #pour éviter le problème de séparateur
    urban_dummies = gdf['urban_type_clean'].str.get_dummies(sep=',') #encodage one-hot
    urban_dummies.columns = ['urban_' + col.strip().replace(' ', '_') for col in urban_dummies.columns]
    df = pd.concat([df, urban_dummies], axis=1)
    
    # Geography Type
    gdf['geography_type_clean'] = gdf['geography_type'].str.replace('N,A', 'geo_N/A') #même chose qu'en haut
    geo_dummies = gdf['geography_type_clean'].str.get_dummies(sep=',')
    geo_dummies.columns = ['geo_' + col.strip().replace(' ', '_') for col in geo_dummies.columns]
    df = pd.concat([df, geo_dummies], axis=1)
    
    # Change Status (toutes les dates)
    for i in range(5):
        status_col = f'change_status_date{i}'
        status_dummies = pd.get_dummies(gdf[status_col], prefix=f'status_{i}')
        df = pd.concat([df, status_dummies], axis=1)
  
    return df

train_gdf = gpd.read_file('train.geojson')
df = engineer_features(train_gdf)
df.to_csv('data.csv')