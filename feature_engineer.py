import geopandas as gpd
import pandas as pd
import numpy as np

def engineer_features(gdf, train=True):

    df = pd.DataFrame()

    #######
    #Label#
    #######

    if train:
        df['target_label'] = gdf['change_type']

    ######################
    #features temporelles#
    ######################

    # Conversion des dates et création d'un mapping
    dates_data = []
    for i in range(5):
        date_col = f'date{i}'
        dates_data.append({
            'date': pd.to_datetime(gdf[date_col], dayfirst=True),
            'original_idx': i
        })
    
    # Pour chaque ligne, trier les dates et garder la trace de l'ordre
    sorted_indices = []
    sorted_dates_list = []
    
    for row_idx in range(len(gdf)):
        row_dates = [(dates_data[i]['date'].iloc[row_idx], i) for i in range(5)]
        row_dates_sorted = sorted(row_dates, key=lambda x: x[0])
        sorted_dates_list.append([d[0] for d in row_dates_sorted])
        sorted_indices.append([d[1] for d in row_dates_sorted])
    
    sorted_indices = np.array(sorted_indices)
    
    # Stocker les dates triées
    for i in range(5):
        df[f'date{i}'] = [sorted_dates_list[row][i] for row in range(len(gdf))]

    # Nombre de jours entre deux dates consécutives
    for i in range(4):
        df[f'days_{i}_{i+1}'] = (df[f'date{i+1}'] - df[f'date{i}']).dt.days

    # ✅ CORRECTION : total_days
    df['total_days'] = (df['date4'] - df['date0']).dt.days

    # Features temporelles supplémentaires
    intervals = np.array([df[f'days_{i}_{i+1}'].values for i in range(4)]).T
    df['interval_std'] = np.std(intervals, axis=1)
    df['interval_mean'] = np.mean(intervals, axis=1)
    df['interval_max'] = np.max(intervals, axis=1)
    df['interval_min'] = np.min(intervals, axis=1)
    
    df['start_month'] = df['date0'].dt.month
    df['end_month'] = df['date4'].dt.month
    df['start_season'] = (df['date0'].dt.month % 12 + 3) // 3
    df['end_season'] = (df['date4'].dt.month % 12 + 3) // 3
    
    df['start_year'] = df['date0'].dt.year
    df['end_year'] = df['date4'].dt.year
    df['year_span'] = df['end_year'] - df['start_year']

    #######################
    #features géométriques#
    #######################

    df['area'] = gdf.geometry.area
    df['perimeter'] = gdf.geometry.length
    df['area_perimeter_ratio'] = df['area'] / (df['perimeter'] + 1e-10)

    #####################
    #features spectrales#
    #####################

    # Réorganiser les données spectrales selon l'ordre des dates triées
    for sorted_pos in range(5):
        for color in ['red', 'green', 'blue']:
            mean_values = []
            std_values = []
            
            for row_idx in range(len(gdf)):
                original_idx = sorted_indices[row_idx, sorted_pos]
                mean_col = f'img_{color}_mean_date{original_idx + 1}'
                std_col = f'img_{color}_std_date{original_idx + 1}'
                mean_values.append(gdf[mean_col].iloc[row_idx])
                std_values.append(gdf[std_col].iloc[row_idx])
            
            df[f'img_{color}_mean_date{sorted_pos + 1}'] = mean_values
            df[f'img_{color}_std_date{sorted_pos + 1}'] = std_values

    # Features spectrales dérivées
    for i in range(1, 6):
        df[f'intensity_mean_date{i}'] = (
            df[f'img_red_mean_date{i}'] + 
            df[f'img_green_mean_date{i}'] + 
            df[f'img_blue_mean_date{i}']
        ) / 3

        df[f'color_variance_date{i}'] = (
            df[f'img_red_std_date{i}']**2 + 
            df[f'img_green_std_date{i}']**2 + 
            df[f'img_blue_std_date{i}']**2
        )

        df[f'greenness_date{i}'] = (
            df[f'img_green_mean_date{i}'] - df[f'img_red_mean_date{i}']
        ) / (df[f'img_green_mean_date{i}'] + df[f'img_red_mean_date{i}'] + 1e-10)
        
        df[f'red_blue_ratio_date{i}'] = df[f'img_red_mean_date{i}'] / (df[f'img_blue_mean_date{i}'] + 1e-10)

        df[f'texture_date{i}'] = (
            df[f'img_red_std_date{i}'] + 
            df[f'img_green_std_date{i}'] + 
            df[f'img_blue_std_date{i}']
        ) / 3

    # Évolution spectrale entre dates
    for i in range(1, 5):
        df[f'intensity_change_{i}_{i+1}'] = df[f'intensity_mean_date{i+1}'] - df[f'intensity_mean_date{i}']
        df[f'intensity_change_rate_{i}_{i+1}'] = df[f'intensity_change_{i}_{i+1}'] / (df[f'days_{i-1}_{i}'] + 1)
        
        df[f'variance_change_{i}_{i+1}'] = df[f'color_variance_date{i+1}'] - df[f'color_variance_date{i}']
        df[f'greenness_change_{i}_{i+1}'] = df[f'greenness_date{i+1}'] - df[f'greenness_date{i}']
        df[f'texture_change_{i}_{i+1}'] = df[f'texture_date{i+1}'] - df[f'texture_date{i}']
        
        # ✅ SUPPRIMÉ : saturation_change (saturation_date n'existe pas)

    #######################
    #features catégoriques#
    #######################

    gdf['urban_type_clean'] = gdf['urban_type'].str.replace('N,A', 'urban_N/A')
    urban_dummies = gdf['urban_type_clean'].str.get_dummies(sep=',')
    urban_dummies.columns = ['urban_' + col.strip().replace(' ', '_') for col in urban_dummies.columns]
    df = pd.concat([df, urban_dummies], axis=1)
    
    gdf['geography_type_clean'] = gdf['geography_type'].str.replace('N,A', 'geo_N/A')
    geo_dummies = gdf['geography_type_clean'].str.get_dummies(sep=',')
    geo_dummies.columns = ['geo_' + col.strip().replace(' ', '_') for col in geo_dummies.columns]
    df = pd.concat([df, geo_dummies], axis=1)
    
    df['num_urban_types'] = gdf['urban_type_clean'].str.split(',').str.len()
    df['num_geo_types'] = gdf['geography_type_clean'].str.split(',').str.len()
    
    # Réorganiser les change_status selon l'ordre des dates triées
    for sorted_pos in range(5):
        status_values = []
        
        for row_idx in range(len(gdf)):
            original_idx = sorted_indices[row_idx, sorted_pos]
            status_col = f'change_status_date{original_idx}'
            status_values.append(gdf[status_col].iloc[row_idx])
        
        status_dummies = pd.get_dummies(status_values, prefix=f'status_{sorted_pos}')
        df = pd.concat([df, status_dummies], axis=1)
    
    # ✅ CORRECTION : Evolution du statut ligne par ligne
    status_changes_list = []
    for row_idx in range(len(gdf)):
        count = 0
        for i in range(4):
            orig_idx_i = sorted_indices[row_idx, i]
            orig_idx_ip1 = sorted_indices[row_idx, i+1]
            status_i = gdf[f'change_status_date{orig_idx_i}'].iloc[row_idx]
            status_ip1 = gdf[f'change_status_date{orig_idx_ip1}'].iloc[row_idx]
            if status_i != status_ip1:
                count += 1
        status_changes_list.append(count)
    
    df['num_status_changes'] = status_changes_list
    df['status_stable'] = (df['num_status_changes'] == 0).astype(int)

    ########################################
    # Suppression des features non utilisées
    ########################################

    df = df.drop(columns=[f'date{i}' for i in range(5)])
  
    return df

train_gdf = gpd.read_file('train.geojson')
df = engineer_features(train_gdf)

df.to_csv('data.csv')
