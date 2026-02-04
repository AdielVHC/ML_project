import geopandas as gpd
from feature_engineer import engineer_features

change_type_map = {'Demolition': 0,
                   'Road': 1,
                   'Residential': 2,
                   'Commercial': 3,
                   'Industrial': 4,
                   'Mega Projects': 5}

## Read csvs

train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)
