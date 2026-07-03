import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path('models/modelo_final.joblib')
DATA_PATH = Path('data/processed/inferencia_df_transformado.csv')

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
print('rows', len(df))
print('unique productos', df['nombre'].nunique())
product_name = df['nombre'].unique()[0]
print('sample product', product_name)
producto_df = df[df['nombre'] == product_name].sort_values('fecha').reset_index(drop=True)
print('product rows', len(producto_df))
producto_df['precio_venta'] = producto_df['precio_base'] * (1 + 0.0)
producto_df['descuento_pct'] = 0.0
producto_df['descuento_porcentaje'] = 0.0
producto_df['ratio_precio'] = producto_df['precio_venta'] / producto_df['precio_competencia'].replace(0, np.nan)
producto_df['ratio_precio'] = producto_df['ratio_precio'].fillna(0)
print('precio_competencia nulls', producto_df['precio_competencia'].isna().sum())
print('first row', producto_df.iloc[0][['lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7','rolling_mean_7']].to_dict())
feature_cols = list(model.feature_names_in_)
print('missing features', [c for c in feature_cols if c not in producto_df.columns])

predictions = []
for idx in range(len(producto_df)):
    X = producto_df.loc[idx, feature_cols].to_numpy().reshape(1, -1).astype(float)
    p = model.predict(X)[0]
    predictions.append(p)
    if idx + 1 < len(producto_df):
        for lag_i in range(7, 1, -1):
            producto_df.at[idx + 1, f'lag_{lag_i}'] = producto_df.at[idx, f'lag_{lag_i-1}']
        producto_df.at[idx + 1, 'lag_1'] = p
        producto_df.at[idx + 1, 'rolling_mean_7'] = float(np.mean(predictions[-7:]))
print('predicted rows', len(predictions))
print('sample predictions', predictions[:5])
print('final row lags', producto_df.loc[1, ['lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7','rolling_mean_7']].to_dict())
print('success')
