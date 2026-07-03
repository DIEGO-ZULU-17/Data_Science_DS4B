import pandas as pd
import joblib
from pathlib import Path

path = Path('data/processed/inferencia_df_transformado.csv')
print('data exists', path.exists())
if path.exists():
    df = pd.read_csv(path)
    print('rows', len(df))
    print('columns sample', list(df.columns)[:40])
    print('lag cols present', all(f'lag_{i}' in df.columns for i in range(1, 8)), 'rolling', 'rolling_mean_7' in df.columns)
    print('unique productos', df['nombre'].nunique() if 'nombre' in df.columns else 'no nombre')
    if 'nombre' in df.columns:
        print('first nombres', df['nombre'].unique()[:10])

model_path = Path('models/modelo_final.joblib')
print('model exists', model_path.exists())
if model_path.exists():
    model = joblib.load(model_path)
    try:
        print('feature count', len(model.feature_names_in_))
        print('feature sample', model.feature_names_in_[:40])
    except Exception as e:
        print('no feature_names_in_', e)
