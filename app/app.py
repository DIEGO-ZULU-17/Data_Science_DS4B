import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Simulación de Ventas Nov 2025", page_icon="🛍️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_final.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"

sns.set_theme(style="whitegrid", palette=["#667eea", "#764ba2"])


def load_resources():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "feature_names_in_"):
            raise AttributeError("El modelo cargado no define feature_names_in_")
    except Exception as exc:
        st.error(f"Error cargando el modelo: {exc}")
        st.stop()

    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo de inferencia en {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        st.error(f"Error cargando los datos: {exc}")
        st.stop()

    return model, df


model, df = load_resources()


def format_currency(value):
    return f"€{value:,.2f}"


def ensure_feature_columns(data: pd.DataFrame, feature_cols):
    df_work = data.copy()
    missing = [col for col in feature_cols if col not in df_work.columns]
    for col in missing:
        if col in ["Amazon", "Decathlon", "Deporvillage"]:
            df_work[col] = df_work.get("precio_competencia", 0.0)
        else:
            df_work[col] = 0.0
    return df_work


def build_competition_columns(data: pd.DataFrame, factor: float) -> pd.DataFrame:
    df_work = data.copy()
    competition_cols = [col for col in ["Amazon", "Decathlon", "Deporvillage"] if col in df_work.columns]
    if competition_cols:
        for col in competition_cols:
            df_work[col] = df_work[col] * (1 + factor)
        df_work["precio_competencia"] = df_work[competition_cols].mean(axis=1)
    else:
        df_work["precio_competencia"] = df_work["precio_competencia"] * (1 + factor)
    return df_work


def recursive_prediction(data: pd.DataFrame) -> pd.DataFrame:
    df_work = data.copy().reset_index(drop=True)
    feature_cols = list(model.feature_names_in_)
    df_work = ensure_feature_columns(df_work, feature_cols)
    missing = [col for col in feature_cols if col not in df_work.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas del modelo: {missing}")

    predictions = []
    for idx in range(len(df_work)):
        X = df_work.loc[[idx], feature_cols].astype(float)
        pred = model.predict(X)[0]
        predictions.append(pred)

        if idx + 1 < len(df_work):
            for lag_i in range(7, 1, -1):
                df_work.at[idx + 1, f"lag_{lag_i}"] = df_work.at[idx, f"lag_{lag_i - 1}"]
            df_work.at[idx + 1, "lag_1"] = pred
            last7 = predictions[-7:]
            if last7:
                df_work.at[idx + 1, "rolling_mean_7"] = float(np.mean(last7))

    df_work["unidades_predichas"] = np.round(predictions).astype(int)
    df_work["ingresos_proyectados"] = df_work["unidades_predichas"] * df_work["precio_venta"]
    return df_work


def prepare_table(df_table: pd.DataFrame) -> pd.DataFrame:
    display = df_table.copy()
    display["precio_venta"] = display["precio_venta"].map(format_currency)
    display["precio_competencia"] = display["precio_competencia"].map(format_currency)
    display["ingresos_proyectados"] = display["ingresos_proyectados"].map(format_currency)
    display["descuento_porcentaje"] = display["descuento_porcentaje"].map("{:.0f}%".format)
    display["unidades_predichas"] = display["unidades_predichas"].map("{:,.0f}".format)
    return display


productos = sorted(df["nombre"].unique())

with st.sidebar:
    st.markdown("# Controles de Simulación")
    producto_seleccionado = st.selectbox("Producto", productos)
    descuento_pct = st.slider("Ajuste de descuento", -50, 50, 0, step=5, format="%d%%")
    escenario_competencia = st.radio(
        "Escenario de competencia",
        ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
    )
    ejecutar = st.button("Simular Ventas")

st.markdown("---")

if not ejecutar:
    st.info("Selecciona un producto, ajusta el descuento y el escenario, luego pulsa **Simular Ventas**.")
    st.stop()

producto_df = df[df["nombre"] == producto_seleccionado].copy()
if producto_df.empty:
    st.error("No se encontró información para el producto seleccionado.")
    st.stop()

producto_df = producto_df.sort_values("fecha").reset_index(drop=True)
producto_df["precio_venta"] = producto_df["precio_base"] * (1 + descuento_pct / 100)
producto_df["descuento_pct"] = descuento_pct
producto_df["descuento_porcentaje"] = descuento_pct
producto_df["ratio_precio"] = producto_df["precio_venta"] / producto_df["precio_competencia"].replace(0, np.nan)
producto_df["ratio_precio"] = producto_df["ratio_precio"].fillna(0)

escenarios = [
    ("Actual (0%)", 0.0),
    ("Competencia -5%", -0.05),
    ("Competencia +5%", 0.05)
]

with st.spinner("Simulando ventas y actualizando lags día a día..."):
    results = {}
    for nombre, factor in escenarios:
        df_scenario = build_competition_columns(producto_df, factor)
        df_scenario["ratio_precio"] = df_scenario["precio_venta"] / df_scenario["precio_competencia"].replace(0, np.nan)
        df_scenario["ratio_precio"] = df_scenario["ratio_precio"].fillna(0)
        results[nombre] = recursive_prediction(df_scenario)

selected_df = results[escenario_competencia]

unidades_totales = int(selected_df["unidades_predichas"].sum())
ingresos_totales = float(selected_df["ingresos_proyectados"].sum())
precio_promedio_venta = float(selected_df["precio_venta"].mean())
descuento_promedio = float(selected_df["descuento_porcentaje"].mean())

st.title(f"📈 Simulación de Ventas - Noviembre 2025")
st.subheader(f"Producto: {producto_seleccionado}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unidades totales proyectadas", f"{unidades_totales:,d}")
c2.metric("Ingresos proyectados", format_currency(ingresos_totales))
c3.metric("Precio promedio de venta", format_currency(precio_promedio_venta))
c4.metric("Descuento promedio", f"{descuento_promedio:.0f}%")

st.markdown("---")

fig, ax = plt.subplots(figsize=(12, 5))

sns.lineplot(
    x="dia_mes",
    y="unidades_predichas",
    data=selected_df,
    marker="o",
    color="#667eea",
    linewidth=2.5,
    ax=ax
)

black_friday_row = selected_df[selected_df["es_black_friday"] == 1]
if not black_friday_row.empty:
    bf_day = int(black_friday_row["dia_mes"].iloc[0])
    bf_units = int(black_friday_row["unidades_predichas"].iloc[0])
    ax.axvline(bf_day, color="#764ba2", linestyle="--", alpha=0.9)
    ax.scatter([bf_day], [bf_units], color="red", s=100, zorder=5)
    ax.annotate(
        "Black Friday",
        xy=(bf_day, bf_units),
        xytext=(bf_day + 1, bf_units * 1.07),
        arrowprops=dict(color="red", arrowstyle="->"),
        color="red",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title("Predicción diaria de unidades vendidas", fontsize=16, color="#333333")
ax.set_xlabel("Día de noviembre", fontsize=12)
ax.set_ylabel("Unidades predichas", fontsize=12)
ax.set_xticks(range(1, 31, 2))
ax.grid(True, alpha=0.25)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.markdown("### 📅 Tabla detallada de noviembre")
use_cols = [
    "fecha",
    "dia_semana",
    "precio_venta",
    "precio_competencia",
    "descuento_porcentaje",
    "unidades_predichas",
    "ingresos_proyectados"
]

display_df = selected_df[use_cols].copy()
display_df = prepare_table(display_df)

styled_df = display_df.style.apply(
    lambda row: [
        "background-color: #fff0f5; font-weight: bold;" if row["fecha"] == "2025-11-28" else ""
        for _ in row
    ],
    axis=1
)

st.write(styled_df)

st.markdown("---")

st.markdown("### 🔍 Comparativa de escenarios")
comparativa = []
for nombre, df_scenario in results.items():
    total_units = int(df_scenario["unidades_predichas"].sum())
    total_revenue = float(df_scenario["ingresos_proyectados"].sum())
    comparativa.append((nombre, total_units, total_revenue))

col1, col2, col3 = st.columns(3)
for column, (nombre, unidades, ingresos) in zip((col1, col2, col3), comparativa):
    column.metric(
        nombre,
        f"{unidades:,d} unidades",
        format_currency(ingresos)
    )

st.markdown("---")

st.info(
    "La aplicación ha generado predicciones recursivas día a día para noviembre 2025. Cambia el descuento y el escenario para comparar cómo impacta en las ventas y en los ingresos."
)
