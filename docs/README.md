# Documentación del proyecto

## Visión general

Este repositorio contiene un proyecto de Machine Learning orientado a la simulación y predicción de ventas de productos deportivos. La solución combina ingeniería de datos, modelado predictivo y una aplicación interactiva en Streamlit para explorar escenarios de descuentos y competencia.

## Objetivo

El objetivo principal es ofrecer una herramienta reproducible que permita:
- procesar y transformar datos históricos de ventas y competencia,
- entrenar un modelo de forecasting sobre series temporales de ventas,
- generar predicciones de unidades vendidas e ingresos proyectados,
- simular estrategias comerciales con descuentos y ajustes de precio frente a la competencia.

## Componentes del proyecto

- `app/app.py`: aplicación Streamlit para simular ventas de noviembre 2025. Permite seleccionar productos, ajustar descuentos y comparar escenarios de precio frente a la competencia.
- `data/raw/`: datos crudos de entrada, incluidos archivos de ventas y competencia para entrenamiento e inferencia.
- `data/processed/`: datos transformados listos para el entrenamiento del modelo y para ejecutar inferencia.
- `models/modelo_final.joblib`: modelo final serializado que se utiliza desde la aplicación.
- `notebooks/`: notebooks de Jupyter con el flujo de trabajo de entrenamiento, análisis exploratorio y desarrollo de la solución.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.

## Flujo de trabajo

1. Extracción y carga de datos desde `data/raw/`.
2. Ingeniería de características y creación de variables temporales, de producto y de competencia.
3. Entrenamiento de un modelo predictivo usando los datos procesados.
4. Serialización del modelo final en `models/modelo_final.joblib`.
5. Ejecución de la aplicación Streamlit para simular distintos escenarios de ventas.

## Qué incluye

- simulación de precios con ajuste de descuento,
- escenarios de competencia (`Amazon`, `Decathlon`, `Deporvillage`),
- predicción iterativa con lags y promedio móvil,
- métricas clave: unidades proyectadas, ingresos proyectados y precio promedio,
- visualización interactiva de la demanda diaria.

## Tecnologías utilizadas

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Cómo ejecutar el proyecto

1. Crear un entorno virtual de Python.
2. Instalar las dependencias con:
   ```bash
   pip install -r requirements.txt
   ```
3. Iniciar la aplicación Streamlit desde la raíz del proyecto:
   ```bash
   streamlit run app/app.py
   ```

## Recomendaciones para GitHub

- Agrega un `README.md` en la raíz del repositorio con un resumen del proyecto y enlaces a esta documentación.
- Incluye un archivo `.gitignore` para excluir `__pycache__`, entornos virtuales y archivos temporales.
- Añade una breve sección de `LICENSE` si deseas compartir el proyecto públicamente.

## LICENSE

Este proyecto puede publicarse con una licencia de código abierto. Una opción común para proyectos de análisis de datos y demostraciones es la licencia MIT, que permite uso, modificación y distribución con atribución.

Si prefieres una licencia más restrictiva, puedes considerar Apache 2.0 o GNU GPL v3. Para un repositorio público, se recomienda agregar también un archivo `LICENSE` en la raíz con el texto completo de la licencia elegida.

## Beneficios del proyecto

Esta solución está diseñada para ayudar a equipos comerciales y analistas a evaluar rápidamente el impacto de promociones y precios competitivos sobre ventas futuras, proporcionando un marco reproducible y fácil de desplegar para experimentación de forecasting.
