"""
Script principal de Streamlit.
Para ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_preprocessing import transform_data
from src.flag_analysis import analyze_flags_by_columns

def main():
    """
    Función principal del dashboard de Streamlit.
    """
    st.title("Tablero de Análisis de Moderación")
    st.write("""
    Dashboard para explorar banderas de moderación (Moderado, Fake, Rollback).
    """)
    
    # 1) Cargar y transformar datos
    @st.cache_data
    def load_and_transform():
        """
        Carga el dataset desde config.settings.DATA_PATH, realiza las transformaciones
        y retorna (df, numeric_cols, categorical_cols).
        """
        df, numeric_cols, categorical_cols = transform_data()
        return df, numeric_cols, categorical_cols

    df, numeric_cols, categorical_cols = load_and_transform()

    # Barra lateral
    st.sidebar.header("Parámetros de Análisis")

    # 2) Filtros para variables categóricas
    #    Permitimos al usuario elegir qué variables categóricas desea filtrar.
    st.sidebar.subheader("Filtros - Variables Categóricas")
    selected_cat_filters = st.sidebar.multiselect(
        "Selecciona columnas categóricas para filtrar:",
        options=categorical_cols,
        default=[]
    )

    # Diccionario para guardar las selecciones de cada variable
    cat_filter_values = {}
    for cat_col in selected_cat_filters:
        # Obtenemos valores únicos
        unique_vals = sorted(df[cat_col].dropna().unique().tolist())
        chosen_vals = st.sidebar.multiselect(
            f"Filtrar por {cat_col}:",
            options=unique_vals,
            default=unique_vals  # Por defecto, todos (sin filtrar)
        )
        cat_filter_values[cat_col] = chosen_vals

    # 3) Filtros para variables numéricas
    st.sidebar.subheader("Filtros - Variables Numéricas")
    selected_num_filters = st.sidebar.multiselect(
        "Selecciona columnas numéricas para filtrar:",
        options=numeric_cols,
        default=[]
    )

    num_filter_ranges = {}
    for num_col in selected_num_filters:
        min_val = float(df[num_col].min())
        max_val = float(df[num_col].max())
        cur_range = st.sidebar.slider(
            f"Rango para {num_col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        num_filter_ranges[num_col] = cur_range

    # 4) Selección de columnas de agrupamiento
    st.sidebar.subheader("Columnas para Agrupar")
    grouping_cols = st.sidebar.multiselect(
        "Selecciona columnas categóricas para agrupar:",
        options=categorical_cols,
        default=["site_id"]  # Ejemplo
    )

    # 5) Selección de métricas a graficar
    st.sidebar.subheader("Métricas a Graficar")
    metrics_available = [
        'FreqArticlesGlobal',
        'FreqModeratedByGroup',
        'FreqModeratedGlobal',
        'FreqFakeByGroup',
        'FreqFakeGlobal',
        'FreqRollbackByGroup',
        'FreqRollbackGlobal'
    ]
    selected_metrics = st.sidebar.multiselect(
        "Selecciona las métricas",
        options=metrics_available,
        default=metrics_available
    )

    # 6) Selección de top_n
    top_n = st.sidebar.number_input(
        label="Número de grupos (top N)",
        min_value=1,
        max_value=100,
        value=10
    )

    # Botón para procesar
    ejecutar_btn = st.sidebar.button("Ejecutar Análisis")

    # 7) Lógica de filtrado y análisis
    if ejecutar_btn:
        st.write("### Resultados del Análisis")
        df_filtered = df.copy()

        # Aplicar filtros categóricos
        for col, chosen_vals in cat_filter_values.items():
            df_filtered = df_filtered[df_filtered[col].isin(chosen_vals)]

        # Aplicar filtros numéricos
        for col, (range_min, range_max) in num_filter_ranges.items():
            df_filtered = df_filtered[(df_filtered[col] >= range_min) & (df_filtered[col] <= range_max)]

        # Validar que haya datos tras los filtros
        if len(df_filtered) == 0:
            st.warning("No hay datos que coincidan con los filtros seleccionados.")
            return

        # Validar que haya columnas de agrupamiento
        if not grouping_cols:
            st.error("Debes seleccionar al menos una columna para agrupar.")
            return

        # Llamar a la función de análisis
        stats_df, figs_dict = analyze_flags_by_columns(
            df=df_filtered,
            group_cols=grouping_cols,
            top_n=top_n,
            selected_metrics=selected_metrics
        )

        # Mostrar la tabla de métricas
        st.write("#### Tabla de Métricas Calculadas")
        st.dataframe(stats_df)

        # Mostrar los gráficos
        for metric_name, fig in figs_dict.items():
            st.write(f"**Gráfico para la métrica:** {metric_name}")
            st.pyplot(fig)


if __name__ == "__main__":
    main()