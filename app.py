# app.py

"""
Script principal de Streamlit.
Para ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd

# Módulos lógicos
from src.data_preprocessing import transform_data
from src.flag_analysis import analyze_flags_by_columns
from src.relationships import boxplot_by_categories
from src.pca_analysis import perform_pca_mixed
from src.cluster_analysis import perform_hierarchical_clustering, compute_cluster_insights
from src.classification import run_fake_classification

# Módulo de visualización
from src.visualizations import plot_pca_biplot, plot_pca_clusters


def main():
    st.title("Tablero de Análisis")

    @st.cache_data
    def load_dataset():
        df, numeric_cols, categorical_cols = transform_data()
        return df, numeric_cols, categorical_cols

    df, numeric_cols, categorical_cols = load_dataset()

    # Cinco pestañas
    (
        tab_flags,
        tab_rels,
        tab_pca,
        tab_class,
        # tab_opt
        ) = st.tabs([
        "Análisis de Banderas",
        "Relaciones / Correlaciones",
        "PCA + Clustering",
        "Clasificación",
        # "Optimización" 
    ])

    with tab_flags:
        show_flags_tab(df, numeric_cols, categorical_cols)

    with tab_rels:
        show_relationships_tab(df, numeric_cols, categorical_cols)

    with tab_pca:
        show_pca_clustering_tab(df, numeric_cols, categorical_cols)

    with tab_class:
        show_classification_tab(df, numeric_cols, categorical_cols)

    # with tab_opt:
    #     show_optimization_tab(df, numeric_cols, categorical_cols)



def show_flags_tab(df, numeric_cols, categorical_cols):
    """
    Pestaña: análisis de banderas (Moderado, Fake, Rollback).
    """
    st.header("Análisis de flags (Moderado, Fake, Rollback)")

    # Filtros categóricos
    st.subheader("Filtros - Variables Categóricas")
    cat_filt = st.multiselect("Selecciona cols cat p/ filtrar", categorical_cols, [])
    chosen_cats = {}
    for ccol in cat_filt:
        uv = sorted(df[ccol].dropna().unique().tolist())
        chosen = st.multiselect(f"Filtrar {ccol}", uv, uv)
        chosen_cats[ccol] = chosen

    # Filtros numéricos
    st.subheader("Filtros - Variables Numéricas")
    num_filt = st.multiselect("Selecciona cols num p/ filtrar", numeric_cols, [])
    num_ranges = {}
    for ncol in num_filt:
        mn, mx = float(df[ncol].min()), float(df[ncol].max())
        rng = st.slider(f"{ncol}", mn, mx, (mn, mx))
        num_ranges[ncol] = rng

    # Agrupamiento
    st.subheader("Agrupamiento")
    group_cols = st.multiselect("Cols cat para agrupar", categorical_cols, ["site_id"])

    # Métricas
    st.subheader("Métricas")
    mets_avail = [
        'FreqArticlesGlobal',
        'FreqModeratedByGroup',
        'FreqModeratedGlobal',
        'FreqFakeByGroup',
        'FreqFakeGlobal',
        'FreqRollbackByGroup',
        'FreqRollbackGlobal'
    ]
    sel_mets = st.multiselect("Métricas a mostrar", mets_avail, mets_avail)

    top_n = st.number_input("Top N grupos", 1, 100, 10)

    btn = st.button("Ejecutar Análisis")

    if btn:
        df_filt = df.copy()

        # Aplicar filtros
        for col, vals in chosen_cats.items():
            df_filt = df_filt[df_filt[col].isin(vals)]
        for col, (rmin, rmax) in num_ranges.items():
            df_filt = df_filt[(df_filt[col]>=rmin) & (df_filt[col]<=rmax)]

        if len(df_filt) == 0:
            st.warning("No hay datos tras los filtros.")
            return
        if not group_cols:
            st.error("Selecciona al menos una col de agrupamiento.")
            return

        stats_df, figs_dict = analyze_flags_by_columns(
            df=df_filt,
            group_cols=group_cols,
            top_n=top_n,
            selected_metrics=sel_mets
        )
        st.write("### Resultados de flags")
        st.dataframe(stats_df)

        for m, fg in figs_dict.items():
            st.write(f"**Gráfico para la métrica: {m}**")
            st.pyplot(fg)


def show_relationships_tab(df, numeric_cols, categorical_cols):
    """
    Pestaña: boxplot de una variable num por combinación de cat_cols subdividida por otra var.
    """
    st.header("Relaciones y Correlaciones")
    st.write("Boxplot de variable cuantitativa vs cat_cols, subdividido por otra variable.")

    num_choice = st.selectbox("Variable Cuantitativa", numeric_cols, index=0)
    cat_choice = st.multiselect("Variables Categóricas p/ agrupar", categorical_cols, ["site_id"])
    subdiv_col = st.selectbox("Variable auxiliar (subdividir)", [None]+categorical_cols+["Fake","Rollback","Moderado"], index=0)

    run_box_btn = st.button("Generar Boxplot")

    if run_box_btn:
        if not cat_choice:
            st.error("Selecciona al menos una variable categórica.")
            return
        if subdiv_col is None:
            st.error("Selecciona una variable para subdividir.")
            return

        fig_box = boxplot_by_categories(df.copy(), num_choice, cat_choice, subdiv_col)
        st.pyplot(fig_box)


def show_pca_clustering_tab(df, numeric_cols, categorical_cols):
    """
    Pestaña para PCA mixto + cluster jerárquico. Sin dendrograma.
    Con estadística adicional de clusters.
    """
    st.header("PCA + Clustering (Jerárquico)")

    st.subheader("PCA Mixto")
    num_clean = [c for c in numeric_cols if c not in ['Fake','Moderado','Rollback']]
    sel_num = st.multiselect("Variables Numéricas", num_clean, ['Score','precio_usd'])
    sel_cat = st.multiselect("Variables Categóricas", categorical_cols, [])
    n_comp = st.slider("Número de Componentes", 2, 5, 2)
    color_var = st.selectbox("Colorear scatter (opcional)", [None,'Fake','Moderado','Rollback']+categorical_cols, 0)

    run_pca_btn = st.button("Ejecutar PCA")
    if "X_pca" not in st.session_state:
        st.session_state["X_pca"] = None
    if "df_pca_coords" not in st.session_state:
        st.session_state["df_pca_coords"] = None

    if run_pca_btn:
        if not sel_num and not sel_cat:
            st.error("Selecciona variables num o cat para el PCA.")
            return
        try:
            df_pca, exp_var, encoded_df = perform_pca_mixed(
                df=df,
                numeric_cols=sel_num,
                categorical_cols=sel_cat,
                n_components=n_comp
            )
            st.session_state["X_pca"] = df_pca.values  # matriz para cluster
            st.session_state["df_pca_coords"] = df_pca  # coords PC1..PCn

            st.write("#### Varianza Explicada por Componente")
            for i, v in enumerate(exp_var, start=1):
                st.write(f"PC{i}: {v:.2%}")

            if n_comp >= 2:
                fig_bi = plot_pca_biplot(df, df_pca, encoded_df, color_var)
                st.pyplot(fig_bi)

        except ValueError as e:
            st.error(f"Error PCA: {e}")

    st.subheader("Cluster Jerárquico (Ward)")

    k_clust = st.number_input("Número de clusters (k)", 2, 20, 3)
    btn_cluster = st.button("Asignar Clusters")

    if btn_cluster:
        if st.session_state["X_pca"] is None:
            st.warning("Ejecuta primero PCA.")
        else:
            from src.cluster_analysis import perform_hierarchical_clustering, compute_cluster_insights
            X = st.session_state["X_pca"]
            labs = perform_hierarchical_clustering(X, n_clusters=k_clust, method="ward")
            df_pca_coords = st.session_state["df_pca_coords"].copy()
            df_pca_coords["cluster"] = labs

            st.write("### Clusters generados")
            st.dataframe(df_pca_coords.head(20))

            fig_clust = plot_pca_clusters(df_pca_coords)
            st.pyplot(fig_clust)

            # Calcular estadísticas de cada cluster
            cluster_stats = compute_cluster_insights(df, df_pca_coords, cluster_label="cluster")
            st.write("### Estadísticas de cada Cluster")
            st.dataframe(cluster_stats)


def show_classification_tab(df, numeric_cols, categorical_cols):
    """
    Pestaña para el mini experimento de clasificación (Fake).
    Con opción de "logistic", "rf" o "xgboost" y se muestra
    la importancia agregada por variable.
    """
    st.header("Mini Experimento de Clasificación para detectar Fakes")

    st.write("""
    Selecciona variables numéricas y categóricas que creas relevantes
    para predecir si un registro es 'Fake'.
    Modelos disponibles: Logistic, Random Forest, XGBoost.
    Se mostrará la exactitud y la importancia agregada por variable.
    """)

    # Escoger variables
    sel_num = st.multiselect("Variables numéricas", numeric_cols, ['Score','precio_usd'])
    sel_cat = st.multiselect("Variables categóricas", categorical_cols, ['site_id','Dominio_normalizado'])

    # Escoger modelo
    model_type = st.selectbox("Modelo", ["logistic", "rf", "xgboost"], index=0)
    test_size = st.slider("Porcentaje test", 0.1, 0.5, 0.3, 0.05)

    btn_classify = st.button("Entrenar Modelo")

    if btn_classify:
        if len(sel_num)==0 and len(sel_cat)==0:
            st.error("Debes seleccionar al menos una variable numérica o categórica.")
            return

        from src.classification import run_fake_classification
        try:
            acc, df_import, df_import_agg = run_fake_classification(
                df=df,
                numeric_cols=sel_num,
                categorical_cols=sel_cat,
                model_type=model_type,
                test_size=test_size
            )

            st.write(f"### Exactitud (test): {acc:.2%}")

            # Mostrar solo la importancia agregada (por variable original)
            st.write("### Importancia Agregada (por Variable Original)")
            st.dataframe(df_import_agg)

            # Opcionalmente, podríamos mostrar la tabla detallada
            with st.expander("Ver detalle de cada subcolumna"):
                st.dataframe(df_import)

        except ValueError as e:
            st.error(f"Error al entrenar modelo: {e}")

def show_optimization_tab(df, numeric_cols, categorical_cols):
    """
    Pestaña de Optimización donde:
      1) Se eligen variables categóricas y numéricas.
      2) Se elige un porcentaje mínimo de cobertura (min_coverage).
      3) Se generan TODAS las subsets (powerset) de categorías, y los umbrales
         para las numéricas.
      4) Se cruzan y se filtra la base, se calcula coverage y pFake.
      5) Se determina la frontera eficiente y se grafican los resultados.
    """
    from src.optimizer import (
        build_cat_subsets,
        find_numeric_cutpoints,
        generate_combinations_and_evaluate,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("Optimización de Filtros con Múltiples Categorías")

    st.write("""
    Selecciona variables categóricas y numéricas, junto con un porcentaje mínimo
    de la base que se desee conservar. Se generarán todas las combinaciones de subsets
    de categorías y puntos de corte numéricos , se calculará la cobertura y el porcentaje de fakes resultante.
    """)

    selected_cat = st.multiselect("Variables Categóricas", categorical_cols, ["site_id"])
    selected_num = st.multiselect("Variables Numéricas", numeric_cols, ["precio_usd"])

    min_cov = st.slider("Porcentaje Mínimo de Cobertura", 0.0, 1.0, 0.5, 0.05)

    run_opt_btn = st.button("Ejecutar Optimización")

    if run_opt_btn:
        if not selected_cat and not selected_num:
            st.error("Selecciona al menos una variable categórica o numérica.")
            return

        # 1) Generar subsets para las variables cat
        cat_subsets_dict = build_cat_subsets(df, selected_cat)

        # 2) Determinar puntos de corte para variables num
        #    (árbol con max_leaf_nodes=3, se pueden cambiar)
        cutpoints_dict = find_numeric_cutpoints(df, selected_num, max_leaf_nodes=3)

        # 3) Generar combos y evaluar
        df_res = generate_combinations_and_evaluate(df, cat_subsets_dict, cutpoints_dict, min_cov)
        if df_res.empty:
            st.warning("No hay combinaciones que cumplan con la cobertura mínima.")
            return

        st.write(f"### Resultados de {len(df_res)} combinaciones")
        st.dataframe(df_res.sort_values("Coverage", ascending=False))

        # 4) Graficar scatter coverage vs pFake
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=df_res, x="Coverage", y="pFake", alpha=0.4, color="gray", ax=ax)

        ax.set_xlabel("Cobertura (%)")
        ax.set_ylabel("% Fake")
        ax.set_title("Cobertura vs. %Fake")
        st.pyplot(fig)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()