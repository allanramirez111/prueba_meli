# src/visualizations.py

"""
Funciones de visualización: PCA biplot, scatter clusters, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def plot_pca_biplot(df, df_pca, encoded_df, color_var=None):
    """
    Biplot con PC1 vs PC2 y vectores de las variables.
    (Igual a tu versión previa)
    """
    if 'PC1' not in df_pca.columns or 'PC2' not in df_pca.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hay PC1 o PC2", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot = df_pca.copy()
    if color_var and color_var in df.columns:
        df_plot[color_var] = df[color_var]
        sns.scatterplot(
            data=df_plot, x='PC1', y='PC2',
            hue=color_var, alpha=0.7, s=30, ax=ax
        )
        ax.legend(title=color_var)
    else:
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', alpha=0.7, s=30, ax=ax)

    ax.set_title("Biplot (PC1 vs PC2)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Reconstruir PCA local para obtener vectores (loadings)
    X = encoded_df.values
    X_scaled = StandardScaler().fit_transform(X)
    pca_local = PCA(n_components=2)
    pca_local.fit(X_scaled)
    loadings = pca_local.components_.T
    feature_names = encoded_df.columns

    arrow_size = 2.0
    for i, feat in enumerate(feature_names):
        x_vec = loadings[i, 0]*arrow_size
        y_vec = loadings[i, 1]*arrow_size
        ax.arrow(0, 0, x_vec, y_vec,
                 color='red', alpha=0.5,
                 head_width=0.02, head_length=0.03,
                 length_includes_head=True)
        ax.text(x_vec*1.15, y_vec*1.15, feat, color='black',
                ha='center', va='center', fontsize=8)
    return fig


def plot_pca_clusters(df_clusters):
    """
    Scatter PC1 vs PC2 coloreado por 'cluster'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_clusters,
        x='PC1', y='PC2',
        hue='cluster',
        palette='tab10',
        s=30,
        ax=ax
    )
    ax.set_title("Clusters en plano PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title='cluster')
    return fig