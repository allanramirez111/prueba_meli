# src/cluster_analysis.py

"""
Módulo para la segmentación jerárquica y obtención de estadísticas de clusters.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import AgglomerativeClustering


def perform_hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    method: str = "ward"
) -> np.ndarray:
    """
    Aplica cluster jerárquico sin dendrograma,
    retornando las etiquetas de cluster.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(X)
    return labels


def compute_cluster_insights(
    df_original: pd.DataFrame,
    df_pca_coords: pd.DataFrame,
    cluster_label: str = "cluster"
) -> pd.DataFrame:
    """
    Dado el DataFrame original (con columns Fake, Moderado, site_id, etc.),
    y un df_pca_coords que tenga la columna 'cluster', calcula:

    1. Cantidad de registros en cada cluster y su frecuencia relativa.
    2. Porcentaje de registros fake en cada cluster.
    3. Porcentaje de esos fake que fueron moderados en cada cluster.
    4. Proporción de países (site_id) dentro de cada cluster (se concatenan en un solo string).

    Args:
        df_original (pd.DataFrame): DataFrame con columns 'Fake','Moderado','site_id', etc.
        df_pca_coords (pd.DataFrame): DataFrame con el índice correspondiente y col 'cluster'.
        cluster_label (str): Nombre de la columna de cluster.

    Returns:
        pd.DataFrame con filas = cluster y columnas con las métricas solicitadas.
    """
    # Unir (join) ambos DF en base al índice para disponer de las banderas
    # y site_id en la misma tabla donde están los clusters
    df_merged = df_original.join(df_pca_coords[[cluster_label]], how="inner")

    # Grupo por cluster
    grouped = df_merged.groupby(cluster_label)

    total_rows = len(df_merged)

    # Contador del cluster
    cluster_stats = pd.DataFrame()
    cluster_stats["Count"] = grouped.size()
    cluster_stats["FreqRel"] = cluster_stats["Count"]/total_rows

    # Porcentaje de registros fake en cada cluster
    cluster_stats["%Fake"] = grouped["Fake"].mean()*100  # mean *100 = porcentaje

    # Porcentaje de fakes que fueron moderados
    # (Fake=1 & Moderado=1) / (Fake=1) en cada cluster
    # Evita /0 si no hay fakes
    def pct_fake_moderated(group):
        fake_sum = group["Fake"].sum()
        if fake_sum == 0:
            return 0.0
        # cuántos de esos fakes tienen Moderado=1
        fm = group[(group["Fake"]==1) & (group["Moderado"]==1)].shape[0]
        return (fm/fake_sum)*100

    cluster_stats["%FakeModerated"] = grouped.apply(pct_fake_moderated)

    # Proporción de países en cada cluster (site_id)
    # Por simplicidad, generamos un string "PaisA:30%, PaisB:20%, etc."
    def site_distribution(group):
        if "site_id" not in group.columns:
            return ""
        total_c = len(group)
        dist = group["site_id"].value_counts().head(5)  # top 5
        # crear un string con form "AR:30%, BR:20%..."
        dist_str = []
        for idx, val in dist.items():
            dist_str.append(f"{idx}:{(val/total_c)*100:.1f}%")
        return ", ".join(dist_str)

    cluster_stats["siteDistribution"] = grouped.apply(site_distribution)

    return cluster_stats