# src/pca_analysis.py

"""
Módulo para realizar PCA con variables numéricas y categóricas.
Usa One-Hot Encoding para variables categóricas, escalado con StandardScaler,
y retorna un DataFrame con PC1, PC2, ..., la varianza explicada y
el DF codificado (encoded_df).
"""

import pandas as pd
from typing import Tuple, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca_mixed(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_components: int = 2,
    scale_data: bool = True
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """
    Realiza PCA sobre variables numéricas y categóricas (One-Hot).
    No imprime DataFrames; retorna la matriz PCA y la varianza explicada.

    Args:
        df (pd.DataFrame): DataFrame original.
        numeric_cols (List[str]): Columnas numéricas a incluir (ej. ['Score', 'precio_usd']).
        categorical_cols (List[str]): Columnas categóricas a codificar.
        n_components (int): Número de componentes principales a extraer.
        scale_data (bool): Si True, aplica estandarización.

    Returns:
        (df_pca, explained_variance, encoded_df):
          - df_pca (pd.DataFrame): PC1, PC2, ... con índice del DF original (tras dropear NAs).
          - explained_variance (List[float]): porcentaje de varianza explicada por cada PC.
          - encoded_df (pd.DataFrame): DataFrame con variables numéricas + dummies (opcional).
    """
    # 1) Filtrar y dropear NAs
    df_numeric = df[numeric_cols].dropna()
    df_categorical = df[categorical_cols].dropna()

    # Intersección de índices
    common_idx = df_numeric.index.intersection(df_categorical.index)
    df_numeric = df_numeric.loc[common_idx]
    df_categorical = df_categorical.loc[common_idx]

    # 2) One-Hot Encoding
    df_cat_encoded = pd.get_dummies(df_categorical, drop_first=False)

    # 3) Unir numéricas y categóricas
    encoded_df = pd.concat([df_numeric, df_cat_encoded], axis=1)

    # 4) Estandarizar
    X = encoded_df.values
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(encoded_df)

    # 5) PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.tolist()

    # 6) Construir df_pca
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(data=pca_data, columns=pc_cols, index=encoded_df.index)

    return df_pca, explained_variance, encoded_df