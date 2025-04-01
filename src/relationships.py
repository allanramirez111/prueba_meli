# src/relationships.py

"""
Módulo para análisis de "Relaciones y Correlaciones".
Aquí, agregamos la función para generar un boxplot:
- variable cuantitativa
- agrupación por combinación de cat_cols
- subdividido por variable auxiliar (ej. Fake, No Fake)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def boxplot_by_categories(
    df: pd.DataFrame,
    numeric_col: str,
    cat_cols: List[str],
    subdiv_col: str
) -> plt.Figure:
    """
    Genera un boxplot de 'numeric_col', agrupado por la combinación
    de 'cat_cols', subdividido por 'subdiv_col'.

    Ej: boxplot de 'precio_usd' por 'Dominio_normalizado',
    subdividido por 'Fake'.

    Args:
        df (pd.DataFrame): DataFrame con las columnas necesarias.
        numeric_col (str): Nombre de la variable cuantitativa.
        cat_cols (List[str]): Lista de columnas categóricas para combinar en una sola.
        subdiv_col (str): Columna para subdividir (hue).

    Returns:
        plt.Figure: Figura de matplotlib.
    """
    # 1) Creamos una columna que combine las cat_cols
    #    si hay más de 1, las concatenamos en un string
    if len(cat_cols) == 1:
        df['_ComboCat'] = df[cat_cols[0]].astype(str)
    else:
        df['_ComboCat'] = df[cat_cols].astype(str).agg('_'.join, axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x='_ComboCat', y=numeric_col,
        hue=subdiv_col,
        ax=ax
    )
    ax.set_title(f"Boxplot de {numeric_col} por {cat_cols} subdividido por {subdiv_col}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Eliminamos la columna temporal
    df.drop(columns=['_ComboCat'], inplace=True)
    return fig