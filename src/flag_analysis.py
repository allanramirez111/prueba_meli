"""
Este módulo proporciona funciones para analizar las banderas de moderación
(Moderado, Fake, Rollback) agrupando por una o varias columnas categóricas
en un DataFrame de pandas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Tuple, Dict


def _validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Verifica que las columnas 'required_cols' existan en el DataFrame 'df'.
    En caso contrario, lanza un ValueError.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Las siguientes columnas no están presentes en el DataFrame: {missing}"
        )


def _flatten_group_index(df_grouped: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Toma un DataFrame con índice de varios niveles (MultiIndex), producto de
    agrupar por múltiples columnas, y crea una columna 'Grouping' que combine
    los valores de esas columnas en un solo string. Si solo hay una, la renombra.
    """
    if len(group_cols) == 1:
        df_grouped = df_grouped.reset_index()
        df_grouped.rename(columns={group_cols[0]: 'Grouping'}, inplace=True)
        return df_grouped

    df_grouped = df_grouped.reset_index()
    # Combina los valores de las columnas de agrupamiento
    df_grouped['Grouping'] = df_grouped[group_cols].astype(str).agg(', '.join, axis=1)
    df_grouped.drop(columns=group_cols, inplace=True)
    return df_grouped


def _analyze_flags_return_figs(
    df: pd.DataFrame,
    group_cols: List[str],
    top_n: int,
    metrics_to_plot: List[str]
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Realiza el cálculo de métricas y genera un diccionario de figuras
    para cada métrica solicitada. Retorna:
      - DataFrame con las columnas:
          [TotalArticles, Moderated, Fake, RollbackMod, FreqArticlesGlobal,
           FreqModeratedByGroup, FreqModeratedGlobal, FreqFakeByGroup,
           FreqFakeGlobal, FreqRollbackByGroup, FreqRollbackGlobal, Grouping]
      - Diccionario {nombre_metrica: figure matplotlib} para cada métrica en metrics_to_plot.
    
    Esta función no muestra nada por pantalla. Se usa internamente
    en analyze_flags_by_columns.
    """
    # Validar columnas base
    required_cols = ['Moderado', 'Fake', 'Rollback']
    _validate_columns(df, required_cols)
    # Validar columnas de agrupamiento
    _validate_columns(df, group_cols)
    
    total_rows = len(df)
    total_mod_global = df['Moderado'].sum()
    total_fake_global = df['Fake'].sum()
    total_rollback_mod_global = ((df['Rollback'] == 1) & (df['Moderado'] == 1)).sum()

    grouped = df.groupby(group_cols)
    group_stats = pd.DataFrame({
        'TotalArticles': grouped.size(),
        'Moderated': grouped['Moderado'].sum(),
        'Fake': grouped['Fake'].sum(),
        'RollbackMod': grouped.apply(
            lambda g: ((g['Rollback'] == 1) & (g['Moderado'] == 1)).sum()
        )
    })

    # Calcular frecuencias / proporciones
    group_stats['FreqArticlesGlobal'] = (
        group_stats['TotalArticles'] / total_rows if total_rows > 0 else 0
    )
    group_stats['FreqModeratedByGroup'] = group_stats['Moderated'] / group_stats['TotalArticles']
    group_stats['FreqModeratedGlobal'] = (
        group_stats['Moderated'] / total_mod_global if total_mod_global > 0 else 0
    )
    group_stats['FreqFakeByGroup'] = group_stats['Fake'] / group_stats['TotalArticles']
    group_stats['FreqFakeGlobal'] = (
        group_stats['Fake'] / total_fake_global if total_fake_global > 0 else 0
    )
    group_stats['FreqRollbackByGroup'] = group_stats.apply(
        lambda row: row['RollbackMod'] / row['Moderated'] if row['Moderated'] > 0 else 0,
        axis=1
    )
    group_stats['FreqRollbackGlobal'] = group_stats.apply(
        lambda row: row['RollbackMod'] / total_rollback_mod_global if total_rollback_mod_global > 0 else 0,
        axis=1
    )

    # Convertir índice MultiIndex a columna 'Grouping'
    group_stats = _flatten_group_index(group_stats, group_cols)
    
    # Ajustar top_n si es mayor al número de filas
    total_groups = len(group_stats)
    if top_n > total_groups:
        top_n = total_groups

    # Generar las figuras pedidas
    figs_dict = {}
    for metric in metrics_to_plot:
        # Ordenar desc y tomar top_n
        subset_sorted = group_stats.sort_values(by=metric, ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(x='Grouping', y=metric, data=subset_sorted, ax=ax)
        ax.set_title(f"Top {top_n} de '{metric}' agrupado por {group_cols}")
        ax.set_xlabel("Grupo (Grouping)")
        ax.set_ylabel(metric)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Guardamos la figura en el diccionario
        figs_dict[metric] = fig
    
    return group_stats, figs_dict


def analyze_flags_by_columns(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    top_n: int = 10,
    selected_metrics: Union[None, List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Función principal para agrupar el DataFrame 'df' por una o varias columnas
    categóricas ('group_cols') y calcular métricas de banderas (Moderado, Fake, Rollback).
    
    Args:
        df (pd.DataFrame): Debe incluir 'Moderado', 'Fake', 'Rollback' y las columnas de agrupamiento.
        group_cols (Union[str, List[str]]): Columna(s) categórica(s) para agrupar.
        top_n (int): Número de grupos a mostrar en las gráficas (orden desc). 
                     Si top_n > número real de grupos, se ajusta al máximo.
        selected_metrics (List[str] o None): Subconjunto de métricas a graficar.
                                             Si None, se grafican todas.
    
    Returns:
        Tuple con:
         - pd.DataFrame: tabla con las métricas calculadas y la columna 'Grouping'.
         - dict: Diccionario { nombre_de_la_métrica: figura_matplotlib }.
    """
    # Métricas disponibles en la lógica interna
    all_metrics = [
        'FreqArticlesGlobal',
        'FreqModeratedByGroup',
        'FreqModeratedGlobal',
        'FreqFakeByGroup',
        'FreqFakeGlobal',
        'FreqRollbackByGroup',
        'FreqRollbackGlobal'
    ]
    
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Si no especifican métricas, tomamos todas
    if selected_metrics is None or len(selected_metrics) == 0:
        metrics_to_plot = all_metrics
    else:
        # Validar que las métricas solicitadas existan
        for m in selected_metrics:
            if m not in all_metrics:
                raise ValueError(f"La métrica '{m}' no está disponible.")
        metrics_to_plot = selected_metrics

    return _analyze_flags_return_figs(
        df=df,
        group_cols=group_cols,
        top_n=top_n,
        metrics_to_plot=metrics_to_plot
    )
