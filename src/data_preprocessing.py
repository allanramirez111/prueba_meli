# src/data_preprocessing.py

"""
Módulo de preprocesamiento de datos.
Contiene funciones para la carga, transformación y limpieza del dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from config.settings import DATA_PATH


def transform_data(
    path: str = DATA_PATH,
    exchange_rates: Dict[str, float] = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Lee el archivo CSV desde 'path', realiza la transformación de precios
    a USD (columna 'precio_usd'), identifica columnas numéricas y categóricas,
    y devuelve el DataFrame procesado junto con las listas de columnas numéricas
    y categóricas.

    Args:
        path (str): Ruta al archivo CSV (por defecto DATA_PATH).
        exchange_rates (Dict[str, float]): Diccionario con tasas de cambio
            aproximadas a USD para cada país (site_id). Si None, usa valores
            por defecto.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]:
            - DataFrame procesado con 'precio_usd'
            - Lista de columnas numéricas
            - Lista de columnas categóricas
    """
    if exchange_rates is None:
        exchange_rates = {
            "BRASIL": 5.9,
            "MEXICO": 20.5,
            "ARGENTINA": 1100.0,
            "COLOMBIA": 4200.0,
            "PERU": 3.70,
            "CHILE": 950.0
        }
    
    # 1) Carga el DataFrame desde CSV
    df = pd.read_csv(path)

    # 2) Convertir precio a USD:
    df = _create_usd_price_column(df, exchange_rates)

    # 3) Identificar columnas numéricas y categóricas.
    #    - Asumimos 'Precio', 'precio_usd' y 'Score' pueden ser numéricas,
    #      aunque 'Precio' ya se usa, es posible que quieras otras limpiezas.
    # numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    # Excluir ID o contadores si no te interesan como numéricos
    numeric_cols = ["Precio", "Score", "precio_usd"]
    
    categorical_cols = [a for a in df.columns if a not in numeric_cols]
    
    # 4) Aquí podrías añadir más transformaciones o limpiezas, por ejemplo:
    #    - Rellenar NAs
    #    - Corregir tipos de datos
    #    - Normalizaciones, etc.

    return df, numeric_cols, categorical_cols


def _create_usd_price_column(
    df: pd.DataFrame,
    exchange_rates: Dict[str, float]
) -> pd.DataFrame:
    """
    Crea la columna 'precio_usd' a partir de la columna 'Precio' y el país (site_id).
    Se basa en un diccionario de tasas de cambio aproximadas.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Precio' y 'site_id'.
        exchange_rates (dict): Diccionario con las tasas de cambio estimadas.

    Returns:
        pd.DataFrame: DataFrame con la columna adicional 'precio_usd'.
    """
    
    def convert_price_to_usd(row):
        site = row.get('site_id', None)
        price_local = row.get('Precio', np.nan)
        
        if pd.isna(price_local):
            return np.nan
        
        rate = exchange_rates.get(site, 1)
        return price_local / rate

    df['precio_usd'] = df.apply(convert_price_to_usd, axis=1)
    return df