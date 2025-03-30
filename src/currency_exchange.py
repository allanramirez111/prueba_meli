import numpy as np
import pandas as pd

def create_usd_price_column(
    df: pd.DataFrame,
    exchange_rates: dict = {
        "BRASIL": 5.9,
        "MEXICO": 20.5,
        "ARGENTINA": 1100.00,
        "COLOMBIA": 4200.00,
        "PERU": 3.70
    }) -> pd.DataFrame:
    """
    Crea la columna 'precio_usd' a partir de la columna 'Precio' y el país (site_id).
    Se basa en un diccionario de tasas de cambio aproximadas.
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Precio' y 'site_id'.
        exchange_rates (dict): Diccionario con las tasas de cambio estimadas.
    
    Returns:
        pd.DataFrame: DataFrame con la columna adicional 'precio_usd'.
    """
    
    # 1) Limpieza y conversión de la columna 'Precio' a float uniforme
    df['Precio'] = df['Precio']
    
    # 2) Función para convertir el precio local a USD
    def convert_price_to_usd(row):
        site = row['site_id']
        price_local = row['Precio']
        
        if pd.isna(price_local):
            return np.nan
        
        # Obtenemos la tasa de cambio según el país; si no existe, asumimos 1
        rate = exchange_rates.get(site, 1)
        return price_local / rate
    
    # 3) Crear la nueva columna 'precio_usd'
    df['precio_usd'] = df.apply(convert_price_to_usd, axis=1)
    
    return df