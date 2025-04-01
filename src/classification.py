# src/classification.py

"""
Módulo para realizar un mini experimento de clasificación que predice 'Fake'.
Ahora incluye la opción de modelo XGBoost y
agrega la funcionalidad de agrupar importancias por variable original.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def run_fake_classification(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    model_type: str = "logistic",
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Entrena un modelo para predecir la columna 'Fake' usando
    las columnas numéricas y categóricas indicadas, con pipeline
    que incluye OneHotEncoder y StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame que contiene 'Fake' y las indicadas en numeric_cols, categorical_cols.
        numeric_cols (List[str]): variables numéricas a usar como features.
        categorical_cols (List[str]): variables categóricas a usar como features.
        model_type (str): "logistic", "rf" o "xgboost".
        test_size (float): proporción de datos para test (por defecto, 0.3).
        random_state (int): semilla para reproducibilidad.

    Returns:
        (accuracy, df_import, df_import_agg):
          - accuracy (float): Exactitud en el conjunto de prueba.
          - df_import (pd.DataFrame): Importancia detallada por cada subcolumna
            generada tras OHE (o coef. en logistic).
          - df_import_agg (pd.DataFrame): Importancia **agregada** por variable original.
    """
    # 1) Filtrar filas donde 'Fake' no sea nulo
    df = df.dropna(subset=["Fake"])

    # 2) Preparar X (features) e y (target)
    X = df[numeric_cols + categorical_cols].copy()
    y = df["Fake"].astype(int)  # Asegurar que sea 0/1

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4) Definir preprocesamiento
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first")  # o drop=False

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # 5) Escoger modelo
    if model_type == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif model_type == "xgboost":
        model = XGBClassifier(
            random_state=random_state,
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        raise ValueError("Modelo no reconocido. Usa 'logistic', 'rf' o 'xgboost'.")

    # 6) Crear pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # 7) Entrenar
    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    # 8) Obtener nombres de features transformados
    cat_enc = pipe.named_steps["preprocessor"].transformers_[1][1]  # OneHotEncoder
    cat_feature_names = cat_enc.get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_feature_names)

    # 9) Obtener importancias o coeficientes
    if model_type == "logistic":
        coefs = pipe.named_steps["classifier"].coef_[0]
        df_import = pd.DataFrame({
            "feature": feature_names,
            "importance": coefs
        }).sort_values("importance", ascending=False)
    else:
        importances = pipe.named_steps["classifier"].feature_importances_
        df_import = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

    # 10) Agregar importancias por variable original
    df_import_agg = _aggregate_importances_by_variable(
        df_import, numeric_cols, categorical_cols
    )

    return accuracy, df_import, df_import_agg


def _aggregate_importances_by_variable(
    df_import: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str]
) -> pd.DataFrame:
    """
    Agrupa la importancia de cada subcolumna OHE en su variable original.
    Para variables numéricas se deja tal cual.

    Se suma la importancia en valor absoluto (para no cancelar positivos/negativos).
    Retorna un DataFrame con 'variable' y 'importance'.
    """
    # Diccionario para acumular importancias
    agg_dict = {}

    for row in df_import.itertuples():
        feat = row.feature
        imp = row.importance

        # Tomamos el valor absoluto de la importancia
        val = abs(imp)

        # 1) Si es una columna numérica "pura" -> feat en numeric_cols
        if feat in numeric_cols:
            agg_dict[feat] = agg_dict.get(feat, 0) + val
        else:
            # 2) De lo contrario, es un subcol. Buscamos a qué cat col corresponde
            #    Ejemplo: "site_id_ARGENTINA" => variable original "site_id"
            matched_cat = None
            for cat_col in cat_cols:
                prefix = cat_col + "_"  # si drop_first=False
                if feat.startswith(prefix):
                    matched_cat = cat_col
                    break

            # Si no se encontró, podría ser un subcol sin underscore => fallback
            if matched_cat is None:
                # Comprobar si el feat = cat_col por si 'drop="first"' no generó sub
                # o no hay underscore. Puede pasar con logistic si drop_first=False
                # como 'Fake_1', etc.
                for cat_col in cat_cols:
                    if feat == cat_col:
                        matched_cat = cat_col
                        break

            # Agregamos
            if matched_cat is not None:
                agg_dict[matched_cat] = agg_dict.get(matched_cat, 0) + val
            else:
                # Si nada matchea, lo consideramos como feat original
                agg_dict[feat] = agg_dict.get(feat, 0) + val

    # Pasar a DataFrame
    df_agg = pd.DataFrame(list(agg_dict.items()), columns=["variable", "importance"])
    df_agg = df_agg.sort_values("importance", ascending=False).reset_index(drop=True)

    return df_agg