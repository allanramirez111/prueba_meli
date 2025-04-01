# src/classification.py

"""
Módulo para realizar un mini experimento de clasificación que predice 'Fake'.
Ahora incluye las opciones:
- Logistic Regression
- Random Forest
- XGBoost
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
from xgboost import XGBClassifier  # <-- Importamos XGBoost


def run_fake_classification(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    model_type: str = "logistic",
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[float, pd.DataFrame]:
    """
    Entrena un modelo para predecir la columna 'Fake' usando
    las columnas numéricas y categóricas indicadas, con pipeline
    que incluye OneHotEncoder y StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame que contiene 'Fake' y las columnas
            numéricas/categóricas a usar como features.
        numeric_cols (List[str]): Lista de nombres de columnas numéricas.
        categorical_cols (List[str]): Lista de nombres de columnas categóricas.
        model_type (str): "logistic", "rf" o "xgboost".
        test_size (float): Proporción de datos para test (por defecto, 0.3).
        random_state (int): Semilla para reproducibilidad.

    Returns:
        (accuracy, feature_importances_df):
          - accuracy (float): Exactitud en el conjunto de prueba.
          - feature_importances_df (pd.DataFrame): Importancia de cada feature
            (coeficientes si es 'logistic', o feature_importances_ si es 'rf' o 'xgboost').
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

    # 4) Definir preprocesamiento:
    #    - Numeric: StandardScaler
    #    - Categorical: OneHotEncoder
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first")  # drop="first" p/ evitar colinealidad

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # 5) Escoger modelo según model_type
    if model_type == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif model_type == "xgboost":
        # XGBoost. Ajusta hyperparámetros según tus necesidades
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

    # 9) Obtener importancias
    if model_type == "logistic":
        coefs = pipe.named_steps["classifier"].coef_[0]
        df_import = pd.DataFrame({
            "feature": feature_names,
            "importance": coefs
        }).sort_values("importance", ascending=False)
    else:
        # Para 'rf' o 'xgboost', usamos feature_importances_
        importances = pipe.named_steps["classifier"].feature_importances_
        df_import = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

    return accuracy, df_import