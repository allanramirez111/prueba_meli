# src/optimizer.py

"""
Módulo para la optimización de filtros, permitiendo múltiples categorías
simultáneamente para las variables categóricas y puntos de corte para las
variables numéricas. Se busca que cada combinación de filtros mantenga al menos
un porcentaje mínimo de la base y se calcula la frontera eficiente (cobertura vs. %Fake).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from itertools import chain, combinations, product
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from tqdm import tqdm


def powerset_of_categories(cat_values: List[Any]) -> List[List[Any]]:
    """
    Genera la potencia de 'cat_values' (todas las subsets posibles),
    excluyendo la subset vacía.

    Ejemplo:
        Si cat_values = [A, B], retorna [[A], [B], [A, B]].

    Args:
        cat_values (List[Any]): Lista de valores únicos de la variable.

    Returns:
        List[List[Any]]: Lista de subsets (cada subset es una lista de valores).
    """
    ps = chain.from_iterable(combinations(cat_values, r) for r in range(1, len(cat_values) + 1))
    subsets = [list(s) for s in ps]
    return subsets


def build_cat_subsets(
    df: pd.DataFrame,
    cat_cols: List[str]
) -> Dict[str, List[List[Any]]]:
    """
    Para cada variable categórica, genera todas las subsets posibles (no vacías)
    de categorías, e incluye la opción None para indicar "no filtrar" esa variable.

    Retorna un diccionario con la forma:
        { col: [None, [cat1], [cat2], [cat1, cat2], ...], ... }

    Args:
        df (pd.DataFrame): DataFrame original.
        cat_cols (List[str]): Lista de nombres de variables categóricas.

    Returns:
        Dict[str, List[List[Any]]]: Diccionario con las subsets para cada variable.
    """
    subsets_dict = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        if not unique_vals:
            subsets_dict[col] = [None]  # Si no hay valores, solo "no filtrar"
        else:
            all_subsets = powerset_of_categories(unique_vals)
            subsets_dict[col] = [None] + all_subsets
    return subsets_dict


def find_numeric_cutpoints(
    df: pd.DataFrame,
    num_cols: List[str],
    max_leaf_nodes: int = 3
) -> Dict[str, List[float]]:
    """
    Para cada variable numérica, construye un árbol de decisión (Fake ~ var_num)
    con 'max_leaf_nodes' para extraer umbrales (thresholds).

    Se retornan los thresholds únicos ordenados para cada variable en forma de diccionario:
        { col: [threshold1, threshold2, ...], ... }

    Args:
        df (pd.DataFrame): DataFrame original.
        num_cols (List[str]): Lista de variables numéricas.
        max_leaf_nodes (int): Número máximo de nodos hoja para el árbol (por defecto, 3).

    Returns:
        Dict[str, List[float]]: Diccionario con umbrales por variable numérica.
    """
    cutpoints_dict = {}
    for col in num_cols:
        X = df[[col]].dropna()
        y = df.loc[X.index, "Fake"].fillna(0).astype(int)
        if len(X) < 2:
            cutpoints_dict[col] = []
            continue
        tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
        tree.fit(X, y)
        thresholds = []
        for node_idx in range(tree.tree_.node_count):
            feat_id = tree.tree_.feature[node_idx]
            thr = tree.tree_.threshold[node_idx]
            # -2.0 es el valor por defecto para nodos sin split
            if feat_id == 0 and thr != -2.0:
                thresholds.append(thr)
        thresholds = sorted(set(thresholds))
        cutpoints_dict[col] = thresholds
    return cutpoints_dict


def generate_combinations_and_evaluate(
    df: pd.DataFrame,
    cat_subsets_dict: Dict[str, List[List[Any]]],
    num_cutpoints_dict: Dict[str, List[float]],
    min_coverage: float
) -> pd.DataFrame:
    """
    Genera todas las combinaciones de filtros a partir de:
      - Opciones de subsets para cada variable categórica.
      - Puntos de corte para cada variable numérica, interpretados en dos formas:
          "col <= threshold" y "col > threshold".
    Para cada combinación, se aplica el filtro al DataFrame, se calcula la cobertura
    (Coverage) y el porcentaje de Fake (pFake). Se descartan aquellas combinaciones
    que no cumplen con 'min_coverage' (como fracción, ej. 0.5 para 50%).

    Args:
        df (pd.DataFrame): DataFrame original.
        cat_subsets_dict (Dict[str, List[List[Any]]]): Opciones de subsets para variables categóricas.
        num_cutpoints_dict (Dict[str, List[float]]): Puntos de corte para variables numéricas.
        min_coverage (float): Cobertura mínima requerida (fracción).

    Returns:
        pd.DataFrame: DataFrame con columnas:
            - FilterDesc: Descripción textual del filtro.
            - Coverage: Porcentaje de registros que sobreviven.
            - pFake: Porcentaje de Fake en los registros filtrados.
    """
    results = []
    total_rows = len(df)
    cat_cols = list(cat_subsets_dict.keys())
    num_cols = list(num_cutpoints_dict.keys())

    # Crear lista de opciones para cada variable categórica
    cat_filter_space = [cat_subsets_dict[col] for col in cat_cols]
    all_cat_combos = list(product(*cat_filter_space))

    # Crear lista de opciones para variables numéricas:
    # Para cada variable, las opciones son: None (no filtrar) y para cada threshold,
    # las dos opciones ("le", threshold) y ("gt", threshold).
    num_filter_space = []
    for col in num_cols:
        thr_list = num_cutpoints_dict[col]
        if not thr_list:
            num_filter_space.append([None])
        else:
            opts = [None]
            for t in thr_list:
                opts.append(("le", t))
                opts.append(("gt", t))
            num_filter_space.append(opts)
    all_num_combos = list(product(*num_filter_space))

    # Función para evaluar una combinación de filtros
    def evaluate_combo(cat_combo, num_combo) -> Any:
        """
        Aplica una combinación de filtros (categóricos y numéricos) y
        calcula la cobertura y el porcentaje de Fake. Retorna un dict con
        FilterDesc, Coverage y pFake, o None si no se cumple la cobertura mínima.
        """
        # Construir dict para filtros categóricos
        cat_filter_dict = {cat_cols[i]: cat_combo[i] for i in range(len(cat_cols))}
        # Construir lista para filtros numéricos
        num_filter_list = []
        for i, col in enumerate(num_cols):
            val = num_combo[i]
            if val is not None:
                num_filter_list.append((col, val[0], val[1]))

        # Aplicar filtros al DataFrame
        df_sub = df.copy()
        for col, subset in cat_filter_dict.items():
            if subset is not None:
                df_sub = df_sub[df_sub[col].isin(subset)]
        for (col, op, thr) in num_filter_list:
            if op == "le":
                df_sub = df_sub[df_sub[col] <= thr]
            else:
                df_sub = df_sub[df_sub[col] > thr]

        coverage = len(df_sub) / total_rows
        if coverage < min_coverage:
            return None

        pfake = 0.0 if len(df_sub) == 0 else df_sub["Fake"].mean() * 100.0

        # Crear descripción del filtro
        cat_parts = [f"{col} in {cat_filter_dict[col]}" for col in cat_cols if cat_filter_dict[col] is not None]
        num_parts = [f"{col} {'<=' if op == 'le' else '>'} {thr:.2f}" for col, op, thr in num_filter_list]
        filter_desc = " AND ".join(cat_parts + num_parts) if (cat_parts or num_parts) else "SIN FILTRO"
        return {"FilterDesc": filter_desc, "Coverage": coverage * 100, "pFake": pfake}

    # Ejecutar todas las combinaciones en paralelo utilizando joblib
    all_combos = list(product(all_cat_combos, all_num_combos))
    results_list = Parallel(n_jobs=-1, prefer="processes")(
        delayed(evaluate_combo)(cat_combo, num_combo) for cat_combo, num_combo in tqdm(all_combos, 'Analizando combinaciones')
    )

    # Filtrar resultados válidos
    valid_results = [res for res in results_list if res is not None]
    df_res = pd.DataFrame(valid_results)
    return df_res


def efficient_frontier(
    df_res: pd.DataFrame,
    coverage_col: str = "Coverage",
    fake_col: str = "pFake"
) -> pd.DataFrame:
    """
    Calcula la frontera eficiente (Pareto) a partir de un DataFrame que contiene
    'Coverage' y 'pFake' para cada combinación de filtros. Se ordena por cobertura
    descendente y porcentaje de Fake ascendente, y se retiene cada punto que
    representa un nuevo mínimo en pFake.

    Args:
        df_res (pd.DataFrame): DataFrame con columnas 'Coverage' y 'pFake'.
        coverage_col (str): Nombre de la columna de cobertura.
        fake_col (str): Nombre de la columna de porcentaje de Fake.

    Returns:
        pd.DataFrame: DataFrame con las combinaciones que forman la frontera eficiente.
    """
    df_sorted = df_res.sort_values(by=[coverage_col, fake_col], ascending=[False, True]).reset_index(drop=True)
    frontier_idx = []
    best_fake = float('inf')
    for i, row in df_sorted.iterrows():
        if row[fake_col] < best_fake:
            frontier_idx.append(i)
            best_fake = row[fake_col]
    df_front = df_sorted.loc[frontier_idx].copy()
    df_front.sort_values(by=[coverage_col, fake_col], ascending=[False, True], inplace=True)
    df_front.reset_index(drop=True, inplace=True)
    return df_front