# src/utils/json_saver.py

import os
import json
import numpy as np
import pandas as pd


def save_json(data, filename: str, proc_dir: str) -> str:
    """
    Objetivo:
        Serializar y guardar resultados del pipeline en formato JSON.
        Convierte tipos no serializables (numpy, pandas, frozenset) automáticamente.

    Parámetros:
        data: Resultados a guardar (dict, DataFrame, list).
        filename (str): Nombre del archivo (ej. "apriori.json").
        proc_dir (str): Ruta a data/processed/.

    Retorna:
        str: Ruta completa del archivo guardado.
    """
    os.makedirs(proc_dir, exist_ok=True)
    path = os.path.join(proc_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_serializer)

    print(f"[JSON] Guardado: {path}")
    return path


def _serializer(obj):
    """
    Objetivo:
        Convertir tipos no serializables por JSON estándar.

    Parámetros:
        obj: Objeto a serializar.

    Retorna:
        Versión serializable del objeto.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, frozenset):
        return sorted(list(obj))
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Tipo no serializable: {type(obj)}")