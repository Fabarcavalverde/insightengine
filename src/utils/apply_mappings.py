import pandas as pd


def apply_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Objetivo:
        Reemplazar códigos crípticos por etiquetas legibles según config.yaml.

    Parámetros:
        df (pd.DataFrame): Dataset limpio con códigos originales.
        mappings (dict): Sección column_mappings del config.yaml.

    Retorna:
        pd.DataFrame: Dataset con valores reemplazados.
    """
    df = df.copy()

    for col, mapping in mappings.items():
        if col not in df.columns:
            continue

        mapping_str = {str(k): str(v) for k, v in mapping.items()}

        # convertir a object explícitamente para compatibilidad con pandas 3
        df[col] = df[col].astype(object)
        df[col] = df[col].map(lambda x: mapping_str.get(str(x), str(x)))

        print(f"[MAPPING] {col}: {df[col].unique()[:3]}")

    return df