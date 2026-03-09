# FUNCIÓN: LIMPIEZA Y PREPROCESAMIENTO DE DATOS

def clean_data(df, config):
    """
    Aplica limpieza básica a los datos:
    - Convierte tipos de datos.
    - Elimina filas con valores nulos en columnas esenciales.
    - Elimina duplicados (misma transacción, mismo ítem).
    - Aplica filtros adicionales si están definidos.

    Parámetros:
        df (DataFrame): Datos originales.
        config (dict): Configuración con nombres de columnas y filtros.

    Retorna:
        DataFrame limpio.
    """
    # Crear una copia para no modificar el DataFrame original
    df_clean = df.copy()

    # ========== CONVERSIÓN DE TIPOS ==========

    # Convertir id_transaccion a string para evitar problemas con números mezclados con texto
    # También se asegura de que no haya valores nulos aquí (se manejarán después)
    df_clean[config['transaction_id']] = df_clean[config['transaction_id']].astype(str)

    # Convertir nombre del ítem a string y eliminar espacios en blanco al inicio y final
    df_clean[config['item_name']] = df_clean[config['item_name']].astype(str).str.strip()

    # ========== ELIMINACIÓN DE VALORES NULOS ==========

    # Registrar el número de filas antes de eliminar nulos
    before_drop = len(df_clean)

    # Eliminar filas donde id_transaccion o item_name sean nulos (NaN) o vacíos después de limpiar
    # Nota: después de convertir a string, los NaN se convierten en 'nan', por lo que también filtramos eso
    df_clean = df_clean[
        (df_clean[config['transaction_id']] != 'nan') &
        (df_clean[config['item_name']] != 'nan') &
        (df_clean[config['transaction_id']].str.strip() != '') &
        (df_clean[config['item_name']].str.strip() != '')
    ]

    after_drop = len(df_clean)
    print(f"Filas eliminadas por valores nulos o vacíos en columnas esenciales: {before_drop - after_drop}")

    # ========== ELIMINACIÓN DE DUPLICADOS ==========

    # Eliminar filas duplicadas exactas (misma transacción, mismo ítem)
    # Esto evita contar dos veces el mismo producto en una misma transacción (aunque en algunos casos podría ser válido, lo común es mantener uno)
    before_dup = len(df_clean)
    df_clean.drop_duplicates(subset=[config['transaction_id'], config['item_name']], inplace=True)
    after_dup = len(df_clean)
    print(f"Filas duplicadas eliminadas: {before_dup - after_dup}")

    # Mostrar resumen final
    print(f"Datos después de limpieza: {len(df_clean)} filas.")
    return df_clean