# FUNCIÓN: CARGAR DATOS DESDE DIFERENTES FORMATOS


def load_data(config):
    """
    Carga datos desde un archivo según la configuración.
    Soporta CSV, Excel y JSON. Si el formato no es soportado, lanza una excepción.

    Parámetros:
        config (dict): Diccionario de configuración con las claves necesarias.

    Retorna:
        DataFrame de pandas con los datos cargados.
    """
    # Extraer parámetros de configuración
    file_path = config['file_path']
    file_type = config['file_type'].lower()  # Convertir a minúsculas para comparar

    # Verificar que el archivo existe antes de intentar cargarlo
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    # Intentar cargar según el tipo de archivo
    try:
        if file_type == 'csv':
            # Cargar CSV con opciones de configuración
            df = pd.read_csv(
                file_path,
                encoding=config.get('csv_encoding', 'utf-8'),  # Usa 'utf-8' si no se especifica
                sep=config.get('csv_sep', ',')                 # Usa ',' si no se especifica
            )
            print(f"Archivo CSV cargado correctamente: {file_path}")

        elif file_type == 'excel':
            # Cargar Excel (requiere openpyxl o xlrd)
            df = pd.read_excel(file_path)
            print(f"Archivo Excel cargado correctamente: {file_path}")

        elif file_type == 'json':
            # Cargar JSON
            df = pd.read_json(file_path)
            print(f"Archivo JSON cargado correctamente: {file_path}")

        else:
            # Si el tipo no está soportado, lanzar error
            raise ValueError(f"Tipo de archivo no soportado: {file_type}. Use 'csv', 'excel' o 'json'.")

        # Mostrar información básica del DataFrame
        print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas.")
        print(f"Columnas encontradas: {list(df.columns)}")
        return df

    except Exception as e:
        # Capturar cualquier error durante la carga y relanzarlo con mensaje claro
        raise Exception(f"Error al cargar el archivo {file_path}: {e}")