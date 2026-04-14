# src/ingestion/data_loader.py

import os
import pandas as pd


class data_loader:
    """
    Objetivo:
        Cargar cualquier dataset CSV/data desde data/raw/.
        Lee la configuración desde config.yaml para adaptarse a cada archivo.

    Uso:
        loader = DataLoader(raw_dir)
        df = loader.load(cfg["data"])
    """

    def __init__(self, raw_dir: str):
        """
        Objetivo:
            Inicializar con la ruta al directorio raw.

        Parámetros:
            raw_dir (str): Ruta absoluta a data/raw/.

        Retorna:
            None
        """
        self.raw_dir = raw_dir

    def load(self, dcfg: dict) -> pd.DataFrame:
        """
        Objetivo:
            Leer el archivo indicado en config.yaml y asignar columnas si aplica.

        Parámetros:
            dcfg (dict): Sección "data" del config.yaml. Claves esperadas:
                         filename, delimiter, decimal, header, columnas (opcional).

        Retorna:
            pd.DataFrame: Dataset crudo.

        Raises:
            FileNotFoundError: Si el archivo no existe en raw_dir.

        Output format:
            pd.DataFrame con las columnas del dataset o nombres asignados desde config.
        """
        path = os.path.join(self.raw_dir, dcfg["filename"])

        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo: {path}")

        # header: null en yaml → None en Python → sin encabezado
        header = dcfg.get("header", "infer")
        if header == "null" or header is None:
            header = None

        df = pd.read_csv(
            path,
            sep=dcfg.get("delimiter", ","),
            decimal=dcfg.get("decimal", "."),
            header=header
        )

        # asignar nombres de columnas si se definen en config
        columnas = dcfg.get("columnas")
        if columnas and header is None:
            df.columns = columnas

        return df