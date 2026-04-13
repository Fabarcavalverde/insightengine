# src/preprocessing/data_loader

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import chardet
import logging

# Configuración de logging con fecha
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class data_loader:
    """
    Clase para cargar datasets desde diferentes formatos (CSV, Excel, JSON)
    con detección automática de encoding y separador.
    """

    def __init__(self):
        """Inicializa el cargador de datos."""
        self.df = None
        self.file_path = None

    def detect_encoding(self, file_path):
        """
        Detecta la codificación del archivo leyendo los primeros 10000 bytes.
        Retorna el nombre de la codificación (ej. 'utf-8', 'latin-1', etc.).
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding']

    def detect_separator(self, file_path, encoding):
        """
        Detecta el separador de columnas en un archivo CSV.
        Prueba con comas, punto y coma, tabulador y barra vertical.
        Retorna el separador encontrado o ',' por defecto.
        """
        separators = [',', ';', '\t', '|']
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
        for sep in separators:
            if sep in first_line:
                return sep
        return ','

    def load_file(self, file_path=None):
        """
        Permite al usuario seleccionar un archivo (CSV, Excel o JSON) mediante un diálogo
        o recibir una ruta directamente.
        Detecta el tipo de archivo, la codificación (para CSV) y el separador (para CSV).
        Carga los datos en un DataFrame de pandas y muestra información básica.
        Retorna el DataFrame o None si ocurre un error.
        """

        # Si no se proporciona ruta, abrir selector de archivos
        if file_path is None:
            root = tk.Tk()
            root.withdraw()

            file_path = filedialog.askopenfilename(
                title="Selecciona el archivo de datos",
                filetypes=[("Archivos soportados", "*.csv;*.xlsx;*.xls;*.json"),
                           ("Todos los archivos", "*.*")]
            )

            if not file_path:
                logger.warning("No se seleccionó ningún archivo.")
                return None

        self.file_path = file_path
        logger.info(f"Archivo seleccionado: {self.file_path}")

        # Obtener la extensión del archivo
        ext = os.path.splitext(self.file_path)[1].lower()

        try:
            # Cargar según extensión
            if ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
                logger.info("Archivo Excel cargado correctamente.")

            elif ext == '.json':
                self.df = pd.read_json(self.file_path)
                logger.info("Archivo JSON cargado correctamente.")

            elif ext == '.csv':
                encoding = self.detect_encoding(self.file_path)
                logger.info(f"Encoding detectado: {encoding}")

                sep = self.detect_separator(self.file_path, encoding)
                logger.info(f"Separador detectado: '{sep}'")

                # Lista de codificaciones a probar
                encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252']
                self.df = None

                for enc in encodings_to_try:
                    try:
                        self.df = pd.read_csv(self.file_path, sep=sep, encoding=enc)
                        logger.info(f"Archivo CSV cargado con encoding: {enc}")
                        break
                    except Exception:
                        continue

                if self.df is None:
                    raise ValueError("No se pudo leer el archivo con los encodings probados.")

            else:
                logger.error(f"Tipo de archivo no soportado: {ext}")
                return None

            # Validaciones del dataset

            # Dataset vacío
            if self.df.empty:
                raise ValueError("El dataset está vacío")

            # Dataset sin columnas
            if self.df.shape[1] == 0:
                raise ValueError("El dataset no tiene columnas")

            # Columnas duplicadas
            if self.df.columns.duplicated().any():
                logger.warning("Hay columnas duplicadas")

            # Dataset muy grande
            if self.df.shape[0] > 1_000_000:
                logger.warning("Dataset muy grande, el EDA puede tardar")

            # Mostrar información del dataset
            logger.info(f"Dimensiones: {self.df.shape[0]} filas, {self.df.shape[1]} columnas.")
            logger.info("Primeras 5 filas:")
            print(self.df.head())

            return self.df

        except Exception as e:
            logger.error(f"Error al cargar el archivo: {e}")
            return None

    def get_dataframe(self):
        """Retorna el DataFrame cargado (None si no se ha cargado nada)."""
        return self.df


if __name__ == "__main__":
    # Prueba rápida de la clase
    loader = data_loader()

    # Uso con ruta directa
    df = loader.load_file(r"C:\Users\ferna\Downloads\dataset_riesgo_crediticio.csv")
    #se coloca la ruta del archivo para que sea formato pipeline
    #usar r antes de la ruta para que ignore los "\"

    # Uso con selector de archivos
    #df = loader.load_file()

    if df is not None:
        logger.info("Carga exitosa.")