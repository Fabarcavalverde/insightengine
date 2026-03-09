# CONFIGURACIÓN INICIAL DEL PIPELINE DE LIMPIEZA
from pathlib import Path  # Para manejo de rutas de forma más limpia

# Definir parámetros de configuración en un diccionario
# (Esto para modificar el comportamiento del pipeline sin tocar el código)
CONFIG = {
    # Ruta del archivo de entrada
    'file_path': r'C:\Repositorios\insightengine\data\test\MuestraCredito5000V2.csv',   # Cambia según el dataset

    # Tipo de archivo: 'csv', 'excel', 'json' (por ahora)
    'file_type': 'csv',

    # Configuración específica para CSV (encoding, separador)
    'csv_encoding': 'utf-8',            # Codificación del archivo CSV
    'csv_sep': ';',','                      # Separador de columnas

    # Nombres de las columnas esenciales (deben coincidir con el dataset)
    'id_cliente': 'id_cliente',  # Columna que identifica la transacción
    'id_producto': 'id_producto',      # Columna que contiene el ítem (producto/servicio)

    'additional_filters': None,          # Diccionario con {columna: valor} para filtrar

    # Ruta de salida para el archivo de transacciones
    'output_path': 'data/transacciones_listas.csv'
}