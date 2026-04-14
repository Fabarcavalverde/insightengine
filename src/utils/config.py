import os
import yaml

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config() -> dict:
    """
    Objetivo:
        Cargar el archivo config.yaml desde la raíz del proyecto.

    Parámetros:
        None

    Retorna:
        dict: Configuración completa del pipeline.

    Raises:
        FileNotFoundError: Si config.yaml no existe en la raíz.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"No se encontró config.yaml en {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)