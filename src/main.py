# src/main.py

import os
import pandas as pd
from classification.classification_models import classification_models

# rutas relativas a la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")


def main():
    # cargar datos
    df = pd.read_csv(os.path.join(RAW_DIR, "playtennis.csv"), delimiter=",", decimal=".")
    target = "PlayTennis"

    # corregir typos en el target
    df[target] = df[target].replace("Ye", "Yes")

    # eliminar filas con NaN
    df = df.dropna()

    # codificar solo las features, no el target
    features = [c for c in df.columns if c != target]
    df_encoded = pd.get_dummies(df[features])
    df_encoded[target] = df[target].values

    cm = classification_models(df_encoded, target=target)

    # correr todos los modelos
    resultados = cm.run_all()
    print(resultados.to_string())


if __name__ == "__main__":
    main()