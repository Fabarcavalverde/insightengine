# src/main.py

import os
import pandas as pd
from classification.classification_models import classification_models
from dimensionality.pca_reducer  import PCAReducer
from dimensionality.tsne_reducer import TSNEReducer
from dimensionality.umap_reducer import UMAPReducer
from preprocessing.data_preparator import DataPreparator

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

    # --- reducción de dimensiones ---
    df_dim = prep.for_dimensionality()

    print("\n--- ACP ---")
    pca = PCAReducer(df_dim, n_clusters=3)
    print(pca.fit())

    print("\n--- t-SNE ---")
    tsne = TSNEReducer(df_dim, perplexity=3, n_clusters=3)
    print(tsne.fit())

    print("\n--- UMAP ---")
    umap = UMAPReducer(df_dim, n_neighbors=2, n_clusters=3)
    print(umap.fit())
if __name__ == "__main__":
    main()