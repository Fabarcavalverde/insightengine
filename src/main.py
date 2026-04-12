# src/main.py

import os
import pandas as pd
from preprocessing.data_preparator import DataPreparator
from classification.classification import classification
from dimensionality.pca_reducer  import pca_reducer
from dimensionality.tsne_reducer import tsne_reducer
from dimensionality.umap_reducer import umap_reducer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")


def main():
    # cargar datos — reemplazar con data_loader de Fernando
    df = pd.read_csv(os.path.join(RAW_DIR, "playtennis.csv"), delimiter=",", decimal=".")
    target = "PlayTennis"

    # corregir typos y NaN
    df[target] = df[target].replace("Ye", "Yes")
    df = df.dropna()

    # preparador central
    prep = DataPreparator(df, target=target)

    # --- clasificación ---
    print("\n--- Clasificación ---")
    classification_df = classification(prep.for_classification(), target=target)
    print(classification_df.run_all().to_string())

    # --- reducción de dimensiones ---
    dimensionality_df = prep.for_dimensionality()

    print("\n--- ACP ---")
    print(pca_reducer(dimensionality_df, n_clusters=3).fit())

    print("\n--- t-SNE ---")
    print(tsne_reducer(dimensionality_df, perplexity=3, n_clusters=3).fit())

    print("\n--- UMAP ---")
    print(umap_reducer(dimensionality_df, n_neighbors=2, n_clusters=3).fit())


if __name__ == "__main__":
    main()