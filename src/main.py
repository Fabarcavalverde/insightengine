# src/main.py

import os
import sys

# agregar src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config, RAW_DIR
from ingestion.data_loader import DataLoader
from preprocessing.data_preparator import data_preparator
from association.apriori import Apriori
from association.eclat import eclat
from classification.classification import classification
from dimensionality.pca_reducer import pca_reducer
from dimensionality.tsne_reducer import tsne_reducer
from dimensionality.umap_reducer import umap_reducer


def main():
    cfg = load_config()
    dcfg = cfg["data"]
    acfg = cfg["association"]
    ccfg = cfg["classification"]
    dimcfg = cfg["dimensionality"]
    km = dimcfg["kmeans"]

    # cargar datos
    loader = DataLoader(RAW_DIR)
    df = loader.load(dcfg)

    # corregir typos si los hay
    for col, reemplazos in dcfg.get("typos", {}).items():
        for malo, bueno in reemplazos.items():
            df[col] = df[col].replace(malo, bueno)

    if dcfg.get("drop_na"):
        df = df.dropna()

    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(df.head(3))

    # preparador central — un solo df para todo
    prep = data_preparator(df, target=dcfg["target"])

    # ----------------------------------------------------------------
    # asociación
    # ----------------------------------------------------------------
    columnas_asoc = dcfg.get("columnas_asociacion") or None
    transacciones = prep.for_association(columnas=columnas_asoc)

    print("\n--- Apriori: itemsets frecuentes ---")
    ap = Apriori(transacciones, min_support=acfg["apriori"]["min_support"])
    itemsets_ap = ap.fit()
    print(itemsets_ap)

    print("\n--- Apriori: reglas de asociación ---")
    reglas_ap = ap.get_rules(min_confidence=acfg["rules"]["min_confidence"])
    print(reglas_ap)

    print("\n--- ECLAT: itemsets frecuentes ---")
    ec = eclat(transacciones, min_support=acfg["eclat"]["min_support"])
    print(ec.fit())

    # ----------------------------------------------------------------
    # clasificación
    # ----------------------------------------------------------------
    print("\n--- Clasificación ---")
    cm = classification(
        prep.for_classification(),
        target=dcfg["target"],
        train_size=ccfg["train_size"],
        random_state=ccfg["random_state"]
    )
    print(cm.run_all().to_string())

    # ----------------------------------------------------------------
    # reducción de dimensiones
    # ----------------------------------------------------------------
    df_dim = prep.for_dimensionality()

    print("\n--- ACP ---")
    print(pca_reducer(df_dim, n_components=dimcfg["n_components"],
                     n_clusters=dimcfg["n_clusters"], max_iter=km["max_iter"],
                     n_init=km["n_init"], random_state=km["random_state"]).fit())

    print("\n--- t-SNE ---")
    print(tsne_reducer(df_dim, n_components=dimcfg["n_components"],
                      perplexity=dimcfg["tsne"]["perplexity"],
                      learning_rate=dimcfg["tsne"]["learning_rate"],
                      n_clusters=dimcfg["n_clusters"], max_iter=km["max_iter"],
                      n_init=km["n_init"], random_state=km["random_state"]).fit())

    print("\n--- UMAP ---")
    print(umap_reducer(df_dim, n_components=dimcfg["n_components"],
                      n_neighbors=dimcfg["umap"]["n_neighbors"],
                      n_clusters=dimcfg["n_clusters"], max_iter=km["max_iter"],
                      n_init=km["n_init"], random_state=km["random_state"]).fit())


if __name__ == "__main__":
    main()