# src/main.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config, RAW_DIR, PROC_DIR
from utils.json_saver import save_json
from utils.apply_mappings import apply_mappings
from ingestion.data_loader import data_loader
from preprocessing.data_cleaner import data_cleaner
from eda.exploratory_analysis import exploratory_analysis
from preprocessing.data_preparator import data_preparator
from association.apriori import Apriori
from association.eclat import eclat
from classification.classification import classification
from dimensionality.pca_reducer import pca_reducer
from dimensionality.tsne_reducer import tsne_reducer
from dimensionality.umap_reducer import umap_reducer
from regularization.lasso_ridge import lasso_ridge


def main():
    cfg    = load_config()
    dcfg   = cfg["data"]
    acfg   = cfg["association"]
    ccfg   = cfg["classification"]
    dimcfg = cfg["dimensionality"]
    rcfg   = cfg["regularization"]
    ecfg   = cfg["eda"]
    km     = dimcfg["kmeans"]

    # ----------------------------------------------------------------
    # 1. carga de datos
    # ----------------------------------------------------------------
    loader = data_loader(RAW_DIR)
    df     = loader.load(dcfg)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    save_json({
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "columnas_nombres": df.columns.tolist(),
        "muestra": df.head(5).to_dict(orient="records")
    }, "01_ingestion.json", PROC_DIR)

    # ----------------------------------------------------------------
    # 2. mapeo de columnas — antes de limpiar y del EDA
    # ----------------------------------------------------------------
    mappings = cfg.get("column_mappings", {})
    if mappings:
        print("\n--- Aplicando mapeo de columnas ---")
        df = apply_mappings(df, mappings)
        print(df[["checking_account", "credit_history", "housing"]].head(3))

    # ----------------------------------------------------------------
    # 3. limpieza
    # ----------------------------------------------------------------
    print("\n--- Limpieza ---")
    cleaner = data_cleaner(df)
    cleaner.drop_constant_columns()
    cleaner.drop_id_columns()
    cleaner.remove_duplicates()
    cleaner.impute_numeric(strategy="median")
    cleaner.impute_categorical(strategy="mode")

    for col, reemplazos in dcfg.get("typos", {}).items():
        for malo, bueno in reemplazos.items():
            cleaner.df[col] = cleaner.df[col].replace(malo, bueno)

    if dcfg.get("drop_na"):
        cleaner.df = cleaner.df.dropna()

    df_clean = cleaner.get_clean_dataframe()
    print(f"Dataset limpio: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")

    # ----------------------------------------------------------------
    # 4. EDA — sobre datos limpios y mapeados
    # ----------------------------------------------------------------
    print("\n--- EDA ---")
    eda     = exploratory_analysis(df_clean, output_dir=ecfg["output_dir"])
    results = eda.run_full_analysis(
        output_plots=ecfg["output_plots"],
        cardinality_threshold=ecfg["cardinality_threshold"]
    )
    save_json(results, "02_eda.json", PROC_DIR)

    # ----------------------------------------------------------------
    # 5. preparador central
    # ----------------------------------------------------------------
    prep = data_preparator(df_clean, target=dcfg["target"])

    # ----------------------------------------------------------------
    # 6. asociación
    # ----------------------------------------------------------------
    columnas_asoc = dcfg.get("columnas_asociacion") or None
    transacciones = prep.for_association(columnas=columnas_asoc)

    print("\n--- Apriori: itemsets frecuentes ---")
    ap          = Apriori(transacciones, min_support=acfg["apriori"]["min_support"])
    itemsets_ap = ap.fit()
    print(itemsets_ap)

    print("\n--- Apriori: reglas de asociación ---")
    reglas_ap = ap.get_rules(min_confidence=acfg["rules"]["min_confidence"])
    print(reglas_ap)

    save_json({
        "itemsets": itemsets_ap.to_dict(orient="records"),
        "reglas":   reglas_ap.to_dict(orient="records")
    }, "05_apriori.json", PROC_DIR)

    print("\n--- ECLAT: itemsets frecuentes ---")
    ec          = eclat(transacciones, min_support=acfg["eclat"]["min_support"])
    itemsets_ec = ec.fit()
    print(itemsets_ec)

    save_json({
        "itemsets": itemsets_ec.to_dict(orient="records")
    }, "05_eclat.json", PROC_DIR)

    # ----------------------------------------------------------------
    # 7. clasificación
    # ----------------------------------------------------------------
    print("\n--- Clasificación ---")
    cm             = classification(
        prep.for_classification(),
        target=dcfg["target"],
        train_size=ccfg["train_size"],
        random_state=ccfg["random_state"]
    )
    resultados_clf = cm.run_all()
    print(resultados_clf.to_string())

    save_json(resultados_clf, "06_classification.json", PROC_DIR)

    # ----------------------------------------------------------------
    # 8. reducción de dimensiones
    # ----------------------------------------------------------------
    df_dim = prep.for_dimensionality()

    print("\n--- ACP ---")
    pca_result = pca_reducer(
        df_dim, n_components=dimcfg["n_components"],
        n_clusters=dimcfg["n_clusters"], max_iter=km["max_iter"],
        n_init=km["n_init"], random_state=km["random_state"]
    ).fit()
    print(pca_result)
    save_json({"resultado": pca_result}, "07_pca.json", PROC_DIR)

    print("\n--- t-SNE ---")
    tsne_result = tsne_reducer(
        df_dim, n_components=dimcfg["n_components"],
        perplexity=dimcfg["tsne"]["perplexity"],
        learning_rate=dimcfg["tsne"]["learning_rate"],
        n_clusters=dimcfg["n_clusters"], max_iter=km["max_iter"],
        n_init=km["n_init"], random_state=km["random_state"]
    ).fit()
    print(tsne_result)
    save_json({"resultado": tsne_result}, "07_tsne.json", PROC_DIR)

    print("\n--- UMAP ---")
    umap_result = umap_reducer(
        df_dim, n_components=dimcfg["n_components"],
        n_neighbors=dimcfg["umap"]["n_neighbors"],
        n_clusters=dimcfg["n_clusters"], max_iter=km["max_iter"],
        n_init=km["n_init"], random_state=km["random_state"]
    ).fit()
    print(umap_result)
    save_json({"resultado": umap_result}, "07_umap.json", PROC_DIR)

    # ----------------------------------------------------------------
    # 9. Lasso y Ridge
    # ----------------------------------------------------------------
    print("\n--- Lasso y Ridge ---")
    prep_r        = data_preparator(df_clean, target=rcfg["target_numerico"])
    X, y          = prep_r.for_regularization()
    lr            = lasso_ridge(alpha=rcfg["alpha"], test_size=rcfg["test_size"])
    resultados_lr = lr.run(X, y)

    for nombre, metricas in resultados_lr.items():
        print(f"\n{nombre.upper()}")
        for metrica, valor in metricas.items():
            print(f"  {metrica.upper()}: {valor}")

    save_json(resultados_lr, "08_lasso_ridge.json", PROC_DIR)

    print("\nPipeline completado. JSONs guardados en data/processed/")


if __name__ == "__main__":
    main()