# src/main.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config, RAW_DIR, PROC_DIR
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
from reporting.html_exporter import html_exporter


# ── estado compartido entre pasos ──────────────────────────────────
_state = {}


def _ok(paso, archivo):
    print(f"  [OK] Paso {paso} completado — guardado en data/processed/{archivo}")


def paso_1_carga(cfg):
    dcfg   = cfg["data"]
    loader = data_loader(RAW_DIR)
    df     = loader.load(dcfg)

    mappings = cfg.get("column_mappings", {})
    if mappings:
        df = apply_mappings(df, mappings)

    _state["df"]   = df
    _state["dcfg"] = dcfg

    save_json({
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "columnas_nombres": df.columns.tolist(),
        "muestra": df.head(5).to_dict(orient="records")
    }, "01_ingestion.json", PROC_DIR)

    _ok("1", "01_ingestion.json")


def paso_2_eda(cfg):
    ecfg = cfg["eda"]

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        eda     = exploratory_analysis(_state["df"], output_dir=ecfg["output_dir"])
        results = eda.run_full_analysis(
            output_plots=ecfg["output_plots"],
            cardinality_threshold=ecfg["cardinality_threshold"]
        )

    save_json(results, "02_eda.json", PROC_DIR)
    _ok("2", "02_eda.json  +  eda/exploratory_analysis.html")


def paso_3_limpieza(cfg):
    dcfg = _state["dcfg"]

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        cleaner = data_cleaner(_state["df"])
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

    df_clean           = cleaner.get_clean_dataframe()
    _state["df_clean"] = df_clean
    _state["prep"]     = data_preparator(df_clean, target=dcfg["target"])

    _ok("3", "(datos listos en memoria)")


def paso_4_asociacion(cfg):
    acfg          = cfg["association"]
    dcfg          = _state["dcfg"]
    prep          = _state["prep"]
    columnas_asoc = dcfg.get("columnas_asociacion") or None
    transacciones = prep.for_association(columnas=columnas_asoc)

    ap          = Apriori(transacciones, min_support=acfg["apriori"]["min_support"])
    itemsets_ap = ap.fit()
    reglas_ap   = ap.get_rules(min_confidence=acfg["rules"]["min_confidence"])
    save_json({
        "itemsets": itemsets_ap.to_dict(orient="records"),
        "reglas":   reglas_ap.to_dict(orient="records")
    }, "05_apriori.json", PROC_DIR)
    _ok("4a", "05_apriori.json")

    ec          = eclat(transacciones, min_support=acfg["eclat"]["min_support"])
    itemsets_ec = ec.fit()
    save_json({"itemsets": itemsets_ec.to_dict(orient="records")}, "05_eclat.json", PROC_DIR)
    _ok("4b", "05_eclat.json")


def paso_5_clasificacion(cfg):
    ccfg = cfg["classification"]
    dcfg = _state["dcfg"]
    prep = _state["prep"]
    cm   = classification(
        prep.for_classification(),
        target=dcfg["target"],
        train_size=ccfg["train_size"],
        random_state=ccfg["random_state"]
    )
    resultados_clf = cm.run_all()
    save_json(resultados_clf, "06_classification.json", PROC_DIR)
    _ok("5", "06_classification.json")


def paso_6_dimensionalidad(cfg):
    dimcfg = cfg["dimensionality"]
    km     = dimcfg["kmeans"]
    df_dim = _state["prep"].for_dimensionality()

    for nombre, Clase, kwargs in [
        ("pca",  pca_reducer,  {}),
        ("tsne", tsne_reducer, {"perplexity": dimcfg["tsne"]["perplexity"], "learning_rate": dimcfg["tsne"]["learning_rate"]}),
        ("umap", umap_reducer, {"n_neighbors": dimcfg["umap"]["n_neighbors"]}),
    ]:
        result = Clase(
            df_dim,
            n_components=dimcfg["n_components"],
            n_clusters=dimcfg["n_clusters"],
            max_iter=km["max_iter"],
            n_init=km["n_init"],
            random_state=km["random_state"],
            **kwargs
        ).fit()
        save_json({"resultado": result}, f"07_{nombre}.json", PROC_DIR)
        _ok(f"6 ({nombre.upper()})", f"07_{nombre}.json")


def paso_7_regularizacion(cfg):
    rcfg   = cfg["regularization"]
    prep_r = data_preparator(_state["df_clean"], target=rcfg["target_numerico"])
    X, y   = prep_r.for_regularization()
    lr     = lasso_ridge(alpha=rcfg["alpha"], test_size=rcfg["test_size"])
    resultados = lr.run(X, y)
    save_json(resultados, "08_lasso_ridge.json", PROC_DIR)
    _ok("7", "08_lasso_ridge.json")


def paso_html(cfg):
    exporter = html_exporter(proc_dir=PROC_DIR, eda_dir=cfg["eda"]["output_dir"])
    exporter.export_all()
    _ok("HTML", "../outputs/  (todas las páginas generadas)")


# ── menú ────────────────────────────────────────────────────────────
PASOS = {
    "1": ("Carga + mapeo de columnas",    paso_1_carga),
    "2": ("EDA",                          paso_2_eda),
    "3": ("Limpieza",                     paso_3_limpieza),
    "4": ("Asociación (Apriori + ECLAT)", paso_4_asociacion),
    "5": ("Clasificación",                paso_5_clasificacion),
    "6": ("Reducción de dimensiones",     paso_6_dimensionalidad),
    "7": ("Lasso y Ridge",                paso_7_regularizacion),
}

DEPENDENCIAS = {
    "2": ["1"],
    "3": ["1"],
    "4": ["1", "3"],
    "5": ["1", "3"],
    "6": ["1", "3"],
    "7": ["1", "3"],
}


def verificar_dependencias(opcion):
    reqs = DEPENDENCIAS.get(opcion, [])
    faltantes = []
    if "1" in reqs and "df" not in _state:
        faltantes.append("Paso 1 — Carga")
    if "3" in reqs and "df_clean" not in _state:
        faltantes.append("Paso 3 — Limpieza")
    return faltantes


def mostrar_menu():
    print("\n" + "="*45)
    print("  InsightEngine — Pipeline de Minería")
    print("="*45)
    for k, (nombre, _) in PASOS.items():
        print(f"  {k}. {nombre}")
    print("  T. Ejecutar todo el pipeline")
    print("  H. Generar reporte HTML")
    print("  Q. Salir")
    print("="*45)


def main():
    cfg = load_config()

    while True:
        mostrar_menu()
        opcion = input("  Selecciona una opción: ").strip().upper()

        if opcion == "Q":
            print("  Saliendo...")
            break

        elif opcion == "T":
            print("\n  Ejecutando pipeline completo...\n")
            for _, fn in PASOS.values():
                fn(cfg)
            print("\n  Pipeline completado.")

        elif opcion == "H":
            paso_html(cfg)

        elif opcion in PASOS:
            faltantes = verificar_dependencias(opcion)
            if faltantes:
                print(f"\n  Primero ejecuta: {', '.join(faltantes)}")
            else:
                PASOS[opcion][1](cfg)

        else:
            print("  Opción no válida.")


if __name__ == "__main__":
    main()