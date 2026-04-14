# src/main.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config, RAW_DIR, PROC_DIR, OUT_DIR
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


def _ok(paso, detalle=""):
    msg = f"  [OK] Paso {paso} completado"
    if detalle:
        msg += f" — {detalle}"
    print(msg)


def paso_1_carga(cfg):
    dcfg   = cfg["data"]
    loader = data_loader(RAW_DIR)
    df     = loader.load(dcfg)

    mappings = cfg.get("column_mappings", {})
    if mappings:
        df = apply_mappings(df, mappings)

    _state["df"]   = df
    _state["dcfg"] = dcfg
    _state["ingestion"] = {"filas": df.shape[0], "columnas": df.shape[1]}

    save_json({
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "columnas_nombres": df.columns.tolist(),
        "muestra": df.head(5).to_dict(orient="records")
    }, "01_ingestion.json", PROC_DIR)

    _ok("1 — Carga", f"{df.shape[0]} filas · {df.shape[1]} columnas")


def paso_2_eda(cfg):
    import contextlib, io
    ecfg = cfg["eda"]
    buf  = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eda     = exploratory_analysis(_state["df"], output_dir=ecfg["output_dir"])
        results = eda.run_full_analysis(
            output_plots=ecfg["output_plots"],
            cardinality_threshold=ecfg["cardinality_threshold"]
        )
    _state["eda_results"] = results
    save_json(results, "02_eda.json", PROC_DIR)
    _ok("2 — EDA", "estadísticas guardadas en memoria")


def paso_3_limpieza(cfg):
    import contextlib, io
    dcfg = _state["dcfg"]
    buf  = io.StringIO()
    with contextlib.redirect_stdout(buf):
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

    _ok("3 — Limpieza", f"{df_clean.shape[0]} filas limpias")


def paso_4_asociacion(cfg):
    acfg          = cfg["association"]
    dcfg          = _state["dcfg"]
    prep          = _state["prep"]
    columnas_asoc = dcfg.get("columnas_asociacion") or None
    transacciones = prep.for_association(columnas=columnas_asoc)

    ap          = Apriori(transacciones, min_support=acfg["apriori"]["min_support"])
    itemsets_ap = ap.fit()
    reglas_ap   = ap.get_rules(min_confidence=acfg["rules"]["min_confidence"])

    _state["apriori"] = {"itemsets": itemsets_ap, "reglas": reglas_ap}
    save_json({
        "itemsets": itemsets_ap.to_dict(orient="records"),
        "reglas":   reglas_ap.to_dict(orient="records")
    }, "05_apriori.json", PROC_DIR)
    _ok("4a — Apriori", f"{len(itemsets_ap)} itemsets · {len(reglas_ap)} reglas")

    ec          = eclat(transacciones, min_support=acfg["eclat"]["min_support"])
    itemsets_ec = ec.fit()
    _state["eclat"] = {"itemsets": itemsets_ec}
    save_json({"itemsets": itemsets_ec.to_dict(orient="records")}, "05_eclat.json", PROC_DIR)
    _ok("4b — ECLAT", f"{len(itemsets_ec)} itemsets")


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
    resultados_clf         = cm.run_all()
    _state["clasificacion"] = resultados_clf
    save_json(resultados_clf, "06_classification.json", PROC_DIR)
    mejor = resultados_clf["PG"].idxmax()
    _ok("5 — Clasificación", f"mejor: {mejor} ({resultados_clf.loc[mejor,'PG']:.2%})")


def paso_6_dimensionalidad(cfg):
    dimcfg = cfg["dimensionality"]
    km     = dimcfg["kmeans"]
    df_dim = _state["prep"].for_dimensionality()

    for nombre, key, Clase, kwargs in [
        ("ACP",   "pca",  pca_reducer,  {}),
        ("t-SNE", "tsne", tsne_reducer, {"perplexity": dimcfg["tsne"]["perplexity"], "learning_rate": dimcfg["tsne"]["learning_rate"]}),
        ("UMAP",  "umap", umap_reducer, {"n_neighbors": dimcfg["umap"]["n_neighbors"]}),
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
        _state[key] = result
        save_json({"resultado": result}, f"07_{key}.json", PROC_DIR)
        _ok(f"6 — {nombre}")


def paso_7_regularizacion(cfg):
    rcfg   = cfg["regularization"]
    prep_r = data_preparator(_state["df_clean"], target=rcfg["target_numerico"])
    X, y   = prep_r.for_regularization()
    lr     = lasso_ridge(alpha=rcfg["alpha"], test_size=rcfg["test_size"])
    resultados              = lr.run(X, y)
    _state["regularizacion"] = resultados
    save_json(resultados, "08_lasso_ridge.json", PROC_DIR)
    _ok("7 — Lasso/Ridge", f"R² Lasso={resultados['lasso']['r2']} · R² Ridge={resultados['ridge']['r2']}")


def paso_html(cfg):
    print("  Generando HTML...")
    exporter = html_exporter(
        out_dir      = OUT_DIR,
        eda_results  = _state.get("eda_results", {}),
        datos        = _state
    )
    exporter.export_all()
    _ok("H — HTML", f"outputs/ (7 páginas generadas)")


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
    "H": ["4"],
}


def verificar_dependencias(opcion):
    reqs = DEPENDENCIAS.get(opcion, [])
    faltantes = []
    if "1" in reqs and "df" not in _state:
        faltantes.append("Paso 1 — Carga")
    if "3" in reqs and "df_clean" not in _state:
        faltantes.append("Paso 3 — Limpieza")
    if "4" in reqs and "apriori" not in _state:
        faltantes.append("Paso 4 — Asociación")
    return faltantes


def mostrar_menu():
    print("\n" + "="*45)
    print("  InsightEngine — Pipeline de Minería")
    print("="*45)
    for k, (nombre, _) in PASOS.items():
        estado = " ✓" if k == "1" and "df" in _state else \
                 " ✓" if k == "3" and "df_clean" in _state else \
                 " ✓" if k == "4" and "apriori" in _state else \
                 " ✓" if k == "5" and "clasificacion" in _state else \
                 " ✓" if k == "6" and "pca" in _state else \
                 " ✓" if k == "7" and "regularizacion" in _state else ""
        print(f"  {k}. {nombre}{estado}")
    print("  T. Ejecutar todo el pipeline")
    print("  H. Generar reporte HTML")
    print("  Q. Salir")
    print("="*45)


def main():
    cfg = load_config()

    while True:
        mostrar_menu()
        try:
            opcion = input("  Selecciona una opción: ").strip().upper()
        except (UnicodeDecodeError, EOFError):
            continue

        if opcion == "Q":
            print("  Saliendo...")
            break

        elif opcion == "T":
            print("\n  Ejecutando pipeline completo...\n")
            for _, fn in PASOS.values():
                fn(cfg)
            print("\n  Pipeline completado.")

        elif opcion == "H":
            faltantes = verificar_dependencias("H")
            if faltantes:
                print(f"\n  Primero ejecuta: {', '.join(faltantes)}")
            else:
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