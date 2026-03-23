# InsightEngine

Pipeline de minería de datos orientado a objetos para descubrimiento de patrones de asociación, reducción de dimensiones y regularización.

**Curso:** Minería de Datos II — BD-162  
**Institución:** Colegio Universitario de Cartago  
**Grupo:** #1 — Abarca Valverde Fiorella · Contreras Artavia Fernando · Barquero Carvajal Johel
---

## Estructura del proyecto

```
InsightEngine/
│
├── data/
│   ├── raw/                        # Datasets originales sin modificar
│   └── processed/                  # Outputs intermedios del pipeline
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point — orquesta todo el pipeline
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── data_loader.py          # Paso 1: carga de datos
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   └── exploratory_analysis.py # Paso 2: análisis exploratorio
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_cleaner.py         # Paso 3: limpieza y transformación
│   │
│   ├── association/
│   │   ├── __init__.py
│   │   ├── apriori.py              # Paso 5.1: algoritmo Apriori
│   │   └── eclat.py                # Paso 5.2: algoritmo ECLAT
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── predictive_analysis.py  # Clase base AnalisisPredictivo
│   │   └── classification_models.py# Naive Bayes, LDA, QDA, KNN
│   │
│   ├── dimensionality/
│   │   ├── __init__.py
│   │   ├── pca_reducer.py          # Paso 7.1: ACP
│   │   ├── tsne_reducer.py         # Paso 7.2: t-SNE
│   │   └── umap_reducer.py         # Paso 7.3: UMAP
│   │
│   ├── regularization/
│   │   ├── __init__.py
│   │   └── lasso_ridge.py          # Paso 8: Lasso y Ridge
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py               # Parámetros globales y rutas
│   │
│   └── reporting/
│       ├── __init__.py
│       └── html_exporter.py        # Paso 9: genera report.html
││
├── tests/
│   └── test_pipeline.py
│
├── README.md
└── requirements.txt
```

---

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
mlxtend
umap-learn
jinja2
```

---

## Ejecución

```bash
python src/main.py
```

El pipeline ejecuta todos los pasos en orden y genera `outputs/report.html` con los resultados completos.

---

## Pasos del pipeline

| Paso | Módulo | Responsable | Descripción |
|------|--------|-------------|-------------|
| 1 | `ingestion/data_loader.py` | Fernando    | Carga del dataset desde `data/raw/` |
| 2 | `eda/exploratory_analysis.py` | Fernando    | Estadísticas descriptivas y visualizaciones |
| 3 | `preprocessing/data_cleaner.py` | Fernando    | Limpieza, imputación y transformación |
| 5.1 | `association/apriori.py` | Fernando    | Itemsets frecuentes con Apriori |
| 5.2 | `association/eclat.py` | Fernando    | Itemsets frecuentes con ECLAT |
| 6 | `association/association_rules.py` | Fiorella    | Generación y filtrado de reglas |
| 7.1 | `dimensionality/pca_reducer.py` | Fiorella    | Reducción con ACP |
| 7.2 | `dimensionality/tsne_reducer.py` | Fiorella    | Reducción con t-SNE |
| 7.3 | `dimensionality/umap_reducer.py` | Fiorella    | Reducción con UMAP |
| 8 | `regularization/lasso_ridge.py` | ---         | Regresión Lasso y Ridge |
| 9 | `reporting/html_exporter.py` | ---         | Exportación del reporte HTML |

---

## Estándares de código

### Nomenclatura

- Archivos `.py` en minúsculas con guion bajo. El nombre del archivo coincide con la función principal que contiene.
- Funciones nombradas en inglés con convención descriptiva.

### Documentación de funciones

Cada función debe documentar:

```python
def run_apriori(transactions: list[list[str]], min_support: float = 0.05) -> pd.DataFrame:
    """
    Objective:
        Apply the Apriori algorithm to find frequent itemsets.

    Parameters:
        transactions (list[list[str]]): List of transactions.
        min_support (float): Minimum support threshold (0.0 - 1.0).

    Returns:
        pd.DataFrame: Frequent itemsets with columns ['support', 'itemsets'].

    Input format:
        [['item_a', 'item_b'], ['item_b', 'item_c'], ...]

    Output format:
        | support | itemsets         |
        |---------|------------------|
        | 0.12    | frozenset({'a'}) |
    """
```

---

## Estándares de commits

### Ramas

Cada miembro trabaja en su rama personal. La rama `main` se actualiza solo con versiones estables.

```
main
├── fiorella
├── fernando
└── johel
```

### Formato de commits

```
feat:     Nueva funcionalidad
fix:      Corrección de bug
docs:     Solo documentación
style:    Formato, sin cambio de lógica
refactor: Refactorización sin nueva funcionalidad
perf:     Mejora de rendimiento
test:     Pruebas
chore:    Mantenimiento y dependencias
build:    Herramientas de compilación
ci:       Configuración de integración continua
revert:   Reversión de commit anterior
```

**Ejemplos:**

```bash
feat: add apriori frequent itemset mining with min_support param
feat: add association_rules generation with lift filtering
feat: add pca_reducer with explained variance plot
fix: handle empty transactions in eclat preprocessing
docs: update README with pipeline execution steps
refactor: extract transaction encoder to data_cleaner
```

---

## Datasets

Los datasets deben colocarse en `data/raw/` antes de ejecutar el pipeline. El archivo `src/utils/config.py` centraliza las rutas y parámetros configurables.