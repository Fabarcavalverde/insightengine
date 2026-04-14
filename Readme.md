# InsightEngine

Pipeline de minería de datos orientado a objetos para descubrimiento de patrones de asociación y generación de reglas de recomendación para el negocio.

**Curso:** Minería de Datos II — BD-162  
**Institución:** Colegio Universitario de Cartago  
**Grupo:** #1 — Abarca Valverde Fiorella · Contreras Artavia Fernando · Barquero Carvajal Johel  
**Entrega:** 14 de abril de 2026

---

## Estructura del proyecto

```
InsightEngine/
│
├── data/
│   ├── raw/                         # Datasets originales sin modificar
│   └── processed/                   # Outputs del pipeline (generados, no subir a git)
│
├── src/
│   ├── __init__.py
│   ├── main.py                      # Entry point — menú interactivo
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── data_loader.py           # Paso 1: carga de datos
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   └── exploratory_analysis.py  # Paso 2: estadísticas y visualizaciones
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py          # Paso 3: limpieza e imputación
│   │   └── data_preparator.py       # Paso 4: transformación por módulo
│   │
│   ├── association/
│   │   ├── __init__.py
│   │   ├── apriori.py               # Paso 5.1: itemsets frecuentes con Apriori
│   │   └── eclat.py                 # Paso 5.2: itemsets frecuentes con ECLAT
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   └── classification.py        # Paso 6: KNN, SVM, árboles, NaiveBayes, LDA, QDA
│   │
│   ├── dimensionality/
│   │   ├── __init__.py
│   │   ├── pca_reducer.py           # Paso 7.1: ACP
│   │   ├── tsne_reducer.py          # Paso 7.2: t-SNE
│   │   └── umap_reducer.py          # Paso 7.3: UMAP
│   │
│   ├── regularization/
│   │   ├── __init__.py
│   │   └── lasso_ridge.py           # Paso 8: Lasso y Ridge
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                # Rutas y carga del config.yaml
│   │   ├── apply_mappings.py        # Mapeo de códigos a etiquetas legibles
│   │   └── json_saver.py            # Serialización de resultados a JSON
│   │
│   └── reporting/
│       ├── __init__.py
│       └── html_exporter.py         # Generación del reporte HTML desde JSONs
│
├── outputs/                         # Reporte HTML final (generado, no subir a git)
│
├── tests/
│   └── test_pipeline.py
│
├── config.yaml                      # Parámetros configurables del pipeline
├── environment.yml                  # Entorno conda reproducible
├── README.md
└── requirements.txt
```

---

## Configuración del entorno

### Clonar el repositorio

```bash
git clone https://github.com/Fabarcavalverde/insightengine.git
cd insightengine
```

### Crear el entorno conda

```bash
conda env create -f environment.yml
conda activate InsightEngine
```


---

## Uso con cualquier dataset

InsightEngine puede usarse con cualquier dataset tabular. Solo hay que colocar el archivo en `data/raw/` y configurar `config.yaml`. No se modifica ningún archivo de código.

### Requisitos del dataset

| Requisito | Descripción |
|-----------|-------------|
| Formato | CSV, TSV o cualquier archivo de texto delimitado |
| Columnas | Al menos una columna categórica para las reglas de asociación |
| Target | Una columna que actúe como variable objetivo (puede ser categórica o numérica) |
| Tamaño recomendado | Entre 500 y 100,000 filas. Datasets muy grandes pueden hacer ECLAT lento |

### Configuración en config.yaml

```yaml
# ── OBLIGATORIO ─────────────────────────────────────────────────────

data:
  filename: "mi_dataset.csv"   # nombre del archivo en data/raw/
  delimiter: ","               # separador: "," para CSV, ";" para punto y coma, " " para espacio
  decimal: "."                 # separador decimal: "." o ","
  header: infer                # "infer" si el archivo tiene encabezado, null si no tiene
  target: "mi_columna"         # columna a predecir (clasificación) o excluir (asociación)
  drop_na: true                # eliminar filas con valores nulos

  # solo si header: null — lista de nombres para las columnas en orden
  columnas: []

  # columnas categóricas a usar en asociación
  # vacío [] = usa automáticamente todas las categóricas excepto el target
  columnas_asociacion: []


# ── OPCIONAL — solo si los valores tienen códigos crípticos ─────────

column_mappings:
  nombre_columna:
    "codigo_original": "etiqueta_legible"
    "A11": "saldo_negativo"
  otra_columna:
    "1": "riesgo_bueno"
    "2": "riesgo_malo"


# ── PARÁMETROS DE MODELOS ────────────────────────────────────────────

association:
  apriori:
    min_support: 0.3        # proporción mínima (0.0 - 1.0). Bajar si hay pocos patrones
  eclat:
    min_support: 300        # valor absoluto (número de transacciones)
  rules:
    min_confidence: 0.6     # confianza mínima para filtrar reglas (0.0 - 1.0)
    min_threshold: 1.0      # lift mínimo

classification:
  train_size: 0.7           # proporción de entrenamiento
  random_state: 42

dimensionality:
  n_components: 2           # dimensiones de salida (2 para visualización)
  n_clusters: 3             # clusters para K-Medias
  tsne:
    perplexity: 30          # entre 5 y 50. Aumentar para datasets grandes
    learning_rate: "auto"
  umap:
    n_neighbors: 15         # entre 5 y 50. Valores bajos = estructura local
  kmeans:
    max_iter: 2000
    n_init: 150
    random_state: 42

regularization:
  target_numerico: "columna_numerica"  # columna numérica continua a predecir
  alpha: 1.0                           # penalización (mayor = más regularización)
  test_size: 0.2

eda:
  output_dir: "data/processed/eda"
  cardinality_threshold: 50
  output_plots: true
```

### Guía rápida para cambiar de dataset

1. Colocar el archivo en `data/raw/`
2. Actualizar `data.filename` y `data.target` en `config.yaml`
3. Si el archivo no tiene encabezado, cambiar `header: null` y definir `columnas`
4. Si los valores son códigos crípticos, definir `column_mappings`
5. Ajustar `min_support` según el tamaño del dataset (datasets grandes → valor mayor)
6. Correr `python src/main.py` y seleccionar `T`

---

## Ejecución

```bash
conda activate InsightEngine
python src/main.py
```

El pipeline se controla desde un menú interactivo:

```
=============================================
  InsightEngine — Pipeline de Minería
=============================================
  1. Carga + mapeo de columnas
  2. EDA
  3. Limpieza
  4. Asociación (Apriori + ECLAT)
  5. Clasificación
  6. Reducción de dimensiones
  7. Lasso y Ridge
  T. Ejecutar todo el pipeline
  H. Generar reporte HTML
  Q. Salir
=============================================
```

Cada paso guarda sus resultados en `data/processed/` como JSON. La opción `H` genera el reporte HTML desde esos JSONs sin necesidad de correr el pipeline de nuevo.

---

## Pasos del pipeline

| Paso    | Módulo                             | Descripción                                              |
| ------- | ---------------------------------- | -------------------------------------------------------- |
| 1       | `ingestion/data_loader.py`         | Carga del dataset + mapeo de columnas                    |
| 2       | `eda/exploratory_analysis.py`      | Estadísticas descriptivas y visualizaciones interactivas |
| 3       | `preprocessing/data_cleaner.py`    | Limpieza, corrección de typos, imputación                |
| 4       | `preprocessing/data_preparator.py` | Transformación a formato transaccional y features        |
| 5.1     | `association/apriori.py`           | Itemsets frecuentes con Apriori + reglas                 |
| 5.2     | `association/eclat.py`             | Itemsets frecuentes con ECLAT                            |
| 6       | `classification/classification.py` | KNN, SVM, árboles, ensembles, NaiveBayes, LDA, QDA       |
| 7.1-7.3 | `dimensionality/`                  | ACP, t-SNE, UMAP con clustering K-Medias                 |
| 8       | `regularization/lasso_ridge.py`    | Regresión Lasso y Ridge                                  |
| H       | `reporting/html_exporter.py`       | Reporte HTML desde JSONs                                 |

---

## Archivos ignorados por git

Los siguientes directorios se generan al correr el pipeline y **no deben subirse al repositorio**:

```
data/processed/
outputs/
**/__pycache__/
**/*.pyc
```

---

## Estándares de código

- Archivos `.py` en minúsculas con guion bajo
- Funciones nombradas en inglés
- Docstrings y comentarios en español
- Una clase por archivo, nombre del archivo = nombre de la clase principal

### Formato de commits

```
feat:     Nueva funcionalidad
fix:      Corrección de bug
docs:     Solo documentación
style:    Formato sin cambio de lógica
refactor: Refactorización sin nueva funcionalidad
chore:    Mantenimiento y dependencias
revert:   Reversión de commit anterior
```

### Ramas

```
main        ← solo versiones estables
├── fiorella
├── fernando
└── johel
```