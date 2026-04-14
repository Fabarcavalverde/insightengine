import os
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
from typing import Dict, List, Optional, Tuple, Any

# Configuración inicial
warnings.filterwarnings('ignore')
pio.renderers.default = 'browser'


class exploratory_analysis:
    """
    Clase para realizar Análisis Exploratorio de Datos (EDA) y generar
    un reporte HTML interactivo con gráficos Plotly.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "data/processed/eda"):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.results: Dict = {}
        self.figures: Dict[str, Any] = {
            'numeric_histograms': [],
            'categorical_bars': [],
            'date_lines': [],
            'boolean_pies': [],
            'boxplots': [],
            'heatmap': None
        }

    # Helper: detectar columnas booleanas (binarias 0/1)
    def _get_boolean_columns(self) -> List[str]:
        """
        Detecta columnas booleanas: tipo bool nativo O columnas enteras
        cuyos únicos valores no-nulos son {0, 1}.
        Se excluyen de boxplots y matriz de correlación.
        """
        bool_cols = []
        for col in self.df.columns:
            if pd.api.types.is_bool_dtype(self.df[col]):
                bool_cols.append(col)
                continue
            if pd.api.types.is_integer_dtype(self.df[col]):
                unique_vals = set(self.df[col].dropna().unique())
                if unique_vals.issubset({0, 1}):
                    bool_cols.append(col)
        return bool_cols

    # Métodos de análisis

    def general_info(self) -> None:
        """Muestra información general del dataset."""
        print("\n" + "=" * 50)
        print("1. INFORMACIÓN GENERAL")
        print("=" * 50 + "\n")
        print(f"Filas: {self.df.shape[0]}   |   Columnas: {self.df.shape[1]}")
        print("\nNombres de columnas, tipos de dato y valores nulos:")

        dtype_df = pd.DataFrame({
            'Columna': self.df.columns,
            'Tipo': self.df.dtypes.values,
            'No Nulos': self.df.count().values,
            'Nulos': self.df.isnull().sum().values,
            '% Nulos': (self.df.isnull().sum() / len(self.df) * 100).round(2).values
        })
        print(dtype_df.to_string(index=False))

    def missing_values(self) -> pd.DataFrame:
        """Calcula y muestra los valores faltantes por columna."""
        print("\n" + "=" * 50)
        print("2. VALORES FALTANTES")
        print("=" * 50 + "\n")

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({'Cantidad': missing, 'Porcentaje (%)': missing_pct})
        missing_df = missing_df[missing_df['Cantidad'] > 0].sort_values('Porcentaje (%)', ascending=False)
        missing_df = missing_df.reset_index().rename(columns={'index': 'Columna'})

        if missing_df.empty:
            print("No hay valores nulos en el dataset.")
        else:
            print("Valores faltantes por columna:")
            print(missing_df.to_string(index=False))

        self.results['missing'] = missing_df.to_dict()
        return missing_df

    def check_duplicates(self) -> int:
        """Detecta filas duplicadas."""
        print("\n" + "=" * 50)
        print("3. FILAS DUPLICADAS")
        print("=" * 50 + "\n")

        n_duplicates = self.df.duplicated().sum()
        pct = round(n_duplicates / len(self.df) * 100, 2)

        if n_duplicates == 0:
            print("No hay filas duplicadas en el dataset.")
        else:
            print(f"Filas duplicadas encontradas: {n_duplicates} ({pct}% del total)")

        self.results['duplicates'] = n_duplicates
        return n_duplicates

    def descriptive_stats(self) -> Dict:
        """Genera estadísticas descriptivas para variables numéricas y categóricas."""
        print("\n" + "=" * 50)
        print("4. ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 50 + "\n")

        id_patterns = ['id', 'passengerid', 'customerid', 'userid', 'employeeid', 'productid']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not any(p in col.lower() for p in id_patterns)]

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        date_cols = []

        for col in categorical_cols[:]:
            try:
                pd.to_datetime(self.df[col], errors='raise')
                date_cols.append(col)
                categorical_cols.remove(col)
            except Exception:
                pass

        if numeric_cols:
            print("\nVariables numéricas (mean, std, min, 25%, 50%, 75%, max):")
            print(self.df[numeric_cols].describe(percentiles=[.25, .50, .75]).round(2).to_string())
        else:
            print("\nNo hay variables numéricas (o todas son IDs).")

        cat_df = pd.DataFrame()
        if categorical_cols:
            print("\nVariables categóricas (únicos + valor más frecuente):")
            cat_info = []
            for col in categorical_cols:
                unique_vals = self.df[col].nunique()
                most_freq = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None
                freq_count = self.df[col].value_counts().iloc[0] if unique_vals > 0 else 0
                freq_pct = round(freq_count / len(self.df) * 100, 2)
                cat_info.append([col, unique_vals, most_freq, freq_count, freq_pct])
            cat_df = pd.DataFrame(cat_info,
                                  columns=['Columna', 'Valores únicos',
                                           'Valor más frecuente', 'Frecuencia', '% Frecuencia'])
            print(cat_df.to_string(index=False))

        if boolean_cols:
            print("\nVariables booleanas:")
            for col in boolean_cols:
                print(f"  {col}: {self.df[col].value_counts().to_dict()}")

        if date_cols:
            print("\nVariables de fecha detectadas:")
            for col in date_cols:
                col_dt = pd.to_datetime(self.df[col])
                print(f"  {col}: desde {col_dt.min()} hasta {col_dt.max()}")

        stats = {
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'boolean_cols': boolean_cols,
            'date_cols': date_cols,
            'numeric_stats': self.df[numeric_cols].describe(
                percentiles=[.25, .50, .75]).to_dict() if numeric_cols else {},
            'categorical_info': cat_df.to_dict() if not cat_df.empty else {}
        }
        self.results['descriptive_stats'] = stats
        return stats

    def plot_distributions(self) -> None:
        """Genera gráficos de distribución (histogramas, barras, líneas temporales, pasteles)."""
        print("\n" + "=" * 50)
        print("5. DISTRIBUCIÓN DE VARIABLES")
        print("=" * 50 + "\n")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()

        binary_num_cols = [col for col in numeric_cols if set(self.df[col].dropna().unique()).issubset({0, 1})]
        bool_cols = list(set(bool_cols + binary_num_cols))
        real_numeric_cols = [col for col in numeric_cols if col not in binary_num_cols]

        date_cols = []
        for col in categorical_cols[:]:
            try:
                pd.to_datetime(self.df[col], errors='raise')
                date_cols.append(col)
                categorical_cols.remove(col)
            except Exception:
                pass
        date_cols += self.df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        date_cols = list(set(date_cols))

        for col in real_numeric_cols:
            fig = px.histogram(self.df, x=col, title=f'Distribución de {col}',
                               labels={col: col, 'count': 'Frecuencia'},
                               marginal='box', nbins=30)
            fig.update_layout(bargap=0.1, height=500, width=700)
            self.figures['numeric_histograms'].append(fig)

        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 10:
                counts = self.df[col].value_counts().reset_index()
                counts.columns = [col, 'Frecuencia']
                title = f"Todas las categorías ({n_unique}) — {col}"
            else:
                counts = self.df[col].value_counts().head(10).reset_index()
                counts.columns = [col, 'Frecuencia']
                title = f"Top 10 categorías — {col}"
            fig = px.bar(counts, x='Frecuencia', y=col, orientation='h',
                         title=title, labels={'Frecuencia': 'Frecuencia', col: ''})
            fig.update_layout(height=500, width=700, yaxis={'categoryorder': 'total ascending'})
            self.figures['categorical_bars'].append(fig)

        for col in date_cols:
            if self.df[col].dtype == 'object':
                series = pd.to_datetime(self.df[col], errors='coerce')
            else:
                series = self.df[col]
            series = series.dropna()
            if len(series) == 0:
                continue
            unique_days = series.dt.date.nunique()
            if unique_days > 60:
                freq = series.dt.to_period('M').value_counts().sort_index().reset_index()
                freq.columns = ['Mes', 'Conteo']
                freq['Mes'] = freq['Mes'].astype(str)
                fig = px.line(freq, x='Mes', y='Conteo', title=f'Serie temporal de {col} (por mes)',
                              labels={'Mes': 'Mes', 'Conteo': 'Número de registros'})
            else:
                freq = series.dt.date.value_counts().sort_index().reset_index()
                freq.columns = ['Fecha', 'Conteo']
                fig = px.line(freq, x='Fecha', y='Conteo', title=f'Serie temporal de {col} (por día)',
                              labels={'Fecha': 'Fecha', 'Conteo': 'Número de registros'})
            fig.update_layout(height=500, width=800)
            self.figures['date_lines'].append(fig)

        for col in bool_cols:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, 'Conteo']
            counts[col] = counts[col].astype(str)
            fig = px.pie(counts, names=col, values='Conteo', title=f'Distribución de {col} (booleana)')
            fig.update_layout(height=500, width=600)
            self.figures['boolean_pies'].append(fig)

    def detect_outliers(self) -> Dict:
        """
        Detecta outliers usando el método IQR y genera boxplots.
        Las columnas booleanas (0/1) se excluyen: un boxplot de una variable
        binaria no aporta información y distorsiona el reporte.
        """
        print("\n" + "=" * 50)
        print("6. DETECCIÓN DE OUTLIERS (método IQR)")
        print("=" * 50 + "\n")

        bool_cols = self._get_boolean_columns()  # <- FIX
        all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in all_numeric if col not in bool_cols]  # <- FIX

        if not numeric_cols:
            print("No hay variables numéricas (no booleanas) para analizar.")
            return {}

        if bool_cols:
            print(f"  Excluidas del análisis de outliers (booleanas): {bool_cols}\n")

        outlier_info = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_out = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
            pct = round(n_out / len(self.df) * 100, 2)
            outlier_info[col] = {
                'n_outliers': n_out,
                'pct': pct,
                'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4)
            }
            flag = "⚠" if pct > 5 else "✓"
            print(f"  {flag} {col}: {n_out} outliers ({pct}%)  "
                  f"[límites: {round(lower, 2)} — {round(upper, 2)}]")

            fig = px.box(self.df, y=col, title=f'Boxplot — {col}', points='outliers')
            fig.update_layout(height=500, width=600)
            self.figures['boxplots'].append(fig)

        self.results['outliers'] = outlier_info
        return outlier_info

    def plot_correlation(self) -> Optional[pd.DataFrame]:
        """
        Calcula y visualiza la matriz de correlación (heatmap).
        Las columnas booleanas (0/1) se excluyen: al tener varianza casi nula
        o ser variables objetivo, distorsionan las correlaciones y el heatmap.
        """
        print("\n" + "=" * 50)
        print("7. CORRELACIÓN ENTRE VARIABLES NUMÉRICAS")
        print("=" * 50 + "\n")

        bool_cols = self._get_boolean_columns()  # <- FIX
        all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in all_numeric if col not in bool_cols]  # <- FIX

        if len(numeric_cols) < 2:
            print("Se necesitan al menos 2 columnas numéricas (no booleanas) para calcular correlaciones.")
            return None

        if bool_cols:
            print(f"  [INFO] Excluidas de la matriz de correlación (booleanas): {bool_cols}\n")

        # Spearman es más robusto que Pearson: rankea los valores antes de
        # correlacionar, por lo que outliers residuales y distribuciones no
        # normales no distorsionan los coeficientes. Además captura relaciones
        # monótonas no lineales que Pearson no detecta.
        corr_matrix = self.df[numeric_cols].corr(method='spearman')
        print(corr_matrix.round(3).to_string())

        strong = [(corr_matrix.columns[i], corr_matrix.columns[j],
                   round(corr_matrix.iloc[i, j], 3))
                  for i in range(len(corr_matrix.columns))
                  for j in range(i)
                  if abs(corr_matrix.iloc[i, j]) >= 0.7]
        if strong:
            print("\nCorrelaciones fuertes detectadas (|r| ≥ 0.7):")
            for a, b, v in strong:
                print(f"  {a}  ↔  {b}  :  {v}")

        moderate = [(corr_matrix.columns[i], corr_matrix.columns[j],
                     round(corr_matrix.iloc[i, j], 3))
                    for i in range(len(corr_matrix.columns))
                    for j in range(i)
                    if 0.4 <= abs(corr_matrix.iloc[i, j]) < 0.7]
        if moderate:
            print("\nCorrelaciones moderadas detectadas (0.4 ≤ |r| < 0.7):")
            for a, b, v in moderate:
                print(f"  {a}  ↔  {b}  :  {v}")

        fig = px.imshow(corr_matrix, text_auto='.3f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title='Mapa de calor de correlaciones (Spearman)')
        fig.update_layout(height=max(600, len(corr_matrix) * 30), width=max(600, len(corr_matrix) * 30))
        self.figures['heatmap'] = fig
        self.results['correlation'] = corr_matrix.to_dict()
        return corr_matrix

    def classify_variable_types(self) -> Dict:
        """Clasifica las columnas en numéricas, categóricas, booleanas o fechas."""
        print("\n" + "=" * 50)
        print("8. CLASIFICACIÓN DE VARIABLES")
        print("=" * 50 + "\n")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = self.df.select_dtypes(include=['bool']).columns.tolist()

        binary_num_cols = [col for col in numeric_cols if set(self.df[col].dropna().unique()).issubset({0, 1})]
        binary_str_cols = [col for col in categorical_cols if set(self.df[col].dropna().str.lower().unique()).issubset(
            {'yes', 'no', 'true', 'false', '0', '1'})]
        boolean_cols = list(set(boolean_cols + binary_num_cols + binary_str_cols))

        date_cols = self.df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        print(f"Numéricas    ({len(numeric_cols)}):    {numeric_cols if numeric_cols else 'Ninguna'}")
        print(f"Categóricas  ({len(categorical_cols)}): {categorical_cols if categorical_cols else 'Ninguna'}")
        print(f"Booleanas    ({len(boolean_cols)}):    {boolean_cols if boolean_cols else 'Ninguna'}")
        print(f"Fechas       ({len(date_cols)}):       {date_cols if date_cols else 'Ninguna'}")

        var_types = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'boolean': boolean_cols,
            'datetime': date_cols
        }
        self.results['variable_types'] = var_types
        return var_types

    def check_cardinality(self, threshold: int = 50) -> Dict:
        """Muestra la cardinalidad de las variables categóricas."""
        print("\n" + "=" * 50)
        print("9. CARDINALIDAD DE VARIABLES CATEGÓRICAS")
        print("=" * 50 + "\n")

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        cardinality = {}
        if not categorical_cols:
            print("No hay columnas categóricas en el dataset.")
            return cardinality

        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            cardinality[col] = n_unique
            alert = f"  ALTA CARDINALIDAD (> {threshold})" if n_unique > threshold else ""
            print(f"  {col}: {n_unique} categorías{alert}")

        self.results['cardinality'] = cardinality
        return cardinality

    def basic_validation(self) -> Dict:
        """Valida columnas constantes y valores negativos (excluye booleanas)."""
        print("\n" + "=" * 50)
        print("10. VALIDACIONES BÁSICAS")
        print("=" * 50 + "\n")

        bool_cols = self._get_boolean_columns()  # <- FIX
        all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in all_numeric if col not in bool_cols]  # <- FIX

        constant_cols = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            print(f"⚠ Columnas constantes (sin variación): {constant_cols}")
        else:
            print("✓ No hay columnas constantes.")

        negative_cols = [col for col in numeric_cols if (self.df[col] < 0).any()]
        if negative_cols:
            print(f"⚠ Columnas con valores negativos: {negative_cols} (revisar si es esperado)")
        else:
            print("✓ No hay valores negativos en variables numéricas.")

        validation = {'constant_columns': constant_cols, 'negative_columns': negative_cols}
        self.results['validation'] = validation
        return validation

    # Generación de reporte HTML

    def generate_combined_html(self, filename: str = "exploratory_analysis.html") -> str:
        """
        Genera un archivo HTML con todas las figuras almacenadas en self.figures.
        Los histogramas y boxplots de variables numéricas se muestran lado a lado.
        Abre automáticamente el archivo en el navegador.
        """
        output_path = os.path.join(self.output_dir, filename)
        html_parts = []
        html_parts.append("<html><head><title>Reporte EDA Interactivo</title>")
        html_parts.append("""
        <style>
            body { font-family: Arial; margin: 20px; }
            .graph { margin-bottom: 40px; border: 1px solid #ddd; padding: 10px; }
            .row { display: flex; flex-wrap: wrap; margin-bottom: 30px; }
            .column { flex: 1; padding: 10px; min-width: 300px; }
            @media (max-width: 800px) { .column { flex: 100%; } }
        </style>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        </head><body>
        <h1>Análisis Exploratorio de Datos - Reporte Interactivo</h1>
        """)

        # Variables numéricas: histograma + boxplot lado a lado
        histograms = self.figures.get('numeric_histograms', [])
        boxplots = self.figures.get('boxplots', [])
        if histograms and boxplots:
            n_pairs = min(len(histograms), len(boxplots))
            if n_pairs > 0:
                html_parts.append("<h2>Distribución + outliers de variables numéricas</h2>")
                for i in range(n_pairs):
                    html_parts.append("<div class='row'>")
                    html_parts.append(f"<div class='column'>{histograms[i].to_html(full_html=False)}</div>")
                    html_parts.append(f"<div class='column'>{boxplots[i].to_html(full_html=False)}</div>")
                    html_parts.append("</div>")
        else:
            if self.figures.get('numeric_histograms'):
                html_parts.append("<h2>Distribución de variables numéricas</h2>")
                for fig in self.figures['numeric_histograms']:
                    html_parts.append(f"<div class='graph'>{fig.to_html(full_html=False)}</div>")
            if self.figures.get('boxplots'):
                html_parts.append("<h2>Detección de outliers (boxplots)</h2>")
                for fig in self.figures['boxplots']:
                    html_parts.append(f"<div class='graph'>{fig.to_html(full_html=False)}</div>")

        # Resto de secciones
        secciones = [
            ('categorical_bars', 'Distribución de variables categóricas'),
            ('date_lines', 'Distribución temporal de variables fecha'),
            ('boolean_pies', 'Distribución de variables booleanas'),
        ]
        for key, titulo in secciones:
            if self.figures.get(key):
                html_parts.append(f"<h2>{titulo}</h2>")
                for fig in self.figures[key]:
                    html_parts.append(f"<div class='graph'>{fig.to_html(full_html=False)}</div>")

        if self.figures.get('heatmap'):
            html_parts.append("<h2>Matriz de correlación</h2>")
            html_parts.append(f"<div class='graph'>{self.figures['heatmap'].to_html(full_html=False)}</div>")

        html_parts.append("</body></html>")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_parts))
        print(f"Reporte HTML generado: {output_path}")
        webbrowser.open(f'file://{os.path.abspath(output_path)}')
        return output_path

    #  Método principal

    def run_full_analysis(self, output_plots: bool = True, cardinality_threshold: int = 50) -> Dict:
        """
        Ejecuta todo el pipeline de EDA en orden.

        Parámetros
        ----------
        output_plots : bool
            Si es True, genera gráficos y reporte HTML.
        cardinality_threshold : int
            Umbral para alertar alta cardinalidad.

        Retorna
        -------
        Dict
            Diccionario con todos los resultados del análisis.
        """
        print("=" * 70)
        print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("=" * 70)

        self.general_info()
        self.missing_values()
        self.check_duplicates()
        self.descriptive_stats()

        if output_plots:
            self.plot_distributions()
            self.detect_outliers()
            self.plot_correlation()
            self.generate_combined_html()

        self.classify_variable_types()
        self.check_cardinality(cardinality_threshold)
        self.basic_validation()

        print("\n" + "=" * 50)
        print("EDA COMPLETADO.")
        print("=" * 50)

        return self.results


if __name__ == "__main__":
    from ingestion.data_loader import dataloader

    loader = dataloader()
    df = loader.load_file()

    if df is not None:
        reporte = exploratory_analysis(df, output_dir="data/processed/eda")
        resultados = reporte.run_full_analysis()
        print("\nAnálisis completado. Resultados almacenados en 'resultados'.")
    else:
        print("No se pudo cargar el archivo.")