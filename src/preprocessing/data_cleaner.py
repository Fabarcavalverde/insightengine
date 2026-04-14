import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Any


class data_cleaner:
    """
    Clase para limpieza básica de datos.
    Realiza operaciones fundamentales: eliminar constantes/IDs, imputar nulos,
    manejar outliers, reducir cardinalidad.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log: List[str] = []
        self._log("Inicializado DataCleaner")

    def _log(self, message: str) -> None:
        """Añade un mensaje al registro interno y lo imprime en consola."""
        print(f"[CLEANER] {message}")
        self.cleaning_log.append(message)

    def _get_boolean_columns(self) -> List[str]:
        """
        Detecta columnas booleanas: tipo bool nativo O columnas enteras
        cuyos únicos valores no-nulos son {0, 1}.
        Estas columnas se excluyen de imputación numérica y análisis de outliers.
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

    def get_cleaning_log(self) -> List[str]:
        """Devuelve el registro completo de limpieza."""
        return self.cleaning_log

    def drop_constant_columns(self) -> None:
        """Elimina columnas que tienen un solo valor único (constantes)."""
        constant_cols = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            self.df = self.df.drop(columns=constant_cols)
            self._log(f"Eliminadas columnas constantes: {constant_cols}")
        else:
            self._log("No se encontraron columnas constantes.")

    def drop_id_columns(self, patterns: List[str] = None) -> None:
        """
        Elimina columnas que parecen identificadores.
        Por defecto busca patrones como 'id', 'ID', 'Id' al final o completo.
        También elimina columnas completamente únicas cuyo nombre contenga el patrón.
        """
        if patterns is None:
            patterns = ['id', 'ID', 'Id', 'cliente_id', 'customer_id']
        id_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns):
                if self.df[col].nunique() == len(self.df):
                    id_cols.append(col)
        if id_cols:
            self.df = self.df.drop(columns=id_cols)
            self._log(f"Eliminadas columnas ID: {id_cols}")
        else:
            self._log("No se encontraron columnas ID para eliminar.")

    def convert_dtypes(self, categorical_threshold: int = 20) -> None:
        """
        Convierte tipos de datos:
        - Fechas string a datetime. Anula fechas con año fuera de [2000, año_actual]
          para evitar que valores erróneos distorsionen ejes temporales.
        - Booleanos en string ('Sí'/'No', 'True'/'False', etc.) a bool.
        - Columnas object con pocos valores únicos a category.
        """
        import datetime
        current_year = datetime.datetime.now().year
        date_columns = []

        # 1. Detectar y convertir fechas
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                converted = pd.to_datetime(self.df[col], errors='coerce', format='mixed')
                non_null_original = self.df[col].notna().sum()
                if non_null_original == 0:
                    continue
                success_rate = converted.notna().sum() / non_null_original
                if success_rate >= 0.8:
                    # Anular fechas con años fuera de rango
                    invalid_year_mask = (
                            converted.notna() &
                            ((converted.dt.year < 2000) | (converted.dt.year > current_year))
                    )
                    if invalid_year_mask.sum() > 0:
                        converted[invalid_year_mask] = pd.NaT
                        self._log(
                            f"Anuladas {invalid_year_mask.sum()} fechas con año fuera de "
                            f"[2000, {current_year}] en '{col}'"
                        )
                    self.df[col] = converted
                    self._log(f"Convertida columna '{col}' a datetime (tasa de éxito {success_rate:.1%})")
                    date_columns.append(col)
            except Exception:
                pass

        # 2. Detectar y convertir booleanos en string
        bool_map = {'si': True, 'sí': True, 'yes': True, 'true': True, '1': True,
                    'no': False, 'false': False, '0': False}
        for col in self.df.select_dtypes(include=['object']).columns:
            if col in date_columns:
                continue
            unique_lower = self.df[col].dropna().astype(str).str.lower().unique()
            if len(unique_lower) > 0 and set(unique_lower).issubset(bool_map.keys()):
                self.df[col] = self.df[col].astype(str).str.lower().map(bool_map)
                self.df[col] = self.df[col].astype('boolean')
                self._log(f"Convertida columna '{col}' a booleana")

        # 3. Convertir categóricas de baja cardinalidad a tipo 'category'
        for col in self.df.select_dtypes(include=['object']).columns:
            if col in date_columns:
                continue
            if self.df[col].nunique() <= categorical_threshold:
                self.df[col] = self.df[col].astype('category')
                self._log(f"Convertida columna '{col}' a category (cardinalidad {self.df[col].nunique()})")

    def impute_numeric(self, strategy: str = 'median', fill_value: float = None) -> None:
        """
        Imputa valores nulos en columnas numéricas.
        Las columnas booleanas (0/1) se excluyen automáticamente.

        Parámetros:
        -----------
        strategy : str
            'mean', 'median', 'constant'
        fill_value : float
            Valor a usar si strategy='constant'.
        """
        bool_cols = self._get_boolean_columns()
        numeric_cols = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if col not in bool_cols
        ]

        if len(numeric_cols) == 0:
            self._log("No hay columnas numéricas (no booleanas) para imputar.")
            return

        if bool_cols:
            self._log(f"Excluidas de imputación numérica las columnas booleanas: {bool_cols}")

        for col in numeric_cols:
            if self.df[col].isnull().sum() == 0:
                continue
            if strategy == 'mean':
                val = self.df[col].mean()
            elif strategy == 'median':
                val = self.df[col].median()
            elif strategy == 'constant':
                if fill_value is None:
                    raise ValueError("Para strategy='constant' debe proporcionar fill_value")
                val = fill_value
            else:
                raise ValueError(f"Estrategia '{strategy}' no soportada. Use 'mean', 'median' o 'constant'.")
            self.df[col] = self.df[col].fillna(val)
            self._log(f"Imputados nulos en columna numérica '{col}' con {strategy}={val:.4f}")

    def impute_categorical(self, strategy: str = 'mode', fill_value: str = 'DESCONOCIDO') -> None:
        """
        Imputa valores nulos en columnas categóricas (object o category).

        Parámetros:
        -----------
        strategy : str
            'mode' (valor más frecuente) o 'constant'
        fill_value : str
            Valor a usar si strategy='constant'.
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            self._log("No hay columnas categóricas para imputar.")
            return

        for col in categorical_cols:
            if self.df[col].isnull().sum() == 0:
                continue
            if strategy == 'mode':
                mode_val = self.df[col].mode()
                val = mode_val[0] if len(mode_val) > 0 else fill_value
                self.df[col] = self.df[col].fillna(val)
                self._log(f"Imputados nulos en columna categórica '{col}' con moda={val}")
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)
                self._log(f"Imputados nulos en columna categórica '{col}' con valor constante '{fill_value}'")
            else:
                raise ValueError(f"Estrategia '{strategy}' no soportada. Use 'mode' o 'constant'.")

    def handle_negative_values(
            self,
            columns: List[str] = None,
            method: str = 'clip',
            fill_value: float = 0.0
    ) -> None:
        """
        Trata valores negativos en columnas numéricas que no deberían tenerlos.
        Las columnas booleanas se excluyen automáticamente.

        Parámetros:
        -----------
        columns : list, optional
            Columnas a tratar. Si None, se usan todas las numéricas no booleanas.
        method : str
            'clip'     → reemplaza negativos por fill_value (por defecto 0).
            'nan'      → convierte negativos a NaN (para imputar después).
            'absolute' → usa el valor absoluto.
        fill_value : float
            Valor de reemplazo cuando method='clip'. Por defecto 0.0.
        """
        bool_cols = self._get_boolean_columns()
        if columns is None:
            columns = [
                col for col in self.df.select_dtypes(include=[np.number]).columns
                if col not in bool_cols
            ]
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                self._log(f"[AVISO] Columnas no encontradas en el DataFrame, se omiten: {missing}")
            columns = [c for c in columns if c in self.df.columns and c not in bool_cols]

        for col in columns:
            neg_mask = self.df[col] < 0
            n_neg = neg_mask.sum()
            if n_neg == 0:
                continue
            if method == 'clip':
                self.df.loc[neg_mask, col] = fill_value
                self._log(f"'{col}': {n_neg} valores negativos reemplazados por {fill_value}")
            elif method == 'nan':
                self.df.loc[neg_mask, col] = np.nan
                self._log(f"'{col}': {n_neg} valores negativos convertidos a NaN")
            elif method == 'absolute':
                self.df.loc[neg_mask, col] = self.df.loc[neg_mask, col].abs()
                self._log(f"'{col}': {n_neg} valores negativos convertidos a valor absoluto")
            else:
                raise ValueError(f"Método '{method}' no soportado. Use 'clip', 'nan' o 'absolute'.")

    def validate_numeric_range(
            self,
            column: str,
            min_val: Optional[float] = None,
            max_val: Optional[float] = None,
            method: str = 'nan'
    ) -> None:
        """
        Invalida o corrige valores fuera de un rango de dominio conocido.
        Aplica reglas de negocio fijas, independiente de la distribución estadística.

        Diferencia con handle_outliers_iqr: este método usa límites conocidos del dominio
        (ej. edad bancaria válida: 18-100), no límites estadísticos (IQR).

        Parámetros:
        -----------
        column : str
            Columna numérica a validar.
        min_val : float, optional
            Valor mínimo permitido (inclusive).
        max_val : float, optional
            Valor máximo permitido (inclusive).
        method : str
            'nan'  → convierte valores fuera de rango a NaN (se imputan después).
            'clip' → recorta al límite más cercano.
            'drop' → elimina las filas con valores fuera de rango.

        Ejemplos de uso:
        ----------------
        cleaner.validate_numeric_range('edad', min_val=18, max_val=100, method='nan')
        cleaner.validate_numeric_range('tasa_interes_asignada', min_val=5.0, max_val=30.0, method='nan')
        cleaner.validate_numeric_range('porc_utilizacion_tarjeta', min_val=0.0, max_val=1.0, method='clip')
        """
        if column not in self.df.columns:
            self._log(f"Columna '{column}' no encontrada. Se omite validación de rango.")
            return

        mask = pd.Series([False] * len(self.df), index=self.df.index)
        if min_val is not None:
            mask |= self.df[column] < min_val
        if max_val is not None:
            mask |= self.df[column] > max_val

        n_invalid = mask.sum()
        if n_invalid == 0:
            self._log(f"'{column}': todos los valores dentro del rango [{min_val}, {max_val}].")
            return

        if method == 'nan':
            self.df.loc[mask, column] = np.nan
            self._log(
                f"'{column}': {n_invalid} valores fuera de [{min_val}, {max_val}] "
                f"→ NaN (se imputarán en impute_numeric)"
            )
        elif method == 'clip':
            self.df[column] = self.df[column].clip(lower=min_val, upper=max_val)
            self._log(
                f"'{column}': {n_invalid} valores recortados al rango [{min_val}, {max_val}]"
            )
        elif method == 'drop':
            before = len(self.df)
            self.df = self.df[~mask].reset_index(drop=True)
            self._log(
                f"'{column}': {before - len(self.df)} filas eliminadas "
                f"por valores fuera de [{min_val}, {max_val}]"
            )
        else:
            raise ValueError(f"Método '{method}' no soportado. Use 'nan', 'clip' o 'drop'.")

    def standardize_categorical(
            self,
            column: str,
            mapping: Dict[str, str],
            default: Optional[str] = None,
            case_sensitive: bool = False
    ) -> None:
        """
        Estandariza los valores de una columna categórica según un mapeo explícito.
        Por defecto el matching es insensible a mayúsculas/minúsculas y espacios
        laterales, por lo que no es necesario listar todas las variantes de casing.

        Parámetros:
        -----------
        column : str
            Nombre de la columna a estandarizar.
        mapping : dict
            Diccionario {valor_original: valor_estandarizado}.
            Con case_sensitive=False las claves se normalizan internamente,
            así 'BUENO', 'bueno' y 'Bueno' son equivalentes.
        default : str, optional
            Valor para registros que no estén en el mapping.
            Si None → se convierten a NaN (recomendado: imputar después).
        case_sensitive : bool
            Si True, el matching es exacto. Por defecto False.
        """
        if column not in self.df.columns:
            self._log(f"Columna '{column}' no encontrada. Se omite estandarización.")
            return

        original_dtype = self.df[column].dtype
        before_unique = self.df[column].nunique(dropna=False)

        if case_sensitive:
            lookup = mapping
            self.df[column] = self.df[column].map(
                lambda x: lookup.get(x, default) if pd.notna(x) else x
            )
        else:
            normalized_mapping = {
                str(k).strip().lower(): v for k, v in mapping.items()
            }

            def _match(x):
                if pd.isna(x):
                    return x
                key = str(x).strip().lower()
                if key in normalized_mapping:
                    return normalized_mapping[key]
                return default

            self.df[column] = self.df[column].map(_match)

        after_unique = self.df[column].nunique(dropna=False)
        n_nulls_new = self.df[column].isnull().sum()

        if original_dtype.name == 'category':
            self.df[column] = self.df[column].astype('category')

        self._log(
            f"Estandarizada '{column}': {before_unique} → {after_unique} valores únicos "
            f"({n_nulls_new} nulos tras estandarización)"
        )

    def handle_outliers_iqr(self, columns: List[str] = None, method: str = 'cap') -> None:
        """
        Detecta y trata outliers usando el rango intercuartil (IQR).
        Las columnas booleanas (0/1) se excluyen automáticamente para evitar
        boxplots sin sentido y alteraciones de la variable objetivo.

        Parámetros:
        -----------
        columns : list, optional
            Lista de columnas numéricas a procesar. Si None, se usan todas las
            numéricas no booleanas.
        method : str
            'cap'    → winsorize: limita valores a los límites IQR.
            'remove' → elimina filas con outliers.
            'median' → reemplaza outliers por la mediana.
        """
        bool_cols = self._get_boolean_columns()

        if columns is None:
            columns = [
                col for col in self.df.select_dtypes(include=[np.number]).columns.tolist()
                if col not in bool_cols
            ]
        else:
            columns = [col for col in columns if col not in bool_cols]

        if not columns:
            self._log("No hay columnas numéricas (no booleanas) para tratar outliers.")
            return

        if bool_cols:
            self._log(f"Excluidas del análisis de outliers las columnas booleanas: {bool_cols}")

        rows_before = len(self.df)
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = (self.df[col] < lower) | (self.df[col] > upper)
            n_outliers = outliers.sum()
            if n_outliers == 0:
                continue

            if method == 'cap':
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                self._log(f"Outliers en '{col}' capados a [{lower:.2f}, {upper:.2f}] ({n_outliers} valores)")
            elif method == 'remove':
                self.df = self.df[~outliers]
                self._log(f"Eliminadas {n_outliers} filas con outliers en '{col}'")
            elif method == 'median':
                median_val = self.df[col].median()
                self.df.loc[outliers, col] = median_val
                self._log(f"Outliers en '{col}' reemplazados por mediana={median_val} ({n_outliers} valores)")
            else:
                raise ValueError(f"Método '{method}' no soportado. Use 'cap', 'remove' o 'median'.")

        rows_after = len(self.df)
        if method == 'remove' and rows_after != rows_before:
            self._log(f"Total de filas eliminadas: {rows_before - rows_after}")

    def remove_duplicates(self) -> None:
        """Elimina filas duplicadas (todas las columnas idénticas)."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        if before != after:
            self._log(f"Eliminadas {before - after} filas duplicadas")
        else:
            self._log("No se encontraron filas duplicadas")

    def save_cleaned_data(self, filepath: str) -> None:
        """
        Guarda el DataFrame limpio en un archivo CSV.

        Parámetros:
        -----------
        filepath : str
            Ruta completa donde guardar (ej. 'data/processed/clean_data.csv')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.df.to_csv(filepath, index=False)
        self._log(f"DataFrame limpio guardado en {filepath}")

    def get_clean_dataframe(self) -> pd.DataFrame:
        """Devuelve el DataFrame limpio actual."""
        return self.df


# ============================================================================
# Bloque de prueba (se ejecuta solo si se corre directamente este script)
# ============================================================================
if __name__ == "__main__":
    from preprocessing.data_loader import data_loader

    loader = data_loader()
    df = loader.load_file(r"C:\Users\ferna\Downloads\dataset_riesgo_crediticio.csv")

    if df is not None:
        cleaner = data_cleaner(df)

        cleaner.drop_constant_columns()
        cleaner.drop_id_columns()
        cleaner.validate_numeric_range('edad', min_val=18, max_val=100, method='nan')
        cleaner.validate_numeric_range('porc_utilizacion_tarjeta', min_val=0.0, max_val=1.0, method='clip')
        cleaner.validate_numeric_range('tasa_interes_asignada', min_val=5.0, max_val=30.0, method='nan')
        cleaner.handle_negative_values(columns=None, method='clip', fill_value=0.0)
        cleaner.standardize_categorical(
            column='genero',
            mapping={'m': 'M', 'male': 'M', 'masculino': 'M',
                     'f': 'F', 'female': 'F', 'femenino': 'F'},
            default=None
        )
        cleaner.standardize_categorical(
            column='historial_crediticio',
            mapping={'malo': 'Malo', 'regular': 'Regular',
                     'bueno': 'Bueno', 'muy bueno': 'Bueno', 'sin histo': 'Bueno',
                     'excelente': 'Excelente'},
            default=None
        )
        cleaner.convert_dtypes(categorical_threshold=20)
        cleaner.handle_outliers_iqr(method='cap')
        cleaner.impute_numeric(strategy='median')
        cleaner.impute_categorical(strategy='mode')
        cleaner.remove_duplicates()

        print("\n=== REGISTRO DE LIMPIEZA ===")
        for log in cleaner.get_cleaning_log():
            print(log)

        print("\n=== DATAFRAME LIMPIO (primeras 5 filas) ===")
        print(cleaner.get_clean_dataframe().head())

        cleaner.save_cleaned_data("data/processed/cleaned_dataset.csv")
        print("\nLimpieza completada. Archivo guardado en 'data/processed/cleaned_dataset.csv'")
    else:
        print("No se pudo cargar el archivo.")