# src/preprocessing/data_preparator.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreparator:
    """
    Objetivo:
        Preparar el dataset limpio para cada módulo del pipeline.
        Centraliza toda la lógica de transformación previa al modelado.

    Uso:
        prep = DataPreparator(df, target="PlayTennis")
        df_clf  = prep.for_classification()
        df_dim  = prep.for_dimensionality()
        trans   = prep.for_association()
        X, y    = prep.for_regularization()
    """

    def __init__(self, datos: pd.DataFrame, target: str = None):
        """
        Objetivo:
            Inicializar con el dataset ya limpio.

        Parámetros:
            datos (pd.DataFrame): Dataset limpio, puede tener columnas categóricas.
            target (str | None): Columna a predecir. Requerida para clasificación y regularización.

        Retorna:
            None
        """
        self.datos  = datos.copy()
        self.target = target

    def for_classification(self) -> pd.DataFrame:
        """
        Objetivo:
            Preparar datos para ClassificationModels.
            Codifica categóricas con dummies, conserva el target como columna.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Features encoded + columna target al final.

        Raises:
            ValueError: Si no se definió target en el constructor.

        """
        if self.target is None:
            raise ValueError("Se requiere definir target para clasificación.")

        features = [c for c in self.datos.columns if c != self.target]

        # codificar solo las features
        df_features = pd.get_dummies(self.datos[features])
        df_features[self.target] = self.datos[self.target].values

        return df_features

    def for_dimensionality(self) -> pd.DataFrame:
        """
        Objetivo:
            Preparar datos para PCAReducer, TSNEReducer y UMAPReducer.
            Retorna solo columnas numéricas sin el target.
            La estandarización la hace cada reductor internamente.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Solo columnas numéricas, sin target.

        """
        df = self.datos.copy()

        # quitar target si existe
        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])

        # solo numéricas
        df_num = df.select_dtypes(include=[np.number])

        return df_num

    def for_association(self, columnas: list = None) -> list:
        """
        Objetivo:
            Preparar lista de transacciones para Apriori y ECLAT.
            Cada fila del df se convierte en una lista de ítems activos.

        Parámetros:
            columnas (list | None): Columnas a usar como ítems.
                                    Si None, usa todas excepto el target.

        Retorna:
            list[list[str]]: Lista de transacciones.

        """
        df = self.datos.copy()

        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])

        if columnas:
            df = df[columnas]

        # codificar categóricas
        df_encoded = pd.get_dummies(df)

        # cada fila → lista de columnas con valor verdadero
        transacciones = []
        for _, fila in df_encoded.iterrows():
            items = [col for col, val in fila.items() if val]
            transacciones.append(items)

        return transacciones

