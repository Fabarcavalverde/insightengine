# src/preprocessing/data_preparator.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreparator:
    """
    Objetivo:
        Preparar el dataset limpio para cada módulo del pipeline.
        Centraliza toda la lógica de transformación previa al modelado,
        incluyendo encoding, estandarización y formato de transacciones.

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
            Codifica categóricas con dummies, estandariza features y conserva el target.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Features estandarizadas + columna target al final.

        Raises:
            ValueError: Si no se definió target en el constructor.


        """
        if self.target is None:
            raise ValueError("Se requiere definir target para clasificación.")

        features = [c for c in self.datos.columns if c != self.target]

        # codificar categóricas
        df_features = pd.get_dummies(self.datos[features])

        # estandarizar
        df_scaled = pd.DataFrame(
            StandardScaler().fit_transform(df_features),
            columns=df_features.columns,
            index=df_features.index
        )

        df_scaled[self.target] = self.datos[self.target].values

        return df_scaled

    def for_dimensionality(self) -> pd.DataFrame:
        """
        Objetivo:
            Preparar datos para PCAReducer, TSNEReducer y UMAPReducer.
            Retorna solo columnas numéricas sin el target, sin estandarizar
            (cada reductor estandariza internamente).

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Solo columnas numéricas, sin target.

        Output format:
            | feature_1 | feature_2 | feature_3 |
            |-----------|-----------|-----------|
            | 1.2       | 3.4       | 0.8       |
        """
        df = self.datos.copy()

        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])

        return df.select_dtypes(include=[np.number])

    def for_association(self, columnas: list = None) -> list:
        """
        Objetivo:
            Preparar lista de transacciones para Apriori y ECLAT.
            Cada fila se convierte en una lista de ítems activos.

        Parámetros:
            columnas (list | None): Columnas a usar como ítems.
                                    Si None, usa todas excepto el target.

        Retorna:
            list[list[str]]: Lista de transacciones.

        Output format:
            [
                ['Outlook_Sunny', 'Wind_Weak', 'PlayTennis_Yes'],
                ['Outlook_Rain',  'Wind_Strong'],
                ...
            ]
        """
        df = self.datos.copy()

        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])

        if columnas:
            df = df[columnas]

        df_encoded = pd.get_dummies(df)

        transacciones = []
        for _, fila in df_encoded.iterrows():
            items = [col for col, val in fila.items() if val]
            transacciones.append(items)

        return transacciones

    def for_regularization(self) -> tuple:
        """
        Objetivo:
            Preparar (X, y) para LassoRidge.
            Codifica categóricas, estandariza features y separa el target numérico.

        Parámetros:
            None

        Retorna:
            tuple: (X escalado pd.DataFrame, y pd.Series)

        Raises:
            ValueError: Si no se definió target en el constructor.
            TypeError: Si el target no es numérico.

        Output format:
            X: pd.DataFrame con features estandarizadas
            y: pd.Series con valores numéricos del target
        """
        if self.target is None:
            raise ValueError("Se requiere definir target para regularización.")

        if not pd.api.types.is_numeric_dtype(self.datos[self.target]):
            raise TypeError(f"El target '{self.target}' debe ser numérico para Lasso/Ridge.")

        features = [c for c in self.datos.columns if c != self.target]
        X = pd.get_dummies(self.datos[features])
        y = self.datos[self.target]

        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_scaled, y