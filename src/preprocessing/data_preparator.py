# src/preprocessing/data_preparator.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class data_preparator:
    """
    Objetivo:
        Preparar el dataset limpio para cada módulo del pipeline.
        Centraliza encoding, estandarización y formato transaccional.

    Uso:
        prep = DataPreparator(df, target="risk")
        transacciones  = prep.for_association()
        df_clf         = prep.for_classification()
        df_dim         = prep.for_dimensionality()
    """

    def __init__(self, datos: pd.DataFrame, target: str = None):
        """
        Objetivo:
            Inicializar con el dataset ya limpio.

        Parámetros:
            datos (pd.DataFrame): Dataset limpio.
            target (str | None): Columna a predecir o excluir según el módulo.

        Retorna:
            None
        """
        self.datos  = datos.copy()
        self.target = target

    def for_association(self, columnas: list = None) -> list:
        """
        Objetivo:
            Preparar lista de transacciones para Apriori y ECLAT.
            Excluye automáticamente columnas numéricas y el target.

        Parámetros:
            columnas (list | None): Columnas a usar. Si None o vacío,
                                    usa todas las categóricas excepto target.

        Retorna:
            list[list[str]]: Lista de transacciones.
        """
        df = self.datos.copy()

        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])

        if columnas and isinstance(columnas, list) and len(columnas) > 0:
            df = df[columnas]
        else:
            # solo categóricas
            df = df.select_dtypes(exclude=[np.number])

        df_encoded = pd.get_dummies(df)

        transacciones = []
        for _, fila in df_encoded.iterrows():
            items = [col for col, val in fila.items() if val]
            transacciones.append(items)

        return transacciones

    def for_classification(self) -> pd.DataFrame:
        """
        Objetivo:
            Preparar datos para ClassificationModels.
            Codifica categóricas, estandariza features y conserva el target.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Features estandarizadas + columna target.

        Raises:
            ValueError: Si no se definió target.
        """
        if self.target is None:
            raise ValueError("Se requiere target para clasificación.")

        features = [c for c in self.datos.columns if c != self.target]
        df_feat  = pd.get_dummies(self.datos[features])

        df_scaled = pd.DataFrame(
            StandardScaler().fit_transform(df_feat),
            columns=df_feat.columns,
            index=df_feat.index
        )
        df_scaled[self.target] = self.datos[self.target].values

        return df_scaled

    def for_dimensionality(self) -> pd.DataFrame:
        """
        Objetivo:
            Preparar datos para PCAReducer, TSNEReducer y UMAPReducer.
            Retorna solo columnas numéricas sin el target.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Solo columnas numéricas, sin target.
        """
        df = self.datos.copy()

        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])

        return df.select_dtypes(include=[np.number])

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
            ValueError: Si no se definió target.
            TypeError: Si el target no es numérico.
        """
        if self.target is None:
            raise ValueError("Se requiere target para regularización.")

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