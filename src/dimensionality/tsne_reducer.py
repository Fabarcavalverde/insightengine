# src/dimensionality/tsne_reducer.py

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class TSNEReducer:
    """
    Objetivo:
        Reducir dimensiones con t-SNE y agrupar con K-Medias.

    Uso:
        tsne = TSNEReducer(df, n_components=2, perplexity=3, n_clusters=3)
        resultado = tsne.fit()
    """

    def __init__(
        self,
        datos: pd.DataFrame,
        n_components: int = 2,
        perplexity: float = 3,
        learning_rate: str = "auto",
        n_clusters: int = 3,
        max_iter: int = 2000,
        n_init: int = 150,
        random_state: int = 42
    ):
        """
        Objetivo:
            Inicializar el reductor con parámetros de t-SNE y K-Medias.

        Parámetros:
            datos (pd.DataFrame): Dataset numérico sin estandarizar.
            n_components (int): Número de componentes a generar.
            perplexity (float): Vecinos cercanos a considerar. Debe ser menor al número de muestras.
                                Rango recomendado: 5-50. Datasets grandes requieren valores mayores.
            learning_rate (str | float): Tasa de aprendizaje. "auto" es recomendado.
            n_clusters (int): Número de clústeres para K-Medias.
            max_iter (int): Iteraciones máximas de K-Medias.
            n_init (int): Semillas distintas para K-Medias.
            random_state (int): Semilla de reproducibilidad.

        Retorna:
            None
        """
        self.datos         = datos
        self.n_components  = n_components
        self.perplexity    = perplexity
        self.learning_rate = learning_rate
        self.n_clusters    = n_clusters
        self.max_iter      = max_iter
        self.n_init        = n_init
        self.random_state  = random_state

        # estandarizar
        self.datos_std = pd.DataFrame(
            StandardScaler().fit_transform(datos),
            columns=datos.columns,
            index=datos.index
        )

        self.df_resultado = None
        self.grupos       = None
        self.centros      = None

    def fit(self) -> pd.DataFrame:
        """
        Objetivo:
            Aplicar t-SNE y K-Medias sobre los datos estandarizados.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Componentes con columnas [dim1, dim2, ..., cluster].

        """
        # reducción
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            init="random"
        ).fit_transform(self.datos_std)

        cols = [f"dim{i+1}" for i in range(self.n_components)]
        self.df_resultado = pd.DataFrame(tsne, columns=cols, index=self.datos_std.index)

        # clustering
        kmedias = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter,
                         n_init=self.n_init, random_state=self.random_state)
        kmedias.fit(self.df_resultado)
        self.grupos = kmedias.predict(self.df_resultado)
        self.df_resultado["cluster"] = self.grupos

        # centroides sobre datos originales
        self.centros = self._calcular_centros()

        return self.df_resultado

    def _calcular_centros(self) -> np.ndarray:
        """
        Objetivo:
            Calcular el centroide de cada clúster sobre los datos originales.

        Parámetros:
            None

        Retorna:
            np.ndarray: Matriz de centroides (n_clusters x n_features).
        """
        centro = self._centroide(0)
        for j in range(1, self.n_clusters):
            centro = pd.concat([centro, self._centroide(j)])
        return np.array(centro)

    def _centroide(self, num_cluster: int) -> pd.DataFrame:
        """
        Objetivo:
            Calcular la media de un clúster específico.

        Parámetros:
            num_cluster (int): Índice del clúster.

        Retorna:
            pd.DataFrame: Fila con la media de cada variable del clúster.
        """
        ind = self.grupos == num_cluster
        return pd.DataFrame(self.datos.values[ind].mean(axis=0, keepdims=True),
                            columns=self.datos.columns)
