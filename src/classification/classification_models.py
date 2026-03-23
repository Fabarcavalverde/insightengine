# src/classification/classification_models.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)


class classification_models:
    """
    Objetivo:
        Ejecutar cada modelo de clasificación del curso de forma individual.
        Cada método entrena, predice y retorna métricas listas para el reporte.

    Uso:
        cm = ClassificationModels(df, target="Depression")
        resultado = cm.run_knn()
        todos     = cm.run_all()
    """

    def __init__(self, datos: pd.DataFrame, target: str, train_size: float = 0.7, random_state: int = 42):
        """
        Objetivo:
            Preparar y dividir los datos una sola vez para todos los modelos.

        Parámetros:
            datos (pd.DataFrame): Dataset limpio con columnas numéricas.
            target (str): Columna a predecir.
            train_size (float): Proporción de entrenamiento.
            random_state (int): Semilla.

        Retorna:
            None
        """
        self.target  = target
        self.clases  = list(np.unique(datos[target].values))

        X = datos.drop(columns=[target])
        y = datos[target]

        # estandarizar
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X),
            columns=X.columns
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, train_size=train_size, random_state=random_state
        )

    def _evaluar(self, model, nombre: str) -> dict:
        """
        Objetivo:
            Entrenar un modelo y calcular sus métricas.

        Parámetros:
            model: Estimador sklearn sin entrenar.
            nombre (str): Nombre del modelo para identificarlo en resultados.

        Retorna:
            dict: {modelo, PG, error_global, precision_por_clase, matriz_confusion}
        """
        model.fit(self.X_train, self.y_train)
        prediccion = model.predict(self.X_test)
        MC = confusion_matrix(self.y_test, prediccion, labels=self.clases)

        precision_global   = np.sum(MC.diagonal()) / np.sum(MC)
        error_global       = 1 - precision_global
        precision_por_clase = dict(zip(self.clases, MC.diagonal() / np.sum(MC, axis=1)))

        return {
            "modelo":             nombre,
            "precision_global":   round(precision_global, 4),
            "error_global":       round(error_global, 4),
            "precision_por_clase": precision_por_clase,
            "matriz_confusion":   MC
        }

    def run_knn(self) -> dict:
        """
        Objetivo: Ejecutar KNN con k=3.
        Retorna: dict con métricas.
        """
        return self._evaluar(KNeighborsClassifier(n_neighbors=3), "KNN")

    def run_svm_rbf(self) -> dict:
        """
        Objetivo: Ejecutar SVM con kernel RBF.
        Retorna: dict con métricas.
        """
        return self._evaluar(SVC(kernel="rbf"), "SVM_RBF")

    def run_svm_poly(self) -> dict:
        """
        Objetivo: Ejecutar SVM con kernel polinomial grado 3.
        Retorna: dict con métricas.
        """
        return self._evaluar(SVC(kernel="poly", degree=3, C=20), "SVM_Poly")

    def run_svm_sigmoid(self) -> dict:
        """
        Objetivo: Ejecutar SVM con kernel sigmoide.
        Retorna: dict con métricas.
        """
        return self._evaluar(SVC(kernel="sigmoid"), "SVM_Sigmoid")

    def run_tree_entropy(self) -> dict:
        """
        Objetivo: Ejecutar árbol de decisión con criterio entropía.
        Retorna: dict con métricas.
        """
        return self._evaluar(
            DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=42),
            "Tree_Entropy"
        )

    def run_tree_gini(self) -> dict:
        """
        Objetivo: Ejecutar árbol de decisión con criterio Gini.
        Retorna: dict con métricas.
        """
        return self._evaluar(
            DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=42),
            "Tree_Gini"
        )

    def run_gradient_boosting(self) -> dict:
        """
        Objetivo: Ejecutar Gradient Boosting.
        Retorna: dict con métricas.
        """
        return self._evaluar(
            GradientBoostingClassifier(n_estimators=300, min_samples_split=7, random_state=42),
            "GradientBoosting"
        )

    def run_random_forest(self) -> dict:
        """
        Objetivo: Ejecutar Random Forest.
        Retorna: dict con métricas.
        """
        return self._evaluar(
            RandomForestClassifier(n_estimators=300, criterion="gini", min_samples_split=11, random_state=42),
            "RandomForest"
        )

    def run_adaboost(self) -> dict:
        """
        Objetivo: Ejecutar AdaBoost con árbol base.
        Retorna: dict con métricas.
        """
        base = DecisionTreeClassifier(min_samples_split=3, max_depth=None, criterion="gini", random_state=42)
        return self._evaluar(
            AdaBoostClassifier(estimator=base, n_estimators=100, random_state=42),
            "AdaBoost"
        )

    def run_naive_bayes(self) -> dict:
        """
        Objetivo: Ejecutar Naive Bayes gaussiano.
        Retorna: dict con métricas.
        """
        return self._evaluar(GaussianNB(), "NaiveBayes")

    def run_lda(self) -> dict:
        """
        Objetivo: Ejecutar Análisis Discriminante Lineal.
        Retorna: dict con métricas.
        """
        return self._evaluar(LinearDiscriminantAnalysis(), "LDA")

    def run_qda(self) -> dict:
        """
        Objetivo: Ejecutar Análisis Discriminante Cuadrático.
        Retorna: dict con métricas.
        """
        return self._evaluar(QuadraticDiscriminantAnalysis(), "QDA")

    def run_all(self) -> pd.DataFrame:
        """
        Objetivo:
            Ejecutar todos los modelos y retornar tabla comparativa.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Filas = modelos, columnas = [PG, error_global, <clases>].
        """
        runners = [
            self.run_knn, self.run_svm_rbf, self.run_svm_poly, self.run_svm_sigmoid,
            self.run_tree_entropy, self.run_tree_gini, self.run_gradient_boosting,
            self.run_random_forest, self.run_adaboost, self.run_naive_bayes,
            self.run_lda, self.run_qda
        ]

        filas = []
        for run in runners:
            r = run()
            fila = {"PG": r["precision_global"], "Error": r["error_global"]}
            fila.update(r["precision_por_clase"])
            filas.append(fila)

        nombres = [
            "KNN", "SVM_RBF", "SVM_Poly", "SVM_Sigmoid",
            "Tree_Entropy", "Tree_Gini", "GradientBoosting",
            "RandomForest", "AdaBoost", "NaiveBayes", "LDA", "QDA"
        ]

        return pd.DataFrame(filas, index=nombres)