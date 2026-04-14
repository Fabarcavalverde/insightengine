import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# src/regularization/lasso_ridge.py

import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class lasso_ridge:
    """
    Objetivo:
        Ejecutar regresión Lasso y Ridge y comparar métricas.
        Recibe X, y ya preparados desde DataPreparator.for_regularization().

    Uso:
        model = LassoRidge(alpha=1.0, test_size=0.2)
        resultados = model.run(X, y)
    """

    def __init__(self, alpha: float = 1.0, test_size: float = 0.2):
        """
        Objetivo:
            Inicializar con hiperparámetros desde config.yaml.

        Parámetros:
            alpha (float): Factor de penalización para Lasso y Ridge.
            test_size (float): Proporción del conjunto de prueba.

        Retorna:
            None
        """
        self.alpha     = alpha
        self.test_size = test_size

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Objetivo:
            Entrenar Lasso y Ridge y retornar métricas de evaluación.

        Parámetros:
            X (pd.DataFrame): Features estandarizadas.
            y (pd.Series): Variable objetivo numérica.

        Retorna:
            dict: {lasso: {mse, r2}, ridge: {mse, r2}}

        Output format:
            {
                "lasso": {"mse": 0.23, "r2": 0.75},
                "ridge": {"mse": 0.21, "r2": 0.77}
            }
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        lasso = Lasso(alpha=self.alpha, max_iter=1000)
        ridge = Ridge(alpha=self.alpha, max_iter=1000)

        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        lasso_pred = lasso.predict(X_test)
        ridge_pred = ridge.predict(X_test)

        return {
            "lasso": {
                "mse": round(mean_squared_error(y_test, lasso_pred), 4),
                "r2":  round(r2_score(y_test, lasso_pred), 4),
            },
            "ridge": {
                "mse": round(mean_squared_error(y_test, ridge_pred), 4),
                "r2":  round(r2_score(y_test, ridge_pred), 4),
            }
        }