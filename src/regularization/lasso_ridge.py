import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class lasso_ridge:

    def __init__(self, alpha: float = 1.0, test_size: float = 0.2):
        self.alpha = alpha
        self.test_size = test_size

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        lasso = Lasso(alpha=self.alpha)
        ridge = Ridge(alpha=self.alpha)

        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        lasso_pred = lasso.predict(X_test)
        ridge_pred = ridge.predict(X_test)

        return {
            "lasso": {
                "mse": mean_squared_error(y_test, lasso_pred),
                "r2": r2_score(y_test, lasso_pred),
            },
            "ridge": {
                "mse": mean_squared_error(y_test, ridge_pred),
                "r2": r2_score(y_test, ridge_pred),
            }
        }


# Carga del dataset

df = pd.read_csv(r"C:\insightengine\data\dataset_riesgo_crediticio.csv")
print("Columnas del dataset:")
print(df.columns)


# Seleccion

X = df[["ingresos_mensuales", "gastos_mensuales", "deuda_total"]]

y = df["tasa_interes_asignada"]

# Modelo

model = lasso_ridge()

results = model.run(X, y)

print("\n===== Resultados del modelo =====\n")

for model_name, metrics in results.items():
    print(f"Modelo: {model_name.upper()}")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    print("-" * 30)


