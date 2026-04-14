import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class Apriori:
    """
    Objetivo:
        Encontrar itemsets frecuentes usando el algoritmo Apriori.
        Recibe el output de DataPreparator.for_association().

    Uso:
        ap = Apriori(transacciones, min_support=0.3)
        itemsets = ap.fit()
        reglas   = ap.get_rules(min_confidence=0.6)
    """

    def __init__(self, transacciones: list, min_support: float = 0.3):
        """
        Objetivo:
            Inicializar y codificar las transacciones en formato booleano.

        Parámetros:
            transacciones (list[list[str]]): Lista de transacciones.
            min_support (float): Soporte mínimo (0.0 - 1.0).

        Retorna:
            None
        """
        self.min_support = min_support
        self.itemsets    = None
        self.reglas      = None

        # codificar a formato booleano
        te = TransactionEncoder()
        te_array = te.fit_transform(transacciones)
        self.df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    def fit(self) -> pd.DataFrame:
        """
        Objetivo:
            Ejecutar Apriori y retornar itemsets frecuentes.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Columnas [support, itemsets].

        Output format:
            | support | itemsets                  |
            |---------|---------------------------|
            | 0.53    | frozenset({'credit_A32'}) |
        """
        self.itemsets = apriori(
            self.df_encoded,
            min_support=self.min_support
        )

        # agregar nombres de columnas a los frozensets
        col_map = {i: col for i, col in enumerate(self.df_encoded.columns)}
        self.itemsets["itemsets"] = self.itemsets["itemsets"].apply(
            lambda x: frozenset(col_map[i] for i in x)
        )

        return self.itemsets

    def get_rules(self, min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Objetivo:
            Generar reglas de asociación a partir de los itemsets frecuentes.

        Parámetros:
            min_confidence (float): Confianza mínima (0.0 - 1.0).

        Retorna:
            pd.DataFrame: Columnas [antecedents, consequents, support, confidence, lift].

        Raises:
            ValueError: Si fit() no fue ejecutado antes.
        """
        if self.itemsets is None:
            raise ValueError("Ejecutar fit() antes de get_rules().")

        self.reglas = association_rules(
            self.itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        return self.reglas[["antecedents", "consequents", "support", "confidence", "lift"]]