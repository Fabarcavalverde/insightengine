import pandas as pd
from collections import defaultdict


class eclat:
    """
    Objetivo:
        Encontrar itemsets frecuentes usando el algoritmo ECLAT.
        Usa formato vertical (tidsets) e intersección recursiva.
        Basado en el código del profesor (Clase 2).
        Recibe el output de DataPreparator.for_association().

    Uso:
        ec = Eclat(transacciones, min_support=2)
        itemsets = ec.fit()
    """

    def __init__(self, transacciones: list, min_support: int = 2):
        """
        Objetivo:
            Inicializar y construir el formato vertical (item → tidset).

        Parámetros:
            transacciones (list[list[str]]): Lista de transacciones.
            min_support (int): Soporte mínimo absoluto (número de transacciones).

        Retorna:
            None
        """
        self.min_support     = min_support
        self.itemsets        = None
        self._frequent       = {}

        # construir tidsets desde lista de transacciones
        self._item_tidset = self._generate_tidsets(transacciones)

    def _generate_tidsets(self, transacciones: list) -> dict:
        """
        Objetivo:
            Construir el mapa item → conjunto de IDs de transacción (tidset).

        Parámetros:
            transacciones (list[list[str]]): Lista de transacciones.

        Retorna:
            dict: {item (str): tidset (set)}
        """
        item_tidset = defaultdict(set)
        for tid, items in enumerate(transacciones):
            for item in items:
                item_tidset[item].add(tid)
        return item_tidset

    def _eclat(self, prefix: list, items: list):
        """
        Objetivo:
            Búsqueda recursiva en profundidad de itemsets frecuentes por intersección de tidsets.

        Parámetros:
            prefix (list): Prefijo actual del itemset.
            items (list): Lista de (item, tidset) candidatos a extender.

        Retorna:
            None. Acumula resultados en self._frequent.
        """
        while items:
            item, tidset = items.pop()
            support = len(tidset)
            if support >= self.min_support:
                nuevo_itemset = prefix + [item]
                self._frequent[frozenset(nuevo_itemset)] = support

                # construir sufijos por intersección
                sufijo = []
                for otro_item, otro_tidset in items:
                    interseccion = tidset & otro_tidset
                    if len(interseccion) >= self.min_support:
                        sufijo.append((otro_item, interseccion))

                # ordenar para podar más eficientemente
                sufijo = sorted(sufijo, key=lambda x: len(x[1]))
                self._eclat(nuevo_itemset, sufijo)

    def fit(self) -> pd.DataFrame:
        """
        Objetivo:
            Ejecutar ECLAT y retornar itemsets frecuentes.

        Parámetros:
            None

        Retorna:
            pd.DataFrame: Columnas [support, itemsets], ordenado por soporte descendente.

        Output format:
            | support | itemsets                       |
            |---------|--------------------------------|
            | 7       | frozenset({'Butter'})           |
            | 4       | frozenset({'Bread', 'Butter'})  |
        """
        items = sorted(self._item_tidset.items(), key=lambda x: len(x[1]))
        self._eclat([], items)

        resultados = [
            {"support": soporte, "itemsets": itemset}
            for itemset, soporte in self._frequent.items()
        ]

        self.itemsets = pd.DataFrame(resultados) \
                          .sort_values("support", ascending=False) \
                          .reset_index(drop=True)

        return self.itemsets