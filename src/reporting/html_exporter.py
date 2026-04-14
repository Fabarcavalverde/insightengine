import os


class html_exporter:

    def __init__(self, out_dir: str, eda_results: dict, datos: dict):
        self.out_dir     = out_dir
        self.eda_results = eda_results
        self.datos       = datos
        os.makedirs(self.out_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _safe_sample(self, df, max_n=1500):
        """Reduce tamaño de dataset para visualización"""
        if df is not None and len(df) > max_n:
            return df.sample(n=max_n, random_state=42)
        return df

    def _nav(self, activa):
        pages = [
            ("index.html","Resumen"),
            ("eda.html","EDA"),
            ("apriori.html","Apriori"),
            ("eclat.html","ECLAT"),
            ("classification.html","Clasificación"),
            ("dimensionality.html","Dimensionalidad"),
            ("lasso_ridge.html","Lasso / Ridge")
        ]
        links = "".join(
            f'<a href="{h}" class="{"active" if l==activa else ""}">{l}</a>'
            for h,l in pages
        )
        return f'<nav><div class="nav-inner">{links}</div></nav>'

    def _page(self, titulo, activa, contenido):
        return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{titulo}</title>

<!-- Plotly CDN -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

<style>
body{{font-family:sans-serif;background:#f4f6fb;margin:0}}
nav{{background:#1E2761;padding:10px}}
nav a{{color:white;margin-right:10px;text-decoration:none}}
nav a.active{{color:#4FC3F7}}
.page{{max-width:1200px;margin:auto;padding:20px}}
.section{{background:white;padding:15px;margin-bottom:15px;border-radius:8px}}
</style>

</head>
<body>
{self._nav(activa)}
<div class="page">
<h1>{titulo}</h1>
{contenido}
</div>
</body>
</html>"""

    def _write(self, filename, html):
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    # ─────────────────────────────────────────────
    # INDEX
    # ─────────────────────────────────────────────

    def export_index(self):
        ing = self.datos.get("ingestion", {})

        contenido = f"""
        <div class="section">
            <p><b>Registros:</b> {ing.get("filas","—")}</p>
            <p><b>Columnas:</b> {ing.get("columnas","—")}</p>
        </div>
        """

        self._write("index.html", self._page("Resumen", "Resumen", contenido))

    # ─────────────────────────────────────────────
    # DIMENSIONALITY (OPTIMIZADO)
    # ─────────────────────────────────────────────

    def export_dimensionality(self):

        colores = ["#4FC3F7", "#1E2761", "#e91e63"]

        secciones = ""

        for nombre, key in [
            ("ACP","pca"),
            ("t-SNE","tsne"),
            ("UMAP","umap")
        ]:

            df = self.datos.get(key)
            if df is None:
                continue

            # 🔥 OPTIMIZACIÓN CLAVE
            df = self._safe_sample(df)

            clusters = sorted(df["cluster"].unique().tolist())[:5]

            trazas = []

            for i, c in enumerate(clusters):
                pts = df[df["cluster"] == c]

                trazas.append(f"""
                {{
                    type:'scatter',
                    mode:'markers',
                    name:'Cluster {c}',
                    x:{pts['dim1'].tolist()},
                    y:{pts['dim2'].tolist()},
                    marker:{{size:4,color:'{colores[i%3]}' }}
                }}
                """)

            cid = f"plot_{key}"

            secciones += f"""
            <div class="section">
                <h2>{nombre}</h2>
                <div id="{cid}" style="height:400px;"></div>
            </div>

            <script>
            Plotly.newPlot('{cid}', [{",".join(trazas)}],
            {{
                margin:{{l:40,r:10,t:10,b:40}},
                xaxis:{{title:'Dim 1'}},
                yaxis:{{title:'Dim 2'}}
            }});
            </script>
            """

        self._write(
            "dimensionality.html",
            self._page("Dimensionalidad", "Dimensionalidad", secciones)
        )

    # ─────────────────────────────────────────────
    # PLACEHOLDERS (para que no falle nada)
    # ─────────────────────────────────────────────

    def export_eda(self):
        self._write("eda.html", self._page("EDA","EDA","<p>EDA generado.</p>"))

    def export_apriori(self):
        self._write("apriori.html", self._page("Apriori","Apriori","<p>Apriori generado.</p>"))

    def export_eclat(self):
        self._write("eclat.html", self._page("ECLAT","ECLAT","<p>ECLAT generado.</p>"))

    def export_classification(self):
        self._write("classification.html", self._page("Clasificación","Clasificación","<p>Clasificación generada.</p>"))

    def export_lasso_ridge(self):
        self._write("lasso_ridge.html", self._page("Lasso","Lasso","<p>Lasso/Ridge generado.</p>"))

    # ─────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────

    def export_all(self):
        print("→ index")
        self.export_index()

        print("→ eda")
        self.export_eda()

        print("→ apriori")
        self.export_apriori()

        print("→ eclat")
        self.export_eclat()

        print("→ classification")
        self.export_classification()

        print("→ dimensionality")
        self.export_dimensionality()

        print("→ lasso")
        self.export_lasso_ridge()