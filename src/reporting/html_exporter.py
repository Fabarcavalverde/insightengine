# src/reporting/html_exporter.py

import os


class html_exporter:
    """
    Objetivo:
        Generar reporte HTML multipágina desde resultados en memoria.

    Uso:
        exporter = html_exporter(out_dir, eda_results, datos)
        exporter.export_all()
    """

    def __init__(self, out_dir: str, eda_results: dict, datos: dict):
        self.out_dir     = out_dir
        self.eda_results = eda_results
        self.datos       = datos
        os.makedirs(self.out_dir, exist_ok=True)

    def _nav(self, activa):
        pages = [("index.html","Resumen"),("eda.html","EDA"),("apriori.html","Apriori"),
                 ("eclat.html","ECLAT"),("classification.html","Clasificación"),
                 ("dimensionality.html","Dimensionalidad"),("lasso_ridge.html","Lasso / Ridge")]
        links = "".join(f'<a href="{h}" class="{"active" if l==activa else ""}">{l}</a>' for h,l in pages)
        return f'<nav><div class="nav-inner">{links}</div></nav>'

    def _page(self, titulo, activa, contenido):
        return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>InsightEngine — {titulo}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f4f6fb;color:#1a1a2e}}
nav{{background:#1E2761;padding:0 2rem;position:sticky;top:0;z-index:100}}
.nav-inner{{display:flex;align-items:center;gap:0;max-width:1200px;margin:0 auto}}
.nav-inner::before{{content:'InsightEngine';color:#4FC3F7;font-weight:700;font-size:1rem;margin-right:2rem;white-space:nowrap}}
nav a{{color:#CADCFC;text-decoration:none;padding:.9rem 1rem;font-size:.85rem;display:inline-block;transition:color .2s}}
nav a:hover,nav a.active{{color:#4FC3F7;border-bottom:2px solid #4FC3F7}}
.page{{max-width:1200px;margin:2rem auto;padding:0 1.5rem 4rem}}
h1{{font-size:1.6rem;font-weight:600;color:#1E2761;margin-bottom:.3rem}}
.subtitle{{font-size:.9rem;color:#666;margin-bottom:2rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:2rem}}
.card{{background:#1E2761;border-radius:10px;padding:1.2rem 1rem;text-align:center}}
.card-val{{font-size:2rem;font-weight:700;color:#4FC3F7}}
.card-label{{font-size:.78rem;color:#CADCFC;margin-top:.2rem}}
.section{{background:white;border-radius:12px;padding:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.07);margin-bottom:1.5rem}}
.section h2{{font-size:1rem;font-weight:600;color:#1E2761;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:2px solid #e8ecf5}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:#1E2761;color:white;padding:.6rem .8rem;text-align:left;font-weight:500}}
td{{padding:.55rem .8rem;border-bottom:1px solid #eef0f7}}
tr:last-child td{{border-bottom:none}}
tr:nth-child(even) td{{background:#f8f9ff}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:20px;font-size:.75rem;font-weight:600}}
.badge-green{{background:#e8f5e9;color:#2e7d32}}
.badge-blue{{background:#e3f2fd;color:#1565c0}}
.badge-red{{background:#fce4ec;color:#c62828}}
.rec-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.rec-card{{border-left:4px solid #4FC3F7;background:#f8f9ff;border-radius:0 8px 8px 0;padding:1rem 1.2rem}}
.rec-card h3{{font-size:.9rem;font-weight:600;color:#1E2761;margin-bottom:.3rem}}
.rec-rule{{font-size:.78rem;color:#5c6bc0;background:#e8eaf6;padding:.25rem .5rem;border-radius:4px;margin-bottom:.5rem;display:inline-block}}
.rec-card p{{font-size:.83rem;color:#444;line-height:1.5}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:700px){{.rec-grid,.two-col{{grid-template-columns:1fr}}}}
</style></head><body>
{self._nav(activa)}
<div class="page"><h1>{titulo}</h1>{contenido}</div>
</body></html>"""

    def _write(self, filename, html):
        with open(os.path.join(self.out_dir, filename), "w", encoding="utf-8") as f:
            f.write(html)

    def export_index(self):
        ing = self.datos.get("ingestion", {})
        apr = self.datos.get("apriori", {})
        clf = self.datos.get("clasificacion")
        lr  = self.datos.get("regularizacion", {})

        n_itemsets = len(apr.get("itemsets", [])) if isinstance(apr.get("itemsets"), object) and hasattr(apr.get("itemsets",""), "__len__") else "—"
        n_reglas   = len(apr.get("reglas", [])) if isinstance(apr.get("reglas"), object) and hasattr(apr.get("reglas",""), "__len__") else "—"
        try: n_itemsets = len(apr["itemsets"])
        except: pass
        try: n_reglas = len(apr["reglas"])
        except: pass

        mejor_pg = mejor_modelo = "—"
        try:
            mejor_modelo = clf["PG"].idxmax()
            mejor_pg     = f"{clf['PG'].max():.1%}"
        except: pass

        ridge_r2 = lr.get("ridge", {}).get("r2", "—")

        contenido = f"""
        <p class="subtitle">Resultados del pipeline completo</p>
        <div class="cards">
          <div class="card"><div class="card-val">{ing.get('filas','—')}</div><div class="card-label">Registros</div></div>
          <div class="card"><div class="card-val">{ing.get('columnas','—')}</div><div class="card-label">Variables</div></div>
          <div class="card"><div class="card-val">{n_itemsets}</div><div class="card-label">Itemsets frecuentes</div></div>
          <div class="card"><div class="card-val">{n_reglas}</div><div class="card-label">Reglas generadas</div></div>
          <div class="card"><div class="card-val">{mejor_pg}</div><div class="card-label">Mejor precisión ({mejor_modelo})</div></div>
          <div class="card"><div class="card-val">{ridge_r2}</div><div class="card-label">R² Ridge</div></div>
        </div>
        <div class="section"><h2>Navegación</h2>
        <table><tr><th>Página</th><th>Contenido</th></tr>
          <tr><td><a href="eda.html">EDA</a></td><td>Estadísticas descriptivas y distribuciones</td></tr>
          <tr><td><a href="apriori.html">Apriori</a></td><td>Itemsets frecuentes y reglas de asociación</td></tr>
          <tr><td><a href="eclat.html">ECLAT</a></td><td>Itemsets con formato vertical</td></tr>
          <tr><td><a href="classification.html">Clasificación</a></td><td>12 modelos supervisados</td></tr>
          <tr><td><a href="dimensionality.html">Dimensionalidad</a></td><td>ACP, t-SNE y UMAP</td></tr>
          <tr><td><a href="lasso_ridge.html">Lasso / Ridge</a></td><td>Regularización y recomendaciones</td></tr>
        </table></div>"""
        self._write("index.html", self._page("Resumen", "Resumen", contenido))

    def export_eda(self):
        r        = self.eda_results
        stats    = r.get("descriptive_stats", {})
        missing  = r.get("missing", {})
        outliers = r.get("outliers", {})
        dupes    = r.get("duplicates", 0)

        # estadísticas numéricas
        num_stats = stats.get("numeric_stats", {})
        num_cols  = stats.get("numeric_cols", [])
        filas_num = ""
        if num_cols and num_stats:
            for col in num_cols:
                try:
                    mean = round(num_stats.get("mean",{}).get(col,0), 2)
                    std  = round(num_stats.get("std",{}).get(col,0), 2)
                    mn   = round(num_stats.get("min",{}).get(col,0), 2)
                    mx   = round(num_stats.get("max",{}).get(col,0), 2)
                    p50  = round(num_stats.get("50%",{}).get(col,0), 2)
                    filas_num += f"<tr><td>{col}</td><td>{mean}</td><td>{std}</td><td>{mn}</td><td>{p50}</td><td>{mx}</td></tr>"
                except: pass

        # outliers
        filas_out = ""
        for col, info in outliers.items():
            n   = info.get("n_outliers", 0)
            pct = info.get("pct", 0)
            badge = "badge-red" if pct > 5 else "badge-green"
            filas_out += f"<tr><td>{col}</td><td>{n}</td><td><span class='badge {badge}'>{pct}%</span></td><td>{round(info.get('lower_bound',0),2)}</td><td>{round(info.get('upper_bound',0),2)}</td></tr>"

        # variables categóricas
        cat_info = stats.get("categorical_info", {})
        filas_cat = ""
        try:
            cols_cat = cat_info.get("Columna", {})
            uniq_cat = cat_info.get("Valores únicos", {})
            freq_cat = cat_info.get("Valor más frecuente", {})
            pct_cat  = cat_info.get("% Frecuencia", {})
            for k in cols_cat:
                filas_cat += f"<tr><td>{cols_cat[k]}</td><td>{uniq_cat.get(k,'—')}</td><td>{freq_cat.get(k,'—')}</td><td>{pct_cat.get(k,'—')}%</td></tr>"
        except: pass

        contenido = f"""
        <p class="subtitle">Análisis exploratorio — {r.get('variable_types',{}).get('numeric',['']).__len__()} variables numéricas · {r.get('variable_types',{}).get('categorical',['']).__len__()} categóricas</p>
        <div class="cards">
          <div class="card"><div class="card-val">{dupes}</div><div class="card-label">Duplicados</div></div>
          <div class="card"><div class="card-val">{len(outliers)}</div><div class="card-label">Variables con outliers</div></div>
          <div class="card"><div class="card-val">{len(missing.get('Columna',{}))}</div><div class="card-label">Cols con nulos</div></div>
        </div>
        <div class="two-col">
          <div class="section"><h2>Estadísticas numéricas</h2>
            <table><tr><th>Variable</th><th>Media</th><th>Std</th><th>Min</th><th>Mediana</th><th>Max</th></tr>
            {filas_num}</table>
          </div>
          <div class="section"><h2>Outliers (método IQR)</h2>
            <table><tr><th>Variable</th><th>N</th><th>%</th><th>Límite inf.</th><th>Límite sup.</th></tr>
            {filas_out}</table>
          </div>
        </div>
        <div class="section"><h2>Variables categóricas</h2>
          <table><tr><th>Variable</th><th>Valores únicos</th><th>Más frecuente</th><th>Frecuencia</th></tr>
          {filas_cat}</table>
        </div>"""
        self._write("eda.html", self._page("EDA", "EDA", contenido))

    def export_apriori(self):
        apr      = self.datos.get("apriori", {})
        itemsets = apr.get("itemsets")
        reglas   = apr.get("reglas")
        if itemsets is None or reglas is None:
            self._write("apriori.html", self._page("Apriori","Apriori","<p>Sin datos.</p>"))
            return

        top_items  = itemsets.sort_values("support", ascending=False).head(20)
        labels_i   = [str(x)[:55] for x in top_items["itemsets"].tolist()]
        values_i   = top_items["support"].round(4).tolist()
        top_reglas = reglas.sort_values("lift", ascending=False).head(20)
        labels_r   = [f"{str(r['antecedents'])[:30]} → {str(r['consequents'])[:20]}" for _,r in top_reglas.iterrows()]
        values_r   = top_reglas["lift"].round(4).tolist()
        conf_vals  = reglas["confidence"].round(4).tolist()
        lift_vals  = reglas["lift"].round(4).tolist()

        filas = "".join(f"<tr><td>{str(r['antecedents'])[:60]}</td><td>{str(r['consequents'])[:40]}</td><td>{round(r['support'],3)}</td><td>{round(r['confidence'],3)}</td><td>{round(r['lift'],3)}</td></tr>" for _,r in top_reglas.head(15).iterrows())

        contenido = f"""
        <p class="subtitle">{len(itemsets)} itemsets · {len(reglas)} reglas · soporte mín. 0.30 · confianza mín. 0.60</p>
        <div class="cards">
          <div class="card"><div class="card-val">{len(itemsets)}</div><div class="card-label">Itemsets</div></div>
          <div class="card"><div class="card-val">{len(reglas)}</div><div class="card-label">Reglas</div></div>
          <div class="card"><div class="card-val">0.30</div><div class="card-label">Soporte mín.</div></div>
          <div class="card"><div class="card-val">0.60</div><div class="card-label">Confianza mín.</div></div>
        </div>
        <div class="section"><h2>Top 20 itemsets por soporte</h2><div id="ci" style="height:420px"></div></div>
        <div class="section"><h2>Confianza vs Lift</h2><div id="cs" style="height:360px"></div></div>
        <div class="section"><h2>Top reglas por lift</h2><div id="cr" style="height:420px"></div></div>
        <div class="section"><h2>Tabla de reglas</h2>
          <table><tr><th>Antecedente</th><th>Consecuente</th><th>Soporte</th><th>Confianza</th><th>Lift</th></tr>{filas}</table>
        </div>
        <script>
          Plotly.newPlot('ci',[{{type:'bar',orientation:'h',x:{values_i},y:{labels_i},marker:{{color:'#4FC3F7'}}}}],
            {{margin:{{l:300,r:20,t:10,b:40}},xaxis:{{title:'Soporte'}},plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},{{responsive:true}});
          Plotly.newPlot('cs',[{{type:'scatter',mode:'markers',x:{conf_vals},y:{lift_vals},marker:{{color:'#1E2761',size:5,opacity:0.6}}}}],
            {{xaxis:{{title:'Confianza'}},yaxis:{{title:'Lift'}},margin:{{l:60,r:20,t:10,b:50}},plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},{{responsive:true}});
          Plotly.newPlot('cr',[{{type:'bar',orientation:'h',x:{values_r},y:{labels_r},marker:{{color:'#1E2761'}}}}],
            {{margin:{{l:380,r:20,t:10,b:40}},xaxis:{{title:'Lift'}},plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},{{responsive:true}});
        </script>"""
        self._write("apriori.html", self._page("Apriori","Apriori",contenido))

    def export_eclat(self):
        ec       = self.datos.get("eclat", {})
        itemsets = ec.get("itemsets")
        if itemsets is None:
            self._write("eclat.html", self._page("ECLAT","ECLAT","<p>Sin datos.</p>"))
            return

        top    = itemsets.sort_values("support", ascending=False).head(20)
        labels = [str(x)[:55] for x in top["itemsets"].tolist()]
        values = top["support"].tolist()
        filas  = "".join(f"<tr><td>{str(r['itemsets'])[:80]}</td><td>{r['support']}</td></tr>" for _,r in top.iterrows())

        contenido = f"""
        <p class="subtitle">{len(itemsets)} itemsets — formato vertical (tidsets)</p>
        <div class="cards">
          <div class="card"><div class="card-val">{len(itemsets)}</div><div class="card-label">Itemsets</div></div>
          <div class="card"><div class="card-val">300</div><div class="card-label">Soporte mín. (abs.)</div></div>
        </div>
        <div class="section"><h2>Top 20 itemsets</h2><div id="ce" style="height:440px"></div></div>
        <div class="section"><h2>Tabla</h2>
          <table><tr><th>Itemset</th><th>Soporte</th></tr>{filas}</table>
        </div>
        <script>
          Plotly.newPlot('ce',[{{type:'bar',orientation:'h',x:{values},y:{labels},marker:{{color:'#5c6bc0'}}}}],
            {{margin:{{l:320,r:20,t:10,b:40}},xaxis:{{title:'Soporte (nº transacciones)'}},plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},{{responsive:true}});
        </script>"""
        self._write("eclat.html", self._page("ECLAT","ECLAT",contenido))

    def export_classification(self):
        clf = self.datos.get("clasificacion")
        if clf is None:
            self._write("classification.html", self._page("Clasificación","Clasificación","<p>Sin datos.</p>"))
            return

        modelos = clf.index.tolist()
        pgs     = [round(v,4) for v in clf["PG"].tolist()]
        colores = ["#4FC3F7" if v==max(pgs) else "#1E2761" for v in pgs]
        clases  = [c for c in clf.columns if c not in ["PG","Error"]]
        filas   = ""
        for m in modelos:
            pg    = round(clf.loc[m,"PG"],4)
            err   = round(clf.loc[m,"Error"],4)
            badge = "badge-green" if pg==max(pgs) else "badge-blue"
            extra = "".join(f"<td>{round(clf.loc[m,c],4)}</td>" for c in clases)
            filas += f"<tr><td>{m}</td><td><span class='badge {badge}'>{pg}</span></td><td>{err}</td>{extra}</tr>"

        contenido = f"""
        <p class="subtitle">12 modelos — train 70% / test 30%</p>
        <div class="section"><h2>Precisión global por modelo</h2><div id="cc" style="height:420px"></div></div>
        <div class="section"><h2>Tabla comparativa</h2>
          <table><tr><th>Modelo</th><th>PG</th><th>Error</th>{"".join(f"<th>{c}</th>" for c in clases)}</tr>{filas}</table>
        </div>
        <script>
          Plotly.newPlot('cc',[{{type:'bar',x:{modelos},y:{pgs},marker:{{color:{colores}}}}}],
            {{yaxis:{{title:'Precisión Global',range:[0.5,1.0]}},margin:{{l:60,r:20,t:10,b:100}},plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},{{responsive:true}});
        </script>"""
        self._write("classification.html", self._page("Clasificación","Clasificación",contenido))

    def export_dimensionality(self):
        colores = ["#4FC3F7","#1E2761","#e91e63"]
        secciones = ""
        for nombre, key in [("ACP","pca"),("t-SNE","tsne"),("UMAP","umap")]:
            df = self.datos.get(key)
            if df is None: continue
            clusters = sorted(df["cluster"].unique().tolist())
            trazas   = []
            for i,c in enumerate(clusters):
                pts = df[df["cluster"]==c]
                trazas.append(f"{{type:'scatter',mode:'markers',name:'Cluster {c}',x:{pts['dim1'].round(4).tolist()},y:{pts['dim2'].round(4).tolist()},marker:{{color:'{colores[i%3]}',size:5,opacity:0.7}}}}")
            cid = f"cd{key}"
            secciones += f"""<div class="section"><h2>{nombre}</h2><div id="{cid}" style="height:420px"></div>
              <script>Plotly.newPlot('{cid}',[{",".join(trazas)}],
                {{xaxis:{{title:'Dim 1'}},yaxis:{{title:'Dim 2'}},margin:{{l:60,r:20,t:10,b:50}},plot_bgcolor:'#fff',paper_bgcolor:'#fff',legend:{{x:0.8,y:1}}}},{{responsive:true}});
              </script></div>"""

        contenido = f'<p class="subtitle">Reducción a 2 dimensiones con K-Medias (3 clusters)</p>{secciones}'
        self._write("dimensionality.html", self._page("Dimensionalidad","Dimensionalidad",contenido))

    def export_lasso_ridge(self):
        lr    = self.datos.get("regularizacion", {})
        lasso = lr.get("lasso", {})
        ridge = lr.get("ridge", {})

        recs = [
            {"color":"#1565C0","title":"Aprobar sin cuenta bancaria",
             "regla":"sin_cuenta + vivienda_propia → lift 1.082",
             "body":"Sin cuenta corriente no implica alto riesgo si hay vivienda propia. Ofrecer crédito sin exigir cuenta previa."},
            {"color":"#1B5E20","title":"Perfil preferencial — empleado estable",
             "regla":"empleado_calificado + sin_deudores → lift 1.020",
             "body":"Empleados calificados sin deudores representan el menor riesgo. Tasas preferenciales automáticas."},
            {"color":"#4A148C","title":"Segmento trabajador extranjero",
             "regla":"trabajador_extranjero + sin_otros_planes (96%)",
             "body":"Segmento mayoritario, confiable y subatendido. Oportunidad de productos especializados."},
            {"color":"#B71C1C","title":"Alerta de riesgo combinado",
             "regla":"saldo_negativo + pagos_atrasados",
             "body":"Requieren revisión manual antes de aprobación, sin excepción."},
        ]
        cards = "".join(f"""<div class="rec-card" style="border-left-color:{r['color']}">
          <h3>{r['title']}</h3><span class="rec-rule">{r['regla']}</span><p>{r['body']}</p></div>""" for r in recs)

        contenido = f"""
        <p class="subtitle">Predicción de monto de crédito (credit_amount)</p>
        <div class="cards">
          <div class="card"><div class="card-val">{lasso.get('r2','—')}</div><div class="card-label">R² Lasso</div></div>
          <div class="card"><div class="card-val">{lasso.get('mse','—')}</div><div class="card-label">MSE Lasso</div></div>
          <div class="card"><div class="card-val">{ridge.get('r2','—')}</div><div class="card-label">R² Ridge</div></div>
          <div class="card"><div class="card-val">{ridge.get('mse','—')}</div><div class="card-label">MSE Ridge</div></div>
        </div>
        <div class="section"><h2>Recomendaciones de negocio</h2>
          <p class="subtitle" style="margin-bottom:1rem">Basadas en reglas con lift &gt; 1.0 y confianza &gt; 0.60</p>
          <div class="rec-grid">{cards}</div>
        </div>"""
        self._write("lasso_ridge.html", self._page("Lasso / Ridge","Lasso / Ridge",contenido))

    def export_all(self):
        """
        Objetivo:
            Generar todas las páginas del reporte HTML.

        Parámetros:
            None

        Retorna:
            None
        """
        print("  → index.html");          self.export_index()
        print("  → eda.html");            self.export_eda()
        print("  → apriori.html");        self.export_apriori()
        print("  → eclat.html");          self.export_eclat()
        print("  → classification.html"); self.export_classification()
        print("  → dimensionality.html"); self.export_dimensionality()
        print("  → lasso_ridge.html");    self.export_lasso_ridge()