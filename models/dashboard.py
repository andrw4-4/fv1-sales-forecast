# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Vivo Balanced Bites",
    layout="wide",
    page_icon="🥗",
    initial_sidebar_state="expanded",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card { background:#f8f9fa; border-radius:10px; padding:16px 20px;
                   border-left:4px solid #4CAF50; margin-bottom:8px; }
    .metric-label { font-size:13px; color:#666; margin-bottom:4px; }
    .metric-value { font-size:24px; font-weight:700; color:#1a1a1a; }
    .metric-delta { font-size:12px; color:#4CAF50; }
    h2 { color: #2d5a27; }
    .stPlotlyChart { border-radius: 8px; }
    div[data-testid="metric-container"] { background:#f8f9fa; border-radius:8px;
        padding:12px; border-left:4px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_COM = ROOT / "data" / "Compras"
DATA_FAC = ROOT / "data" / "Facturas_lugar"
DATA_PRO = ROOT / "data" / "Productos"

VERDE   = "#4CAF50"
NARANJA = "#FF7043"
AZUL    = "#2196F3"
AMARILLO= "#FFC107"
MORADO  = "#9C27B0"
TURQ    = "#00BCD4"

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def cargar_ventas():
    v = pd.read_csv(DATA_RAW / "ventas.csv", encoding="latin-1", header=0)
    v.columns = ["Consecutivo","Fecha","Tipo_reg","Tipo_clas","Codigo","Nombre",
                 "Vendedor","Cantidad","Precio","Impuesto","Total",
                 "Forma_pago","Num_comp","Establecimiento"]
    v["Fecha"] = pd.to_datetime(v["Fecha"], errors="coerce")
    v = v.dropna(subset=["Fecha"])
    v["Mes"]      = v["Fecha"].dt.to_period("M").astype(str)
    v["DiaSemana"]= v["Fecha"].dt.day_name()
    v["Semana"]   = v["Fecha"].dt.isocalendar().week.astype(int)
    v["Año"]      = v["Fecha"].dt.year

    p = pd.read_csv(DATA_RAW / "info_productos.csv", encoding="latin-1")
    p.columns = ["Nombre","Trans","Cant_total","Precio_prom","Ingresos_total","Categoria"]
    v = v.merge(p[["Nombre","Categoria"]], on="Nombre", how="left")
    v["Categoria"] = v["Categoria"].fillna("Sin categoria")
    return v

@st.cache_data
def cargar_compras_excel():
    """Lee los 4 archivos xlsx de Compras y devuelve DataFrame mensual."""
    dfs = []
    for f in sorted(DATA_COM.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, header=6)
            # Renombrar por posición para evitar encoding
            cols = df.columns.tolist()
            rename = {}
            for i, c in enumerate(cols):
                s = str(c).strip()
                if i == 0: rename[c] = "Tipo"
                elif i == 3: rename[c] = "Fecha"
                elif i == 6: rename[c] = "Proveedor"
                elif i == 8: rename[c] = "Valor"
            df = df.rename(columns=rename)
            if "Fecha" in df.columns and "Valor" in df.columns:
                df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
                df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0)
                dfs.append(df[["Tipo","Fecha","Proveedor","Valor"]].dropna(subset=["Fecha"]))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["Tipo","Fecha","Proveedor","Valor","Mes"])
    res = pd.concat(dfs, ignore_index=True)
    res["Mes"] = res["Fecha"].dt.to_period("M").astype(str)
    return res

def _parse_cop(s):
    """Convierte formato COP colombiano '$ 46.000,00' a float."""
    if pd.isna(s): return 0.0
    return float(str(s).replace("$", "").replace(".", "").replace(",", ".").strip() or 0)

@st.cache_data
def cargar_facturas_excel():
    """Lee Facturas_lugar/*.xlsx — fuente confiable de ingresos reales por mes."""
    dfs = []
    for f in sorted(DATA_FAC.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, header=6)
            df = df.rename(columns={
                df.columns[1]: "Fecha",
                df.columns[7]: "Vendedor",
                df.columns[8]: "Efectivo",
                df.columns[9]: "Tarjetas",
                df.columns[13]: "Total"
            })
            df["Local"] = f.stem.split("-")[0]
            df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
            df["Total"] = df["Total"].apply(_parse_cop)
            # Eliminar filas sin fecha (filas de totales al final del Excel)
            df = df.dropna(subset=["Fecha"])
            dfs.append(df[["Fecha", "Vendedor", "Total", "Local"]])
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["Fecha", "Vendedor", "Total", "Local", "Mes"])
    res = pd.concat(dfs, ignore_index=True)
    res["Mes"] = res["Fecha"].dt.to_period("M").astype(str)
    return res

@st.cache_data
def cargar_productos_excel():
    """Lee todos los xlsx de Productos para historial de ventas completo."""
    dfs = []
    for f in sorted(DATA_PRO.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, header=7)
            cols = df.columns.tolist()
            rename = {}
            for i, c in enumerate(cols):
                if i == 3:  rename[c] = "Fecha"
                elif i == 7: rename[c] = "Nombre"
                elif i == 9: rename[c] = "Cantidad"
                elif i ==10: rename[c] = "Precio"
                elif i ==12: rename[c] = "Total"
                elif i ==13: rename[c] = "Forma_pago"
            df = df.rename(columns=rename)
            keep = [c for c in ["Fecha","Nombre","Cantidad","Precio","Total","Forma_pago"] if c in df.columns]
            if "Fecha" in keep and "Total" in keep:
                df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
                df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)
                df["Cantidad"] = pd.to_numeric(df.get("Cantidad", 0), errors="coerce").fillna(0)
                dfs.append(df[keep].dropna(subset=["Fecha"]))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    res = pd.concat(dfs, ignore_index=True)
    res["Mes"] = res["Fecha"].dt.to_period("M").astype(str)
    return res

with st.spinner("Cargando datos..."):
    ventas   = cargar_ventas()
    compras_x= cargar_compras_excel()
    facturas = cargar_facturas_excel()
    productos_hist = cargar_productos_excel()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🥗 Vivo Balanced Bites")
    st.divider()
    st.markdown("**Filtros**")

    meses = sorted(ventas["Mes"].dropna().unique())
    mes_ini, mes_fin = st.select_slider(
        "Periodo", options=meses, value=(meses[0], meses[-1])
    )

    estabs = ["Todos"] + sorted(ventas["Establecimiento"].dropna().unique().tolist())
    estab  = st.selectbox("Establecimiento", estabs)

    cats = ["Todas"] + sorted(ventas["Categoria"].dropna().unique().tolist())
    cat  = st.selectbox("Categoria BCG", cats)

    st.divider()
    st.caption("Datos desde Ago 2024 · iPPO PICE")

# Aplicar filtros
mask = (ventas["Mes"] >= mes_ini) & (ventas["Mes"] <= mes_fin)
if estab != "Todos":  mask &= ventas["Establecimiento"] == estab
if cat   != "Todas":  mask &= ventas["Categoria"]       == cat
df = ventas[mask].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🥗 Vivo Balanced Bites")
st.markdown(f"**Panel de análisis de ventas** · {mes_ini} → {mes_fin} · {len(df):,} registros")
st.divider()

# ── KPIs ───────────────────────────────────────────────────────────────────────
ingresos_total  = df["Total"].sum()
facturas_total  = df["Consecutivo"].nunique()
unidades_total  = df["Cantidad"].sum()
ticket_prom     = df.groupby("Consecutivo")["Total"].sum().mean()
productos_uniq  = df["Nombre"].nunique()

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("💰 Ingresos totales",   f"${ingresos_total/1_000_000:.1f}M")
k2.metric("🧾 Facturas",           f"{facturas_total:,}")
k3.metric("🍱 Unidades vendidas",  f"{int(unidades_total):,}")
k4.metric("🎫 Ticket promedio",    f"${ticket_prom:,.0f}")
k5.metric("📦 Productos activos",  f"{productos_uniq:,}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: HISTORIAL INGRESOS vs GASTOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Historial de Ingresos vs Gastos")

# Ingresos mensuales — usar Facturas_lugar como fuente confiable (ventas.csv es incompleto en algunos meses)
if not facturas.empty:
    ing_fac = facturas.groupby("Mes")["Total"].sum().reset_index()
    ing_fac.columns = ["Mes","Ingresos"]
    # Completar con ventas.csv para meses no cubiertos por Facturas_lugar
    ing_csv = ventas.groupby("Mes")["Total"].sum().reset_index()
    ing_csv.columns = ["Mes","Ingresos_csv"]
    ing_mes = ing_fac.merge(ing_csv, on="Mes", how="outer")
    ing_mes["Ingresos"] = ing_mes["Ingresos"].fillna(ing_mes["Ingresos_csv"])
    ing_mes = ing_mes[["Mes","Ingresos"]].fillna(0)
else:
    ing_mes = ventas.groupby("Mes")["Total"].sum().reset_index()
    ing_mes.columns = ["Mes","Ingresos"]

# Gastos mensuales de compras_x (Excel 2025) + compras.csv (2024)
gasto_csv = pd.read_csv(DATA_RAW / "compras.csv", encoding="latin-1", header=0)
gasto_csv.columns = ["Consecutivo","Factura","ID","Proveedor","Fecha_c","Fecha_m",
                     "Fecha","Contacto","Tipo_reg","Tipo_clas","Codigo","Nombre",
                     "Cantidad","Precio","Total","Forma_pago","Fecha_v","Periodo"]
gasto_csv["Mes"] = gasto_csv["Periodo"].astype(str).str[:7]
gasto_csv_mes = gasto_csv.groupby("Mes")["Total"].sum().reset_index()
gasto_csv_mes.columns = ["Mes","Gastos"]

# Gastos de los Excel
if not compras_x.empty:
    gasto_x_mes = compras_x.groupby("Mes")["Valor"].sum().reset_index()
    gasto_x_mes.columns = ["Mes","Gastos"]
    gastos_mes = pd.concat([gasto_csv_mes, gasto_x_mes]).groupby("Mes")["Gastos"].sum().reset_index()
else:
    gastos_mes = gasto_csv_mes

hist = ing_mes.merge(gastos_mes, on="Mes", how="outer").fillna(0).sort_values("Mes")
hist["Margen"] = hist["Ingresos"] - hist["Gastos"]
hist["Margen_pct"] = np.where(hist["Ingresos"]>0,
                               hist["Margen"]/hist["Ingresos"]*100, 0)

fig_hist = go.Figure()
fig_hist.add_bar(x=hist["Mes"], y=hist["Ingresos"]/1e6,
                 name="Ingresos", marker_color=VERDE, opacity=0.85)
fig_hist.add_bar(x=hist["Mes"], y=hist["Gastos"]/1e6,
                 name="Gastos", marker_color=NARANJA, opacity=0.85)
fig_hist.add_scatter(x=hist["Mes"], y=hist["Margen"]/1e6,
                     name="Margen bruto", mode="lines+markers",
                     line=dict(color=AZUL, width=2), marker=dict(size=6))
fig_hist.update_layout(
    barmode="group", height=380,
    yaxis_title="Millones COP",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=40, l=60, r=20),
    hovermode="x unified"
)
fig_hist.update_xaxes(tickangle=-45)
st.plotly_chart(fig_hist, use_container_width=True)

# Tabla resumen margen
col_t1, col_t2, col_t3 = st.columns(3)
col_t1.metric("Total ingresos (periodo)", f"${hist['Ingresos'].sum()/1e6:.1f}M")
col_t2.metric("Total gastos (periodo)",   f"${hist['Gastos'].sum()/1e6:.1f}M")
col_t3.metric("Margen bruto promedio",    f"{hist[hist['Ingresos']>0]['Margen_pct'].mean():.1f}%")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: VENTAS MENSUALES + MATRIZ BCG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📈 Tendencia de Ventas")

c1, c2 = st.columns([2, 1])

with c1:
    metrica = st.radio("Ver por", ["Ingresos (COP)", "Unidades", "Facturas"],
                       horizontal=True, key="r1")
    col_m  = {"Ingresos (COP)":"Total","Unidades":"Cantidad","Facturas":"Consecutivo"}
    agg_m  = {"Ingresos (COP)":"sum","Unidades":"sum","Facturas":"nunique"}
    vm = df.groupby("Mes")[col_m[metrica]].agg(agg_m[metrica]).reset_index()
    vm.columns = ["Mes","Val"]

    # Línea de tendencia
    if len(vm) > 2:
        z = np.polyfit(range(len(vm)), vm["Val"], 1)
        trend = np.poly1d(z)(range(len(vm)))
    else:
        trend = vm["Val"].values

    fig_trend = go.Figure()
    fig_trend.add_bar(x=vm["Mes"], y=vm["Val"], name=metrica,
                      marker_color=VERDE, opacity=0.8)
    fig_trend.add_scatter(x=vm["Mes"], y=trend, name="Tendencia",
                          mode="lines", line=dict(color=NARANJA, width=2, dash="dash"))
    fig_trend.update_layout(height=320, margin=dict(t=10,b=40,l=60,r=10),
                             yaxis_title=metrica, hovermode="x unified",
                             legend=dict(orientation="h",y=1.05,x=0))
    fig_trend.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.markdown("**Matriz BCG**")
    bcg = df.groupby("Categoria")["Total"].sum().reset_index()
    COLORES_BCG = {
        "ESTRELLA (Alto Vol / Alto $)":    "#FFD700",
        "VACA (Alto Vol / Bajo $)":        "#4CAF50",
        "INTERROGANTE (Bajo Vol / Alto $)":"#2196F3",
        "PERRO (Bajo Vol / Bajo $)":       "#F44336",
        "Sin categoria":                    "#BDBDBD"
    }
    fig_bcg = px.pie(bcg, values="Total", names="Categoria",
                     color="Categoria", color_discrete_map=COLORES_BCG, hole=0.45)
    fig_bcg.update_layout(height=320, margin=dict(t=5,b=5,l=0,r=0),
                           legend=dict(font=dict(size=10), x=0, y=0))
    fig_bcg.update_traces(textinfo="percent",
                           hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
    st.plotly_chart(fig_bcg, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: TOP PRODUCTOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏆 Análisis de Productos")

n_top = st.slider("Número de productos a mostrar", 5, 25, 12, key="sl1")
top = (df.groupby(["Nombre","Categoria"])
       .agg(Ingresos=("Total","sum"), Unidades=("Cantidad","sum"),
            Facturas=("Consecutivo","nunique"), Precio_prom=("Precio","mean"))
       .reset_index().sort_values("Ingresos", ascending=False).head(n_top))

t1, t2 = st.columns(2)
with t1:
    fig_ti = px.bar(
        top.sort_values("Ingresos"), x="Ingresos", y="Nombre",
        color="Categoria", color_discrete_map=COLORES_BCG,
        orientation="h", title="Por ingresos (COP)",
        text=top.sort_values("Ingresos")["Ingresos"].apply(lambda x: f"${x/1e6:.2f}M"),
        labels={"Nombre":"","Ingresos":"COP"}
    )
    fig_ti.update_traces(textposition="outside", textfont_size=10)
    fig_ti.update_layout(height=420, margin=dict(t=40,b=10,l=10,r=80),
                          showlegend=False,
                          xaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_ti, use_container_width=True)

with t2:
    fig_tu = px.bar(
        top.sort_values("Unidades"), x="Unidades", y="Nombre",
        color="Categoria", color_discrete_map=COLORES_BCG,
        orientation="h", title="Por unidades vendidas",
        text=top.sort_values("Unidades")["Unidades"].astype(int),
        labels={"Nombre":"","Unidades":"Unidades"}
    )
    fig_tu.update_traces(textposition="outside", textfont_size=10)
    fig_tu.update_layout(height=420, margin=dict(t=40,b=10,l=10,r=60),
                          showlegend=False)
    st.plotly_chart(fig_tu, use_container_width=True)

# Concentración de ingresos (Pareto)
st.markdown("**Concentración de ingresos — Curva de Pareto**")
pareto = (df.groupby("Nombre")["Total"].sum()
          .sort_values(ascending=False).reset_index())
pareto["Acum_pct"] = pareto["Total"].cumsum() / pareto["Total"].sum() * 100
pareto["Rank"] = range(1, len(pareto)+1)
n80 = (pareto["Acum_pct"] <= 80).sum() + 1

fig_par = go.Figure()
fig_par.add_bar(x=pareto["Rank"], y=pareto["Total"]/1e3,
                name="Ingresos", marker_color=VERDE, opacity=0.7)
fig_par.add_scatter(x=pareto["Rank"], y=pareto["Acum_pct"],
                    name="% Acumulado", yaxis="y2",
                    line=dict(color=NARANJA, width=2))
fig_par.add_vline(x=n80, line_dash="dash", line_color="gray",
                  annotation_text=f"{n80} productos = 80% ingresos",
                  annotation_position="top right")
fig_par.update_layout(
    height=280,
    yaxis=dict(title="Miles COP"),
    yaxis2=dict(title="% Acumulado", overlaying="y", side="right",
                range=[0,105], ticksuffix="%"),
    margin=dict(t=10,b=40,l=60,r=60),
    hovermode="x unified",
    legend=dict(orientation="h",y=1.05,x=0)
)
st.plotly_chart(fig_par, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: PATRONES TEMPORALES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🗓️ Patrones de Comportamiento")

ORDEN_DIAS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
NOMBRES    = {"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miercoles",
              "Thursday":"Jueves","Friday":"Viernes","Saturday":"Sabado","Sunday":"Domingo"}

p1, p2 = st.columns(2)

with p1:
    vdia = (df.groupby("DiaSemana")
            .agg(Ingresos=("Total","sum"), Facturas=("Consecutivo","nunique"))
            .reindex(ORDEN_DIAS).reset_index())
    vdia["Dia"] = vdia["DiaSemana"].map(NOMBRES)
    vdia["TicketProm"] = vdia["Ingresos"] / vdia["Facturas"].replace(0, np.nan)

    fig_dia = go.Figure()
    fig_dia.add_bar(x=vdia["Dia"], y=vdia["Ingresos"]/1e6,
                    name="Ingresos (M)", marker_color=VERDE, opacity=0.85)
    fig_dia.update_layout(
        title="Ingresos por dia de semana",
        height=300, margin=dict(t=40,b=10,l=60,r=10),
        yaxis_title="Millones COP"
    )
    st.plotly_chart(fig_dia, use_container_width=True)
    st.caption("⚠️ Sabado: operacion minima real (restaurante universitario — Sabado casi no hay estudiantes). Domingo: pedidos o catering.")

with p2:
    # Heatmap semana vs mes
    pivot = (df.groupby(["DiaSemana","Mes"])["Total"]
             .sum().unstack(fill_value=0))
    pivot = pivot.reindex([d for d in ORDEN_DIAS if d in pivot.index])
    pivot.index = [NOMBRES[d] for d in pivot.index]

    fig_heat = px.imshow(
        pivot/1e3, aspect="auto",
        color_continuous_scale="Greens",
        title="Heatmap ingresos: dia vs mes (miles COP)",
        labels=dict(color="Miles COP")
    )
    fig_heat.update_layout(height=300, margin=dict(t=40,b=10,l=80,r=10))
    fig_heat.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    st.plotly_chart(fig_heat, use_container_width=True)

# Ingresos por establecimiento
st.markdown("**Ingresos por establecimiento a lo largo del tiempo**")
vlocal = ventas[mask].groupby(["Mes","Establecimiento"])["Total"].sum().reset_index()
fig_loc = px.line(vlocal, x="Mes", y="Total", color="Establecimiento",
                  markers=True, color_discrete_sequence=[VERDE, NARANJA],
                  labels={"Total":"Ingresos COP","Mes":"","Establecimiento":"Local"})
fig_loc.update_layout(height=260, margin=dict(t=10,b=40,l=60,r=10),
                       yaxis_tickformat="$,.0f", hovermode="x unified")
fig_loc.update_xaxes(tickangle=-45)
st.plotly_chart(fig_loc, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: ANÁLISIS DE TICKET Y PRECIO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 💳 Análisis de Ticket y Precios")

tickets = df.groupby("Consecutivo")["Total"].sum().reset_index()["Total"]

ta, tb, tc = st.columns(3)

with ta:
    fig_hist_tick = px.histogram(
        tickets[tickets > 0], nbins=40,
        title="Distribucion del ticket por factura",
        labels={"value":"Total factura (COP)","count":"Facturas"},
        color_discrete_sequence=[VERDE]
    )
    fig_hist_tick.update_layout(height=280, margin=dict(t=40,b=40,l=60,r=10),
                                  showlegend=False,
                                  xaxis_tickformat="$,.0f")
    st.plotly_chart(fig_hist_tick, use_container_width=True)

with tb:
    ticket_mes = (df.groupby(["Mes","Consecutivo"])["Total"].sum()
                  .reset_index().groupby("Mes")["Total"].mean().reset_index())
    ticket_mes.columns = ["Mes","Ticket_prom"]
    fig_tick_mes = px.line(ticket_mes, x="Mes", y="Ticket_prom",
                            markers=True, title="Ticket promedio por mes",
                            labels={"Ticket_prom":"COP","Mes":""},
                            color_discrete_sequence=[MORADO])
    fig_tick_mes.update_layout(height=280, margin=dict(t=40,b=40,l=60,r=10),
                                 yaxis_tickformat="$,.0f")
    fig_tick_mes.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_tick_mes, use_container_width=True)

with tc:
    # Distribución precio unitario por categoria
    precio_cat = df[df["Precio"] > 0].groupby("Categoria")["Precio"].median().reset_index()
    precio_cat.columns = ["Categoria","Precio_mediano"]
    precio_cat = precio_cat.sort_values("Precio_mediano", ascending=False)
    fig_precio = px.bar(precio_cat, x="Precio_mediano", y="Categoria",
                        orientation="h", title="Precio mediano por categoria",
                        color="Categoria", color_discrete_map=COLORES_BCG,
                        labels={"Precio_mediano":"COP","Categoria":""},
                        text=precio_cat["Precio_mediano"].apply(lambda x: f"${x:,.0f}"))
    fig_precio.update_traces(textposition="outside", textfont_size=10)
    fig_precio.update_layout(height=280, margin=dict(t=40,b=10,l=10,r=80),
                               showlegend=False, xaxis_tickformat="$,.0f")
    st.plotly_chart(fig_precio, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: FORMAS DE PAGO + VENDEDORES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 💰 Pagos y Vendedores")

pa, pb = st.columns(2)

with pa:
    pago = (df.groupby("Forma_pago")["Total"].sum()
            .reset_index().sort_values("Total", ascending=False))
    pago.columns = ["Forma_pago","Total"]
    fig_pago = px.pie(pago, values="Total", names="Forma_pago", hole=0.4,
                      title="Distribucion por forma de pago",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pago.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>"
    )
    fig_pago.update_layout(height=320, margin=dict(t=40,b=10,l=0,r=0),
                            showlegend=False)
    st.plotly_chart(fig_pago, use_container_width=True)

with pb:
    vend = (df.groupby("Vendedor")
            .agg(Ingresos=("Total","sum"), Facturas=("Consecutivo","nunique"),
                 Unidades=("Cantidad","sum"))
            .reset_index().sort_values("Ingresos", ascending=False).head(10))
    vend["TicketProm"] = vend["Ingresos"] / vend["Facturas"]
    fig_vend = px.bar(
        vend.sort_values("Ingresos"), x="Ingresos", y="Vendedor",
        orientation="h", title="Top vendedores por ingresos",
        color="TicketProm", color_continuous_scale="Greens",
        text=vend.sort_values("Ingresos")["Ingresos"].apply(lambda x: f"${x/1e6:.2f}M"),
        labels={"Ingresos":"COP","Vendedor":"","TicketProm":"Ticket prom"}
    )
    fig_vend.update_traces(textposition="outside", textfont_size=10)
    fig_vend.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=80),
                            xaxis_tickformat="$,.0f",
                            coloraxis_showscale=False)
    st.plotly_chart(fig_vend, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: ANÁLISIS DE COMPRAS / PROVEEDORES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🛒 Análisis de Compras y Proveedores")

mask_c = True
if not compras_x.empty:
    dc = compras_x.copy()
    dc_mes = dc.groupby("Mes")["Valor"].sum().reset_index()

    qa, qb = st.columns(2)
    with qa:
        fig_gc = px.bar(dc_mes, x="Mes", y="Valor",
                        color_discrete_sequence=[NARANJA],
                        title="Gastos mensuales (xlsx 2025)",
                        labels={"Valor":"COP","Mes":""},
                        text=dc_mes["Valor"].apply(lambda x: f"${x/1e6:.1f}M"))
        fig_gc.update_traces(textposition="outside", textfont_size=10)
        fig_gc.update_layout(height=300, margin=dict(t=40,b=40,l=60,r=10),
                              yaxis_tickformat="$,.0f")
        fig_gc.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_gc, use_container_width=True)

    with qb:
        if "Proveedor" in dc.columns:
            top_prov = (dc.groupby("Proveedor")["Valor"].sum()
                        .reset_index().sort_values("Valor", ascending=False).head(10))
            top_prov.columns = ["Proveedor","Valor"]
            fig_prov = px.bar(
                top_prov.sort_values("Valor"), x="Valor", y="Proveedor",
                orientation="h", title="Top 10 proveedores",
                color_discrete_sequence=[AMARILLO],
                text=top_prov.sort_values("Valor")["Valor"].apply(lambda x: f"${x/1e6:.1f}M"),
                labels={"Valor":"COP","Proveedor":""}
            )
            fig_prov.update_traces(textposition="outside", textfont_size=10)
            fig_prov.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=80),
                                    xaxis_tickformat="$,.0f")
            st.plotly_chart(fig_prov, use_container_width=True)
else:
    st.info("No se encontraron archivos xlsx en la carpeta Compras. Se usa solo compras.csv")
    dc_csv = pd.read_csv(DATA_RAW / "compras.csv", encoding="latin-1")
    dc_csv.columns = list(range(len(dc_csv.columns)))
    st.write(dc_csv.head())

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: TABLA DETALLADA
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📋 Tabla detallada por producto"):
    tabla = (df.groupby(["Nombre","Categoria"])
             .agg(Ingresos=("Total","sum"), Unidades=("Cantidad","sum"),
                  Facturas=("Consecutivo","nunique"), Precio_prom=("Precio","mean"))
             .reset_index().sort_values("Ingresos", ascending=False))
    tabla["Participacion"] = (tabla["Ingresos"]/tabla["Ingresos"].sum()*100).round(1)
    tabla["Ingresos"]      = tabla["Ingresos"].apply(lambda x: f"${x:,.0f}")
    tabla["Precio_prom"]   = tabla["Precio_prom"].apply(lambda x: f"${x:,.0f}")
    tabla["Participacion"] = tabla["Participacion"].apply(lambda x: f"{x}%")
    st.dataframe(tabla, use_container_width=True, height=450)
