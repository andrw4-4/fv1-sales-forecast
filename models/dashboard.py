# -*- coding: utf-8 -*-
"""Dashboard Vivo Balanced Bites — panel de ventas + prediccion proxima semana."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.recetas import cargar_recetas, construir_mapeo, insumos_para_demanda

# ═══════════════════════════════════════════════════════════════════
# Config + paleta (neutro + verde + amarillo)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Vivo Balanced Bites",
    layout="wide",
    page_icon="🥗",
    initial_sidebar_state="expanded",
)

# Paleta tierra / verde / amarillo
CREMA      = "#FAF7F0"   # fondo principal
BEIGE      = "#EDE6D3"   # fondo tarjetas
VERDE      = "#4F7942"   # verde oliva principal
VERDE_CLARO= "#8FB87A"   # verde suave
VERDE_OSCURO="#2E4A2B"   # titulos
AMARILLO   = "#E6B84A"   # acento calido
AMARILLO_SUAVE="#F4DC8C"
MOSTAZA    = "#C89211"
CARBON     = "#3B3A36"   # texto
GRIS_SUAVE = "#8F8C85"

PALETA_SECUENCIAL = [VERDE_OSCURO, VERDE, VERDE_CLARO, AMARILLO, MOSTAZA]
PALETA_CATEGORIAS = [VERDE, AMARILLO, VERDE_CLARO, MOSTAZA, "#6B8E4E", "#D4A52A"]

st.markdown(f"""
<style>
    .stApp {{ background: {CREMA}; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px; }}
    h1, h2, h3, h4 {{ color: {VERDE_OSCURO}; font-family: 'Helvetica Neue', sans-serif; }}
    h1 {{ font-weight: 700; letter-spacing: -0.5px; }}
    h2 {{ border-bottom: 2px solid {AMARILLO}; padding-bottom: 6px; margin-top: 1rem; }}
    .metric-card {{
        background: {BEIGE}; border-radius: 10px; padding: 18px 20px;
        border-left: 5px solid {VERDE}; margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .pred-card {{
        background: linear-gradient(135deg, {BEIGE} 0%, {AMARILLO_SUAVE} 100%);
        border-radius: 12px; padding: 20px 24px;
        border-left: 6px solid {AMARILLO};
        margin-bottom: 12px;
    }}
    .pred-valor {{ font-size: 42px; font-weight: 800; color: {VERDE_OSCURO}; line-height: 1; }}
    .pred-label {{ font-size: 13px; color: {CARBON}; margin-top: 4px; }}
    div[data-testid="metric-container"] {{
        background: {BEIGE}; border-radius: 8px;
        padding: 12px 14px; border-left: 4px solid {VERDE};
    }}
    div[data-testid="metric-container"] label {{ color: {CARBON} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {BEIGE}; border-radius: 8px 8px 0 0; padding: 10px 18px;
        color: {CARBON}; border: none;
    }}
    .stTabs [aria-selected="true"] {{ background: {VERDE}; color: white; }}
    .caja-info {{
        background: {BEIGE}; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid {AMARILLO}; margin: 8px 0;
        font-size: 14px; color: {CARBON};
    }}
    hr {{ border-top: 1px solid {GRIS_SUAVE}; opacity: 0.3; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Rutas
# ═══════════════════════════════════════════════════════════════════
DATA_RAW  = ROOT / "data" / "raw"
DATA_FAC  = ROOT / "data" / "Facturas_lugar"
DATA_COM  = ROOT / "data" / "Compras"
DATA_PRED = ROOT / "data" / "predicciones"


# ═══════════════════════════════════════════════════════════════════
# Carga de datos (cacheada)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_ventas():
    v = pd.read_csv(DATA_RAW / "ventas.csv", encoding="utf-8", header=0)
    v.columns = ["Consecutivo", "Fecha", "Tipo_reg", "Tipo_clas", "Codigo", "Nombre",
                 "Vendedor", "Cantidad", "Precio", "Impuesto", "Total",
                 "Forma_pago", "Num_comp", "Establecimiento"]
    v["Fecha"] = pd.to_datetime(v["Fecha"], errors="coerce")
    v = v.dropna(subset=["Fecha"])
    v["Mes"] = v["Fecha"].dt.to_period("M").astype(str)
    v["DiaSemana"] = v["Fecha"].dt.day_name()
    v["Semana"] = v["Fecha"].dt.isocalendar().week.astype(int)
    v["Año"] = v["Fecha"].dt.year
    p = pd.read_csv(DATA_RAW / "info_productos.csv", encoding="utf-8")
    p.columns = ["Nombre", "Trans", "Cant_total", "Precio_prom",
                 "Ingresos_total", "Categoria"]
    v = v.merge(p[["Nombre", "Categoria"]], on="Nombre", how="left")
    v["Categoria"] = v["Categoria"].fillna("Sin categoria")
    return v


@st.cache_data
def cargar_facturas():
    dfs = []
    def parse_cop(s):
        if pd.isna(s): return 0.0
        return float(str(s).replace("$", "").replace(".", "").replace(",", ".").strip() or 0)
    for f in sorted(DATA_FAC.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, header=6)
            df = df.rename(columns={
                df.columns[1]: "Fecha", df.columns[13]: "Total"
            })
            df["Local"] = f.stem.split("-")[0]
            df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
            df["Total"] = df["Total"].apply(parse_cop)
            df = df.dropna(subset=["Fecha"])
            dfs.append(df[["Fecha", "Total", "Local"]])
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["Fecha", "Total", "Local", "Mes"])
    res = pd.concat(dfs, ignore_index=True)
    res["Mes"] = res["Fecha"].dt.to_period("M").astype(str)
    return res


@st.cache_data
def cargar_compras():
    dfs = []
    for f in sorted(DATA_COM.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, header=6)
            rename = {}
            for i, c in enumerate(df.columns):
                if i == 3: rename[c] = "Fecha"
                elif i == 6: rename[c] = "Proveedor"
                elif i == 8: rename[c] = "Valor"
            df = df.rename(columns=rename)
            if "Fecha" in df.columns and "Valor" in df.columns:
                df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
                df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0)
                dfs.append(df[["Fecha", "Proveedor", "Valor"]].dropna(subset=["Fecha"]))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["Fecha", "Proveedor", "Valor", "Mes"])
    res = pd.concat(dfs, ignore_index=True)
    res["Mes"] = res["Fecha"].dt.to_period("M").astype(str)
    return res


@st.cache_data
def cargar_predicciones():
    f = DATA_PRED / "predicciones_top10.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f)


@st.cache_data
def cargar_recetas_cached():
    return cargar_recetas()


# ═══════════════════════════════════════════════════════════════════
# Carga inicial
# ═══════════════════════════════════════════════════════════════════
with st.spinner("Cargando datos..."):
    ventas = cargar_ventas()
    facturas = cargar_facturas()
    compras = cargar_compras()
    predicciones = cargar_predicciones()
    recetas = cargar_recetas_cached()


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<h2 style='color:{VERDE_OSCURO}; margin-top:0'>🥗 Vivo</h2>",
                unsafe_allow_html=True)
    st.markdown("**Balanced Bites**")
    st.divider()

    st.markdown("### 📅 Periodo")
    meses = sorted(ventas["Mes"].dropna().unique())
    mes_ini, mes_fin = st.select_slider(
        "Rango de meses", options=meses, value=(meses[0], meses[-1]),
        label_visibility="collapsed",
    )

    st.markdown("### 🏢 Establecimiento")
    estabs = ["Todos"] + sorted(ventas["Establecimiento"].dropna().unique().tolist())
    estab = st.selectbox("Sede", estabs, label_visibility="collapsed")

    st.divider()
    st.caption(f"Datos desde {meses[0]} hasta {meses[-1]}")
    st.caption("iPPO · PICE · Uniandes")


mask = (ventas["Mes"] >= mes_ini) & (ventas["Mes"] <= mes_fin)
if estab != "Todos":
    mask &= ventas["Establecimiento"] == estab
df = ventas[mask].copy()


# ═══════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"# 🥗 Vivo Balanced Bites")
st.markdown(
    f"<p style='color:{CARBON}; margin-top:-8px; font-size:15px'>"
    f"Panel de ventas y prediccion de demanda · "
    f"<b>{mes_ini}</b> → <b>{mes_fin}</b> · "
    f"<b>{len(df):,}</b> registros</p>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# Tabs (navegacion principal)
# ═══════════════════════════════════════════════════════════════════
tab_pred, tab_resumen, tab_productos, tab_patrones, tab_compras = st.tabs([
    "🔮 Prediccion proxima semana",
    "📊 Resumen del negocio",
    "🏆 Productos",
    "🗓️ Patrones",
    "🛒 Compras",
])


# ─────────────────────────────────────────────────────────────────
# TAB: PREDICCION PROXIMA SEMANA
# ─────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("## 🔮 ¿Cuanto se va a vender la proxima semana?")
    st.markdown(
        f"<div class='caja-info'>"
        f"Prediccion generada con un modelo hibrido <b>Prophet + XGBoost</b> "
        f"entrenado con todo el historial disponible. "
        f"El modelo fue validado con walk-forward (re-entrenando semana a semana) "
        f"y optimizado para minimizar el error absoluto medio (MAE)."
        f"</div>",
        unsafe_allow_html=True,
    )

    if predicciones is None or predicciones.empty:
        st.warning(
            "⚠️ Aun no hay predicciones generadas. "
            "Ejecuta `python -m models.generar_predicciones` desde la terminal "
            "para entrenar el modelo y generar las predicciones."
        )
    else:
        fecha_prox = predicciones["fecha_proxima_semana"].iloc[0]
        fecha_str = pd.Timestamp(fecha_prox).strftime("%d %b %Y")
        st.markdown(f"### Semana del lunes **{fecha_str}**")

        # KPIs globales de la proxima semana
        total_unidades = predicciones["prediccion"].sum()
        productos_n = len(predicciones)
        mae_prom = predicciones["mae_hibrido"].mean()
        mejora = predicciones["mejora_mae"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🍱 Unidades previstas", f"{total_unidades:,.0f}")
        k2.metric("📦 Productos top", f"{productos_n}")
        k3.metric("🎯 MAE promedio", f"{mae_prom:.1f} unidades",
                  help="Error absoluto promedio en el test walk-forward")
        k4.metric("⬆️ Mejora vs Prophet solo", f"{mejora:+.1f}",
                  help="Cuantas unidades de MAE gana el modelo hibrido vs Prophet solo")

        st.markdown("---")

        # Selector de plato
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### 🍽️ Elige un plato")
            plato_sel = st.radio(
                "Selecciona",
                predicciones["producto"].tolist(),
                label_visibility="collapsed",
            )

        with c2:
            p = predicciones[predicciones["producto"] == plato_sel].iloc[0]
            pred_plato = float(p["prediccion"])

            st.markdown(
                f"<div class='pred-card'>"
                f"<div class='pred-label'>Prediccion para la semana del {fecha_str}</div>"
                f"<div class='pred-valor'>{pred_plato:,.0f} unidades</div>"
                f"<div class='pred-label' style='margin-top:10px'>"
                f"📐 <b>MAE test:</b> {p['mae_hibrido']:.1f} unidades &nbsp;|&nbsp; "
                f"📈 <b>Prophet solo:</b> {p['prophet_solo']:.0f} &nbsp;|&nbsp; "
                f"📆 <b>Historico:</b> {int(p['n_semanas'])} semanas"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Grafico comparativo de predicciones de la semana
            fig_preds = px.bar(
                predicciones.sort_values("prediccion", ascending=True),
                x="prediccion", y="producto",
                orientation="h",
                color_discrete_sequence=[VERDE],
                labels={"prediccion": "Unidades previstas", "producto": ""},
                text=predicciones.sort_values("prediccion", ascending=True)["prediccion"]
                    .apply(lambda x: f"{x:,.0f}"),
            )
            fig_preds.update_traces(textposition="outside",
                                     marker=dict(line=dict(color=VERDE_OSCURO, width=0.5)))
            fig_preds.update_layout(
                height=340, margin=dict(t=10, b=30, l=10, r=60),
                plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                font=dict(color=CARBON),
                xaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
            )
            # Resaltar el seleccionado
            colores = [AMARILLO if x == plato_sel else VERDE_CLARO
                       for x in predicciones.sort_values("prediccion", ascending=True)["producto"]]
            fig_preds.update_traces(marker_color=colores)
            st.plotly_chart(fig_preds, use_container_width=True)

        # ── Insumos necesarios ────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🥕 Insumos a comprar para esta semana")
        st.markdown(
            f"<div class='caja-info'>"
            f"Multiplica la receta base por las unidades pronosticadas. "
            f"Uselo como guia de compra."
            f"</div>",
            unsafe_allow_html=True,
        )

        mapa = construir_mapeo(recetas, predicciones["producto"].tolist())
        receta_plato = mapa.get(plato_sel)

        if receta_plato is None:
            st.info(
                f"🍴 **{plato_sel}** no tiene receta fija en el Excel "
                f"(probablemente es un plato personalizado donde el cliente elige los ingredientes). "
                f"Para estos casos se recomienda mantener el stock promedio habitual."
            )
        else:
            insumos = insumos_para_demanda(recetas, receta_plato, pred_plato)
            if insumos.empty:
                st.warning(f"No se encontraron ingredientes para la receta '{receta_plato}'.")
            else:
                st.markdown(
                    f"**Receta base:** `{receta_plato}` × **{pred_plato:,.0f}** unidades"
                )
                i1, i2 = st.columns([3, 2])
                with i1:
                    tabla = insumos.copy()
                    tabla["Cantidad"] = tabla["cantidad"].apply(lambda x: f"{x:,.1f}")
                    tabla = tabla[["ingrediente", "Cantidad", "unidad"]].rename(
                        columns={"ingrediente": "Ingrediente", "unidad": "Unidad"}
                    )
                    st.dataframe(tabla, use_container_width=True, height=380,
                                 hide_index=True)
                with i2:
                    fig_ing = px.bar(
                        insumos.head(10).sort_values("cantidad"),
                        x="cantidad", y="ingrediente",
                        orientation="h",
                        color_discrete_sequence=[AMARILLO],
                        labels={"cantidad": "Cantidad total", "ingrediente": ""},
                        title="Top 10 ingredientes por volumen",
                    )
                    fig_ing.update_layout(
                        height=380, margin=dict(t=40, b=30, l=10, r=10),
                        plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                        font=dict(color=CARBON),
                        title_font=dict(color=VERDE_OSCURO, size=14),
                    )
                    st.plotly_chart(fig_ing, use_container_width=True)

        # ── Insumos consolidados (todos los top 10) ─────────────────
        st.markdown("---")
        with st.expander("📋 Ver lista consolidada de TODOS los ingredientes (suma del top 10)"):
            filas = []
            for _, row in predicciones.iterrows():
                r = mapa.get(row["producto"])
                if not r:
                    continue
                insu = insumos_para_demanda(recetas, r, float(row["prediccion"]))
                insu["producto"] = row["producto"]
                filas.append(insu)
            if filas:
                consolidado = pd.concat(filas, ignore_index=True)
                agg = (consolidado.groupby(["ingrediente", "unidad"], as_index=False)["cantidad"]
                       .sum()
                       .sort_values("cantidad", ascending=False))
                agg["cantidad"] = agg["cantidad"].round(1)
                st.dataframe(
                    agg.rename(columns={
                        "ingrediente": "Ingrediente", "unidad": "Unidad", "cantidad": "Cantidad total"
                    }),
                    use_container_width=True, hide_index=True, height=400,
                )


# ─────────────────────────────────────────────────────────────────
# TAB: RESUMEN DEL NEGOCIO (KPIs clave para el cliente)
# ─────────────────────────────────────────────────────────────────
with tab_resumen:
    st.markdown("## 📊 Salud del negocio")

    # KPIs principales
    ingresos = df["Total"].sum()
    n_fact = df["Consecutivo"].nunique()
    unidades = df["Cantidad"].sum()
    ticket_prom = df.groupby("Consecutivo")["Total"].sum().mean()

    # KPIs de crecimiento (mes vs mes)
    ing_mes = df.groupby("Mes")["Total"].sum().reset_index().sort_values("Mes")
    if len(ing_mes) >= 2:
        crec_mes = (ing_mes["Total"].iloc[-1] - ing_mes["Total"].iloc[-2]) / ing_mes["Total"].iloc[-2] * 100
    else:
        crec_mes = 0

    # Gasto / margen
    gasto_csv = pd.read_csv(DATA_RAW / "compras.csv", encoding="utf-8", header=0)
    gasto_csv.columns = ["Consecutivo", "Factura", "ID", "Proveedor", "Fecha_c", "Fecha_m",
                         "Fecha", "Contacto", "Tipo_reg", "Tipo_clas", "Codigo", "Nombre",
                         "Cantidad", "Precio", "Total", "Forma_pago", "Fecha_v", "Periodo"]
    gasto_csv["Mes"] = gasto_csv["Periodo"].astype(str).str[:7]
    gmes_csv = gasto_csv.groupby("Mes")["Total"].sum().reset_index().rename(columns={"Total": "Gastos"})
    if not compras.empty:
        gmes_x = compras.groupby("Mes")["Valor"].sum().reset_index().rename(columns={"Valor": "Gastos"})
        gastos_mes = pd.concat([gmes_csv, gmes_x]).groupby("Mes")["Gastos"].sum().reset_index()
    else:
        gastos_mes = gmes_csv
    gastos_mes = gastos_mes[(gastos_mes["Mes"] >= mes_ini) & (gastos_mes["Mes"] <= mes_fin)]

    total_gasto = gastos_mes["Gastos"].sum()
    margen = ingresos - total_gasto
    margen_pct = (margen / ingresos * 100) if ingresos > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Ingresos", f"${ingresos/1e6:.1f}M",
              f"{crec_mes:+.1f}% vs mes ant." if crec_mes else None)
    k2.metric("🧾 Facturas", f"{n_fact:,}")
    k3.metric("🎫 Ticket promedio", f"${ticket_prom:,.0f}")
    k4.metric("🍱 Unidades vendidas", f"{int(unidades):,}")
    k5.metric("📈 Margen bruto", f"{margen_pct:.1f}%",
              f"${margen/1e6:+.1f}M")

    st.markdown("---")

    # Grafico: ingresos vs gastos por mes
    st.markdown("### Ingresos vs gastos por mes")
    if not facturas.empty:
        ing_fac = facturas.groupby("Mes")["Total"].sum().reset_index().rename(columns={"Total": "Ingresos"})
        ing_csv = ventas.groupby("Mes")["Total"].sum().reset_index().rename(columns={"Total": "Ingresos_csv"})
        ing_total = ing_fac.merge(ing_csv, on="Mes", how="outer")
        ing_total["Ingresos"] = ing_total["Ingresos"].fillna(ing_total["Ingresos_csv"])
        ing_total = ing_total[["Mes", "Ingresos"]].fillna(0)
    else:
        ing_total = ventas.groupby("Mes")["Total"].sum().reset_index().rename(columns={"Total": "Ingresos"})

    gasto_total_csv = gasto_csv.groupby("Mes")["Total"].sum().reset_index().rename(columns={"Total": "Gastos"})
    if not compras.empty:
        gasto_total_x = compras.groupby("Mes")["Valor"].sum().reset_index().rename(columns={"Valor": "Gastos"})
        gasto_total_df = pd.concat([gasto_total_csv, gasto_total_x]).groupby("Mes")["Gastos"].sum().reset_index()
    else:
        gasto_total_df = gasto_total_csv

    hist = ing_total.merge(gasto_total_df, on="Mes", how="outer").fillna(0).sort_values("Mes")
    hist["Margen"] = hist["Ingresos"] - hist["Gastos"]

    fig = go.Figure()
    fig.add_bar(x=hist["Mes"], y=hist["Ingresos"] / 1e6,
                name="Ingresos", marker_color=VERDE)
    fig.add_bar(x=hist["Mes"], y=hist["Gastos"] / 1e6,
                name="Gastos", marker_color=AMARILLO)
    fig.add_scatter(x=hist["Mes"], y=hist["Margen"] / 1e6,
                    name="Margen bruto", mode="lines+markers",
                    line=dict(color=VERDE_OSCURO, width=2.5),
                    marker=dict(size=7))
    fig.update_layout(
        barmode="group", height=380,
        yaxis_title="Millones COP",
        plot_bgcolor=CREMA, paper_bgcolor=CREMA,
        font=dict(color=CARBON),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=40, l=60, r=20),
        hovermode="x unified",
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB: PRODUCTOS
# ─────────────────────────────────────────────────────────────────
with tab_productos:
    st.markdown("## 🏆 Productos mas vendidos")
    n_top = st.slider("Cantidad de productos a mostrar", 5, 25, 12)

    top = (df.groupby(["Nombre", "Categoria"])
           .agg(Ingresos=("Total", "sum"), Unidades=("Cantidad", "sum"),
                Facturas=("Consecutivo", "nunique"),
                Precio_prom=("Precio", "mean"))
           .reset_index().sort_values("Ingresos", ascending=False).head(n_top))

    t1, t2 = st.columns(2)
    with t1:
        fig = px.bar(top.sort_values("Ingresos"), x="Ingresos", y="Nombre",
                     orientation="h", color_discrete_sequence=[VERDE],
                     title="Top por ingresos",
                     text=top.sort_values("Ingresos")["Ingresos"].apply(lambda x: f"${x/1e6:.2f}M"))
        fig.update_traces(textposition="outside", marker_color=VERDE)
        fig.update_layout(height=440, margin=dict(t=40, b=10, l=10, r=80),
                          plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                          title_font=dict(color=VERDE_OSCURO),
                          font=dict(color=CARBON),
                          xaxis=dict(tickformat="$,.0f", showgrid=True, gridcolor="#E0DDD3"),
                          yaxis=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig = px.bar(top.sort_values("Unidades"), x="Unidades", y="Nombre",
                     orientation="h", color_discrete_sequence=[AMARILLO],
                     title="Top por unidades vendidas",
                     text=top.sort_values("Unidades")["Unidades"].astype(int))
        fig.update_traces(textposition="outside", marker_color=AMARILLO)
        fig.update_layout(height=440, margin=dict(t=40, b=10, l=10, r=60),
                          plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                          title_font=dict(color=VERDE_OSCURO),
                          font=dict(color=CARBON),
                          xaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
                          yaxis=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Concentracion de ingresos (regla 80/20)")
    pareto = (df.groupby("Nombre")["Total"].sum()
              .sort_values(ascending=False).reset_index())
    pareto["Acum_pct"] = pareto["Total"].cumsum() / pareto["Total"].sum() * 100
    pareto["Rank"] = range(1, len(pareto) + 1)
    n80 = (pareto["Acum_pct"] <= 80).sum() + 1

    fig = go.Figure()
    fig.add_bar(x=pareto["Rank"], y=pareto["Total"] / 1e3,
                name="Ingresos (miles)", marker_color=VERDE_CLARO)
    fig.add_scatter(x=pareto["Rank"], y=pareto["Acum_pct"],
                    name="% Acumulado", yaxis="y2",
                    line=dict(color=AMARILLO, width=3))
    fig.add_vline(x=n80, line_dash="dash", line_color=VERDE_OSCURO,
                  annotation_text=f"{n80} productos = 80% ingresos",
                  annotation_position="top right")
    fig.update_layout(
        height=300,
        yaxis=dict(title="Miles COP", showgrid=True, gridcolor="#E0DDD3"),
        yaxis2=dict(title="% Acumulado", overlaying="y", side="right",
                    range=[0, 105], ticksuffix="%"),
        plot_bgcolor=CREMA, paper_bgcolor=CREMA,
        font=dict(color=CARBON),
        margin=dict(t=10, b=40, l=60, r=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabla detallada por producto"):
        tabla = (df.groupby(["Nombre", "Categoria"])
                 .agg(Ingresos=("Total", "sum"), Unidades=("Cantidad", "sum"),
                      Facturas=("Consecutivo", "nunique"),
                      Precio_prom=("Precio", "mean"))
                 .reset_index().sort_values("Ingresos", ascending=False))
        tabla["%"] = (tabla["Ingresos"] / tabla["Ingresos"].sum() * 100).round(1)
        tabla["Ingresos"] = tabla["Ingresos"].apply(lambda x: f"${x:,.0f}")
        tabla["Precio_prom"] = tabla["Precio_prom"].apply(lambda x: f"${x:,.0f}")
        tabla["%"] = tabla["%"].apply(lambda x: f"{x}%")
        st.dataframe(tabla, use_container_width=True, height=450, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# TAB: PATRONES
# ─────────────────────────────────────────────────────────────────
with tab_patrones:
    st.markdown("## 🗓️ Comportamiento semanal")

    ORDEN_DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    NOMBRES = {"Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mie",
               "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sab", "Sunday": "Dom"}

    c1, c2 = st.columns(2)
    with c1:
        vdia = (df.groupby("DiaSemana")
                .agg(Ingresos=("Total", "sum"),
                     Facturas=("Consecutivo", "nunique"))
                .reindex(ORDEN_DIAS).reset_index())
        vdia["Dia"] = vdia["DiaSemana"].map(NOMBRES)
        fig = go.Figure()
        fig.add_bar(x=vdia["Dia"], y=vdia["Ingresos"] / 1e6,
                    marker_color=VERDE, text=vdia["Ingresos"].apply(lambda x: f"${x/1e6:.1f}M"),
                    textposition="outside")
        fig.update_layout(
            title="Ingresos por dia de la semana",
            height=330, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            yaxis_title="Millones COP",
            font=dict(color=CARBON), title_font=dict(color=VERDE_OSCURO),
            margin=dict(t=40, b=10, l=60, r=10),
            yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ Sabado: operacion minima (estudiantes no van). Domingo: pedidos/catering.")

    with c2:
        pivot = (df.groupby(["DiaSemana", "Mes"])["Total"]
                 .sum().unstack(fill_value=0))
        pivot = pivot.reindex([d for d in ORDEN_DIAS if d in pivot.index])
        pivot.index = [NOMBRES[d] for d in pivot.index]
        fig = px.imshow(
            pivot / 1e3, aspect="auto",
            color_continuous_scale=[[0, CREMA], [0.5, VERDE_CLARO], [1, VERDE_OSCURO]],
            title="Calor: dia × mes (miles COP)",
            labels=dict(color="Miles COP"),
        )
        fig.update_layout(
            height=330, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            title_font=dict(color=VERDE_OSCURO),
            font=dict(color=CARBON),
            margin=dict(t=40, b=10, l=80, r=10),
        )
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ingresos por establecimiento a lo largo del tiempo")
    vlocal = ventas[mask].groupby(["Mes", "Establecimiento"])["Total"].sum().reset_index()
    fig = px.line(vlocal, x="Mes", y="Total", color="Establecimiento",
                  markers=True, color_discrete_sequence=[VERDE, AMARILLO])
    fig.update_layout(
        height=300, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
        yaxis_tickformat="$,.0f", hovermode="x unified",
        font=dict(color=CARBON),
        margin=dict(t=10, b=40, l=60, r=10),
        yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB: COMPRAS Y PROVEEDORES
# ─────────────────────────────────────────────────────────────────
with tab_compras:
    st.markdown("## 🛒 Compras y proveedores")

    if compras.empty:
        st.info("No hay datos de compras disponibles (carpeta Compras vacia).")
    else:
        dc_mes = compras.groupby("Mes")["Valor"].sum().reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                dc_mes, x="Mes", y="Valor",
                color_discrete_sequence=[AMARILLO],
                title="Gastos mensuales",
                text=dc_mes["Valor"].apply(lambda x: f"${x/1e6:.1f}M"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=340, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                title_font=dict(color=VERDE_OSCURO), font=dict(color=CARBON),
                yaxis_tickformat="$,.0f",
                margin=dict(t=40, b=40, l=60, r=10),
                yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "Proveedor" in compras.columns:
                top_prov = (compras.groupby("Proveedor")["Valor"].sum()
                            .reset_index().sort_values("Valor", ascending=False).head(10))
                fig = px.bar(
                    top_prov.sort_values("Valor"), x="Valor", y="Proveedor",
                    orientation="h",
                    color_discrete_sequence=[VERDE],
                    title="Top 10 proveedores",
                    text=top_prov.sort_values("Valor")["Valor"].apply(lambda x: f"${x/1e6:.1f}M"),
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    height=340, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                    title_font=dict(color=VERDE_OSCURO), font=dict(color=CARBON),
                    xaxis_tickformat="$,.0f",
                    margin=dict(t=40, b=10, l=10, r=80),
                    xaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
                    yaxis=dict(title=""),
                )
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    f"<p style='text-align:center; color:{GRIS_SUAVE}; font-size:12px'>"
    f"🥗 Vivo Balanced Bites · Modelo Prophet + XGBoost · "
    f"Para regenerar predicciones: <code>python -m models.generar_predicciones</code>"
    f"</p>",
    unsafe_allow_html=True,
)
