# -*- coding: utf-8 -*-
"""Dashboard Vivo Balanced Bites — panel de ventas, costos y prediccion multi-semana."""
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

CREMA         = "#FAF7F0"
BEIGE         = "#EDE6D3"
VERDE         = "#4F7942"
VERDE_CLARO   = "#8FB87A"
VERDE_OSCURO  = "#2E4A2B"
AMARILLO      = "#E6B84A"
AMARILLO_SUAVE= "#F4DC8C"
MOSTAZA       = "#C89211"
CARBON        = "#3B3A36"
GRIS_SUAVE    = "#8F8C85"
ROJO_SUAVE    = "#B85C3C"

PALETA_CATEGORIAS = [VERDE, AMARILLO, VERDE_CLARO, MOSTAZA, "#6B8E4E", "#D4A52A",
                     VERDE_OSCURO, "#A5C481", "#F2CA5E"]

st.markdown(f"""
<style>
    .stApp {{ background: {CREMA}; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px; }}
    h1, h2, h3, h4 {{ color: {VERDE_OSCURO}; font-family: 'Helvetica Neue', sans-serif; }}
    h1 {{ font-weight: 700; letter-spacing: -0.5px; }}
    h2 {{ border-bottom: 2px solid {AMARILLO}; padding-bottom: 6px; margin-top: 1rem; }}
    .pred-card {{
        background: linear-gradient(135deg, {BEIGE} 0%, {AMARILLO_SUAVE} 100%);
        border-radius: 12px; padding: 20px 24px;
        border-left: 6px solid {AMARILLO};
        margin-bottom: 12px;
    }}
    .pred-valor {{ font-size: 42px; font-weight: 800; color: {VERDE_OSCURO}; line-height: 1; }}
    .pred-label {{ font-size: 13px; color: {CARBON}; margin-top: 4px; }}
    .sem-card {{
        background: {BEIGE}; border-radius: 10px; padding: 14px 16px;
        border-top: 4px solid {VERDE}; text-align: center;
        height: 100%;
    }}
    .sem-card-weak {{
        background: {BEIGE}; border-radius: 10px; padding: 14px 16px;
        border-top: 4px solid {GRIS_SUAVE}; text-align: center;
        opacity: 0.85;
    }}
    .sem-valor {{ font-size: 28px; font-weight: 700; color: {VERDE_OSCURO}; }}
    .sem-rango {{ font-size: 12px; color: {CARBON}; margin-top: 4px; }}
    .sem-prob {{ font-size: 11px; color: {CARBON}; margin-top: 2px; font-style: italic; }}
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


DATA_RAW  = ROOT / "data" / "raw"
DATA_FAC  = ROOT / "data" / "Facturas_lugar"
DATA_COM  = ROOT / "data" / "Compras"
DATA_PRED = ROOT / "data" / "predicciones"


# ═══════════════════════════════════════════════════════════════════
# Tipos de plato (derivados del nombre)
# ═══════════════════════════════════════════════════════════════════
def categorizar_plato(nombre: str) -> str:
    n = str(nombre).lower()
    if "sándwich" in n or "sandwich" in n:
        return "Sándwich"
    if "bowl" in n:
        return "Bowl"
    if "ensalada" in n:
        return "Ensalada"
    if "pasta" in n:
        return "Pasta"
    if "arma tu plato" in n:
        return "Arma tu plato"
    if any(k in n for k in ["parfait", "pancake", "waffle", "yogurt", "granola",
                             "tostada", "bagel", "wake up", "bananabread", "choco"]):
        return "Desayuno"
    if any(k in n for k in ["smoothie", "jugo", "shot", "hatsu", "agua", "soda",
                             "chocolate", "milo", "cafe", "café", "americano",
                             "latte", "cappuccino"]):
        return "Bebida"
    if "sopa" in n:
        return "Sopa"
    if "pavoneta" in n or "mexi" in n or "rustico" in n or "becha" in n:
        return "Sándwich"
    return "Otros"


# ═══════════════════════════════════════════════════════════════════
# Carga de datos
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_ventas():
    v = pd.read_csv(DATA_RAW / "ventas.csv", encoding="utf-8", header=0)
    v.columns = ["Consecutivo", "Fecha", "Tipo_reg", "Tipo_clas", "Codigo", "Nombre",
                 "Vendedor", "Cantidad", "Precio", "Impuesto", "Total",
                 "Forma_pago", "Num_comp", "Establecimiento"]
    v["Fecha"] = pd.to_datetime(v["Fecha"], errors="coerce")
    v = v.dropna(subset=["Fecha"])
    v["Cantidad"] = pd.to_numeric(v["Cantidad"], errors="coerce").fillna(0)
    v["Precio"] = pd.to_numeric(v["Precio"], errors="coerce").fillna(0)
    v["Total"] = pd.to_numeric(v["Total"], errors="coerce").fillna(0)
    v["Mes"] = v["Fecha"].dt.to_period("M").astype(str)
    v["Semana"] = v["Fecha"].dt.to_period("W").apply(lambda x: x.start_time).dt.date.astype(str)
    v["Dia"] = v["Fecha"].dt.date.astype(str)
    v["DiaSemana"] = v["Fecha"].dt.day_name()
    v["Año"] = v["Fecha"].dt.year
    v["TipoPlato"] = v["Nombre"].apply(categorizar_plato)

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
            df = df.rename(columns={df.columns[1]: "Fecha", df.columns[13]: "Total"})
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


# ═══════════════════════════════════════════════════════════════════
# COSTOS (fix: excluir "Pagos" de los xlsx — son pagos de facturas ya contadas)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_costos():
    """Unifica costos de los xlsx de Compras (2025) y compras.csv (2024).

    Regla: solo se toman como costo las transacciones tipo 'Compra/Gasto' y
    'Documento soporte'. Los 'Pagos' se excluyen porque son pagos de facturas
    que YA fueron registradas como 'Compra/Gasto' (evita doble conteo).
    """
    dfs = []

    # ── xlsx 2025
    for f in sorted(DATA_COM.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, header=6)
            df = df.rename(columns={
                df.columns[0]: "Tipo",
                df.columns[3]: "Fecha",
                df.columns[6]: "Proveedor",
                df.columns[8]: "Valor",
            })
            if "Fecha" in df.columns and "Valor" in df.columns and "Tipo" in df.columns:
                df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
                df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0)
                df = df.dropna(subset=["Fecha"])
                # FIX: solo compras/gastos reales
                df = df[df["Tipo"].isin(["Compra/Gasto", "Documento soporte"])]
                dfs.append(df[["Fecha", "Tipo", "Proveedor", "Valor"]])
        except Exception:
            pass

    # ── csv 2024 (line items por producto, filtrar "Impuesto Total")
    try:
        csv = pd.read_csv(DATA_RAW / "compras.csv", encoding="utf-8", header=0)
        csv.columns = ["Consecutivo", "Factura", "ID", "Proveedor", "Fecha_c", "Fecha_m",
                       "Fecha", "Contacto", "Tipo_reg", "Tipo_clas", "Codigo", "Nombre",
                       "Cantidad", "Precio", "Total", "Forma_pago", "Fecha_v", "Periodo"]
        csv["Fecha"] = pd.to_datetime(csv["Fecha"], errors="coerce")
        csv = csv.dropna(subset=["Fecha"])
        csv = csv[csv["Tipo_reg"] == "Secuencia"]  # excluir totales de impuesto
        csv["Valor"] = pd.to_numeric(csv["Total"], errors="coerce").fillna(0)
        csv["Tipo"] = "Compra/Gasto"
        dfs.append(csv[["Fecha", "Tipo", "Proveedor", "Valor"]])
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame(columns=["Fecha", "Tipo", "Proveedor", "Valor", "Mes", "Semana", "Dia"])
    res = pd.concat(dfs, ignore_index=True)
    res["Mes"] = res["Fecha"].dt.to_period("M").astype(str)
    res["Semana"] = res["Fecha"].dt.to_period("W").apply(lambda x: x.start_time).dt.date.astype(str)
    res["Dia"] = res["Fecha"].dt.date.astype(str)
    return res


@st.cache_data
def cargar_predicciones():
    out = {}
    for nombre, archivo in [
        ("resumen",  "predicciones_top10.parquet"),
        ("4sem",     "predicciones_4_semanas.parquet"),
        ("precios",  "precios_unitarios.parquet"),
        ("historial","historial_walkforward.parquet"),
    ]:
        f = DATA_PRED / archivo
        out[nombre] = pd.read_parquet(f) if f.exists() else None
    return out


@st.cache_data
def cargar_recetas_cached():
    return cargar_recetas()


# ═══════════════════════════════════════════════════════════════════
# Utilidad: agrupar por granularidad (Día / Semana / Mes)
# ═══════════════════════════════════════════════════════════════════
def agrupar_por_granularidad(df: pd.DataFrame, columna_fecha: str,
                              valor: str, granularidad: str,
                              por: str | None = None,
                              agg: str = "sum") -> pd.DataFrame:
    """Agrupa un DataFrame por Día/Semana/Mes y opcionalmente por otra columna."""
    if granularidad == "Día":
        col = "Dia"
    elif granularidad == "Semana":
        col = "Semana"
    else:
        col = "Mes"

    group_cols = [col] + ([por] if por else [])
    if agg == "sum":
        out = df.groupby(group_cols, as_index=False)[valor].sum()
    elif agg == "mean":
        out = df.groupby(group_cols, as_index=False)[valor].mean()
    elif agg == "nunique":
        out = df.groupby(group_cols, as_index=False)[valor].nunique()
    else:
        out = df.groupby(group_cols, as_index=False)[valor].sum()
    out = out.rename(columns={col: "Periodo"})
    return out.sort_values("Periodo")


# ═══════════════════════════════════════════════════════════════════
# Carga inicial
# ═══════════════════════════════════════════════════════════════════
with st.spinner("Cargando datos..."):
    ventas = cargar_ventas()
    facturas = cargar_facturas()
    costos = cargar_costos()
    preds = cargar_predicciones()
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

costos_mask = (costos["Mes"] >= mes_ini) & (costos["Mes"] <= mes_fin) if not costos.empty else None
costos_f = costos[costos_mask].copy() if costos_mask is not None else costos.copy()


# ═══════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════
st.markdown("# 🥗 Vivo Balanced Bites")
st.markdown(
    f"<p style='color:{CARBON}; margin-top:-8px; font-size:15px'>"
    f"Panel de ventas, costos y prediccion de demanda · "
    f"<b>{mes_ini}</b> → <b>{mes_fin}</b> · "
    f"<b>{len(df):,}</b> registros de venta</p>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# Tabs principales
# ═══════════════════════════════════════════════════════════════════
tab_pred, tab_resumen, tab_historico, tab_precision, tab_productos, tab_patrones, tab_compras = st.tabs([
    "🔮 Predicción 4 semanas",
    "📊 Resumen",
    "📈 Histórico ventas",
    "📉 Precisión",
    "🏆 Productos",
    "🗓️ Patrones",
    "🛒 Compras",
])


# ═══════════════════════════════════════════════════════════════════
# TAB: PREDICCION 4 SEMANAS
# ═══════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("## 🔮 Predicción de demanda — próximas 4 semanas")
    st.markdown(
        f"<div class='caja-info'>"
        f"Modelo híbrido <b>Prophet + XGBoost</b>. Entre más lejos en el futuro, "
        f"<b>menor es la probabilidad</b> de que el valor exacto se cumpla "
        f"(el rango de confianza se ensancha)."
        f"</div>",
        unsafe_allow_html=True,
    )

    if preds["resumen"] is None or preds["4sem"] is None:
        st.warning(
            "⚠️ Aun no hay predicciones generadas. "
            "Ejecuta `python -m models.generar_predicciones` desde la terminal."
        )
    else:
        df_4sem = preds["4sem"]
        df_precios = preds["precios"] if preds["precios"] is not None else pd.DataFrame(columns=["producto", "precio_unitario"])

        productos = sorted(df_4sem["producto"].unique().tolist())

        # ── Selector de plato
        c_sel, c_info = st.columns([1, 3])
        with c_sel:
            st.markdown("### 🍽️ Elige un plato")
            plato_sel = st.radio(
                "Selecciona",
                productos,
                label_visibility="collapsed",
            )
        with c_info:
            p4 = df_4sem[df_4sem["producto"] == plato_sel].sort_values("semana_offset")
            precio_plato = float(df_precios.loc[df_precios["producto"] == plato_sel,
                                                 "precio_unitario"].iloc[0]) if not df_precios.empty and plato_sel in df_precios["producto"].values else 0

            # Ganancias estimadas del plato en las 4 semanas
            ingreso_total = (p4["prediccion"] * precio_plato).sum()
            unidades_total = p4["prediccion"].sum()

            st.markdown(
                f"<div class='pred-card'>"
                f"<div class='pred-label'>Predicción 4 semanas — <b>{plato_sel}</b></div>"
                f"<div class='pred-valor'>{unidades_total:,.0f} unidades</div>"
                f"<div class='pred-label' style='margin-top:8px'>"
                f"💰 Ingreso estimado: <b>${ingreso_total:,.0f}</b> "
                f"&nbsp;·&nbsp; Precio unitario: ${precio_plato:,.0f}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 📅 Predicción semana por semana")

        # Tarjetas para sem 1, 2, 3, 4
        cols = st.columns(4)
        for idx, (_, row) in enumerate(p4.iterrows()):
            col = cols[idx]
            fecha_txt = pd.Timestamp(row["fecha"]).strftime("%d %b")
            conf = int(row["confianza_pct"])
            pred = row["prediccion"]
            lower = row["prediccion_lower"]
            upper = row["prediccion_upper"]
            ingreso_sem = pred * precio_plato
            estilo = "sem-card" if conf >= 70 else "sem-card-weak"
            emoji = "🟢" if conf >= 70 else ("🟡" if conf >= 50 else "🟠")
            col.markdown(
                f"<div class='{estilo}'>"
                f"<div style='font-size:12px; color:{GRIS_SUAVE}'>Semana {idx+1}</div>"
                f"<div style='font-size:14px; color:{VERDE_OSCURO}; font-weight:600'>{fecha_txt}</div>"
                f"<div class='sem-valor'>{pred:,.0f}</div>"
                f"<div class='sem-rango'>rango: {lower:,.0f} – {upper:,.0f}</div>"
                f"<div class='sem-rango'>💰 ${ingreso_sem:,.0f}</div>"
                f"<div class='sem-prob'>{emoji} confianza {conf}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        # ── Grafico: banda de confianza
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(p4["fecha"]) + list(p4["fecha"][::-1]),
            y=list(p4["prediccion_upper"]) + list(p4["prediccion_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(143, 184, 122, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="Rango de confianza",
        ))
        fig.add_trace(go.Scatter(
            x=p4["fecha"], y=p4["prediccion"],
            mode="lines+markers+text",
            line=dict(color=VERDE_OSCURO, width=3),
            marker=dict(size=10, color=AMARILLO, line=dict(color=VERDE_OSCURO, width=2)),
            text=[f"{x:,.0f}" for x in p4["prediccion"]],
            textposition="top center",
            name="Predicción",
        ))
        fig.update_layout(
            height=300, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            font=dict(color=CARBON),
            title_font=dict(color=VERDE_OSCURO),
            title=f"Evolución semanal esperada — {plato_sel}",
            margin=dict(t=50, b=40, l=60, r=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Unidades", showgrid=True, gridcolor="#E0DDD3"),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Insumos necesarios semana 1
        st.markdown("---")
        st.markdown("## 🥕 Insumos a comprar — semana 1")
        mapa = construir_mapeo(recetas, productos)
        receta_plato = mapa.get(plato_sel)
        pred_sem1 = float(p4.iloc[0]["prediccion"]) if len(p4) > 0 else 0

        if receta_plato is None:
            st.info(
                f"🍴 **{plato_sel}** no tiene receta fija (plato personalizado). "
                f"Mantener el stock promedio habitual."
            )
        else:
            insumos = insumos_para_demanda(recetas, receta_plato, pred_sem1)
            if insumos.empty:
                st.warning(f"No se encontraron ingredientes para '{receta_plato}'.")
            else:
                st.markdown(
                    f"**Receta base:** `{receta_plato}` × **{pred_sem1:,.0f}** unidades (semana 1)"
                )
                i1, i2 = st.columns([3, 2])
                with i1:
                    tabla = insumos.copy()
                    tabla["Cantidad"] = tabla["cantidad"].apply(lambda x: f"{x:,.1f}")
                    tabla = tabla[["ingrediente", "Cantidad", "unidad"]].rename(
                        columns={"ingrediente": "Ingrediente", "unidad": "Unidad"}
                    )
                    st.dataframe(tabla, use_container_width=True, height=360, hide_index=True)
                with i2:
                    fig_ing = px.bar(
                        insumos.head(10).sort_values("cantidad"),
                        x="cantidad", y="ingrediente", orientation="h",
                        color_discrete_sequence=[AMARILLO],
                        labels={"cantidad": "Cantidad", "ingrediente": ""},
                        title="Top 10 ingredientes",
                    )
                    fig_ing.update_layout(
                        height=360, margin=dict(t=40, b=30, l=10, r=10),
                        plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                        font=dict(color=CARBON),
                        title_font=dict(color=VERDE_OSCURO, size=14),
                    )
                    st.plotly_chart(fig_ing, use_container_width=True)

        # ── Ingreso total esperado de TODOS los platos
        st.markdown("---")
        st.markdown("## 💰 Ingreso total estimado — próximas 4 semanas")
        st.markdown(
            f"<div class='caja-info'>Suma del ingreso esperado de los <b>"
            f"{len(productos)}</b> platos top. Incluye rangos de confianza por semana.</div>",
            unsafe_allow_html=True,
        )

        # merge predicciones con precios
        ingresos_df = df_4sem.merge(df_precios, on="producto", how="left").fillna({"precio_unitario": 0})
        ingresos_df["ingreso"] = ingresos_df["prediccion"] * ingresos_df["precio_unitario"]
        ingresos_df["ingreso_lower"] = ingresos_df["prediccion_lower"] * ingresos_df["precio_unitario"]
        ingresos_df["ingreso_upper"] = ingresos_df["prediccion_upper"] * ingresos_df["precio_unitario"]

        ingreso_por_semana = ingresos_df.groupby(["semana_offset", "fecha", "confianza_pct"], as_index=False).agg(
            ingreso=("ingreso", "sum"),
            ingreso_lower=("ingreso_lower", "sum"),
            ingreso_upper=("ingreso_upper", "sum"),
        )

        cols2 = st.columns(4)
        for idx, (_, row) in enumerate(ingreso_por_semana.iterrows()):
            col = cols2[idx]
            fecha_txt = pd.Timestamp(row["fecha"]).strftime("%d %b")
            conf = int(row["confianza_pct"])
            estilo = "sem-card" if conf >= 70 else "sem-card-weak"
            emoji = "🟢" if conf >= 70 else ("🟡" if conf >= 50 else "🟠")
            col.markdown(
                f"<div class='{estilo}'>"
                f"<div style='font-size:12px; color:{GRIS_SUAVE}'>Semana {idx+1}</div>"
                f"<div style='font-size:14px; color:{VERDE_OSCURO}; font-weight:600'>{fecha_txt}</div>"
                f"<div class='sem-valor'>${row['ingreso']/1e6:.2f}M</div>"
                f"<div class='sem-rango'>${row['ingreso_lower']/1e6:.2f}M – ${row['ingreso_upper']/1e6:.2f}M</div>"
                f"<div class='sem-prob'>{emoji} confianza {conf}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        total_4sem = ingreso_por_semana["ingreso"].sum()
        total_lower = ingreso_por_semana["ingreso_lower"].sum()
        total_upper = ingreso_por_semana["ingreso_upper"].sum()
        st.markdown(
            f"<div class='pred-card'>"
            f"<div class='pred-label'>Total estimado 4 semanas (top {len(productos)} platos)</div>"
            f"<div class='pred-valor'>${total_4sem/1e6:.2f}M</div>"
            f"<div class='pred-label' style='margin-top:8px'>"
            f"Rango: ${total_lower/1e6:.2f}M – ${total_upper/1e6:.2f}M"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# TAB: RESUMEN (KPIs + ingresos vs gastos)
# ═══════════════════════════════════════════════════════════════════
with tab_resumen:
    st.markdown("## 📊 Salud del negocio")

    ingresos = df["Total"].sum()
    n_fact = df["Consecutivo"].nunique()
    unidades = df["Cantidad"].sum()
    ticket_prom = df.groupby("Consecutivo")["Total"].sum().mean()

    # crecimiento mes vs mes
    ing_mes_df = df.groupby("Mes")["Total"].sum().reset_index().sort_values("Mes")
    if len(ing_mes_df) >= 2:
        crec_mes = (ing_mes_df["Total"].iloc[-1] - ing_mes_df["Total"].iloc[-2]) / ing_mes_df["Total"].iloc[-2] * 100
    else:
        crec_mes = 0

    total_gasto = costos_f["Valor"].sum() if not costos_f.empty else 0
    margen = ingresos - total_gasto
    margen_pct = (margen / ingresos * 100) if ingresos > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Ingresos", f"${ingresos/1e6:.1f}M",
              f"{crec_mes:+.1f}% vs mes ant." if crec_mes else None)
    k2.metric("🧾 Facturas", f"{n_fact:,}")
    k3.metric("🎫 Ticket promedio", f"${ticket_prom:,.0f}")
    k4.metric("🍱 Unidades vendidas", f"{int(unidades):,}")
    k5.metric("📈 Margen bruto", f"{margen_pct:.1f}%", f"${margen/1e6:+.1f}M")

    st.markdown("---")

    # ── Selector granularidad
    st.markdown("### Ingresos vs costos")
    g_ini = st.radio("Granularidad", ["Día", "Semana", "Mes"], horizontal=True,
                     index=2, key="g_resumen")

    ing_agrup = agrupar_por_granularidad(df, "Fecha", "Total", g_ini)
    cos_agrup = agrupar_por_granularidad(costos_f, "Fecha", "Valor", g_ini) if not costos_f.empty else pd.DataFrame(columns=["Periodo", "Valor"])

    merged = ing_agrup.rename(columns={"Total": "Ingresos"}).merge(
        cos_agrup.rename(columns={"Valor": "Costos"}),
        on="Periodo", how="outer"
    ).fillna(0).sort_values("Periodo")
    merged["Margen"] = merged["Ingresos"] - merged["Costos"]

    fig = go.Figure()
    fig.add_bar(x=merged["Periodo"], y=merged["Ingresos"] / 1e6,
                name="Ingresos", marker_color=VERDE)
    fig.add_bar(x=merged["Periodo"], y=merged["Costos"] / 1e6,
                name="Costos", marker_color=AMARILLO)
    fig.add_scatter(x=merged["Periodo"], y=merged["Margen"] / 1e6,
                    name="Margen bruto", mode="lines+markers",
                    line=dict(color=VERDE_OSCURO, width=2.5),
                    marker=dict(size=7))
    fig.update_layout(
        barmode="group", height=400,
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

    st.caption(
        f"ℹ️ Costos solo incluyen transacciones tipo **Compra/Gasto** + **Documento soporte** "
        f"(se excluyen 'Pagos' para evitar doble conteo de facturas ya registradas)."
    )


# ═══════════════════════════════════════════════════════════════════
# TAB: HISTORICO DE VENTAS (nuevo)
# ═══════════════════════════════════════════════════════════════════
with tab_historico:
    st.markdown("## 📈 Ventas históricas")
    st.markdown(
        f"<div class='caja-info'>Explora las ventas agrupadas por día, semana o mes. "
        f"Puedes separar por tipo de plato para ver qué categoría jala el negocio.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        gran = st.radio("Granularidad", ["Día", "Semana", "Mes"],
                        horizontal=True, key="g_hist")
    with c2:
        metrica = st.radio("Ver", ["Ingresos", "Unidades", "Facturas"],
                           horizontal=True, key="m_hist")
    with c3:
        agrupar_por_tipo = st.checkbox("Agrupar por tipo de plato", value=False)

    # Seleccionar columna y agregacion
    col_map = {"Ingresos": ("Total", "sum", "Millones COP", 1e6),
               "Unidades": ("Cantidad", "sum", "Unidades", 1),
               "Facturas": ("Consecutivo", "nunique", "Facturas", 1)}
    col, agg, ylabel, divisor = col_map[metrica]

    if agrupar_por_tipo:
        datos = agrupar_por_granularidad(df, "Fecha", col, gran, por="TipoPlato", agg=agg)
        datos["Valor"] = datos[col] / divisor
        fig = px.bar(
            datos, x="Periodo", y="Valor", color="TipoPlato",
            color_discrete_sequence=PALETA_CATEGORIAS,
            labels={"Valor": ylabel, "Periodo": "", "TipoPlato": "Tipo"},
        )
        fig.update_layout(
            barmode="stack", height=450,
            plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            font=dict(color=CARBON),
            margin=dict(t=20, b=40, l=60, r=20),
            xaxis=dict(tickangle=-45, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
        )
    else:
        datos = agrupar_por_granularidad(df, "Fecha", col, gran, agg=agg)
        datos["Valor"] = datos[col] / divisor
        fig = go.Figure()
        fig.add_bar(x=datos["Periodo"], y=datos["Valor"],
                    marker_color=VERDE, name=metrica)
        if len(datos) > 3:
            z = np.polyfit(range(len(datos)), datos["Valor"], 1)
            trend = np.poly1d(z)(range(len(datos)))
            fig.add_scatter(x=datos["Periodo"], y=trend, name="Tendencia",
                            mode="lines", line=dict(color=AMARILLO, width=2, dash="dash"))
        fig.update_layout(
            height=450, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            font=dict(color=CARBON),
            yaxis_title=ylabel,
            margin=dict(t=20, b=40, l=60, r=20),
            xaxis=dict(tickangle=-45, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
            legend=dict(orientation="h", y=1.05, x=0),
            hovermode="x unified",
        )
    st.plotly_chart(fig, use_container_width=True)

    # Resumen por tipo de plato
    st.markdown("### 🥘 Resumen por tipo de plato")
    resumen_tipo = (df.groupby("TipoPlato")
                    .agg(Ingresos=("Total", "sum"),
                         Unidades=("Cantidad", "sum"),
                         Facturas=("Consecutivo", "nunique"))
                    .reset_index().sort_values("Ingresos", ascending=False))
    resumen_tipo["Participación"] = resumen_tipo["Ingresos"] / resumen_tipo["Ingresos"].sum() * 100
    resumen_tipo["TicketProm"] = resumen_tipo["Ingresos"] / resumen_tipo["Facturas"].replace(0, np.nan)

    r1, r2 = st.columns([3, 2])
    with r1:
        tabla = resumen_tipo.copy()
        tabla["Ingresos"] = tabla["Ingresos"].apply(lambda x: f"${x:,.0f}")
        tabla["Unidades"] = tabla["Unidades"].astype(int).apply(lambda x: f"{x:,}")
        tabla["Facturas"] = tabla["Facturas"].apply(lambda x: f"{x:,}")
        tabla["Participación"] = tabla["Participación"].apply(lambda x: f"{x:.1f}%")
        tabla["TicketProm"] = tabla["TicketProm"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
        st.dataframe(tabla.rename(columns={"TipoPlato": "Tipo"}),
                     use_container_width=True, hide_index=True, height=340)
    with r2:
        fig_pie = px.pie(resumen_tipo, values="Ingresos", names="TipoPlato",
                         color_discrete_sequence=PALETA_CATEGORIAS, hole=0.45,
                         title="Participación en ingresos")
        fig_pie.update_layout(height=340, margin=dict(t=40, b=10, l=0, r=0),
                              plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                              font=dict(color=CARBON),
                              title_font=dict(color=VERDE_OSCURO, size=14),
                              legend=dict(font=dict(size=10)))
        fig_pie.update_traces(textinfo="percent+label",
                              hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
        st.plotly_chart(fig_pie, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB: PRECISION DEL MODELO
# ═══════════════════════════════════════════════════════════════════
with tab_precision:
    st.markdown("## 📉 Precisión de los modelos por producto")
    st.markdown(
        f"<div class='caja-info'>"
        f"Comparamos el rendimiento de <b>Prophet</b> (largo plazo) vs <b>Modelo Híbrido</b> (XGBoost). "
        f"El sistema escoge automáticamente el modelo con el mejor MAE en validación (Walk-Forward). "
        f"Aquí se muestran todas las métricas evaluadas (RMSE, MAPE, SMAPE)."
        f"</div>",
        unsafe_allow_html=True,
    )

    if preds["resumen"] is None:
        st.info("Genera las predicciones para ver las métricas.")
    else:
        df_res = preds["resumen"]

        col1, col2 = st.columns(2)
        with col1:
            mejores_hibrido = (df_res["modelo_usado"] == "Híbrido").sum()
            st.metric("🏆 Platos donde ganó Híbrido", f"{mejores_hibrido} de {len(df_res)}")
        with col2:
            mejores_prophet = (df_res["modelo_usado"] == "Prophet").sum()
            st.metric("🏆 Platos donde ganó Prophet", f"{mejores_prophet} de {len(df_res)}")

        st.markdown("### 📊 Comparativa MAE (Error Absoluto Medio)")

        # Grafico comparativo de MAE
        df_res_sorted = df_res.sort_values("mae_hibrido")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Prophet puro", x=df_res_sorted["producto"], y=df_res_sorted["mae_prophet"],
            marker_color="#8F8C85"
        ))
        fig.add_trace(go.Bar(
            name="Híbrido (Prophet+XGB)", x=df_res_sorted["producto"], y=df_res_sorted["mae_hibrido"],
            marker_color=AMARILLO
        ))

        # Add asterisks to the winner
        winner_text = ["★" if row["modelo_usado"] == "Híbrido" else "" for _, row in df_res_sorted.iterrows()]
        fig.add_trace(go.Scatter(
            x=df_res_sorted["producto"], y=df_res_sorted["mae_hibrido"] + 1,
            mode="text", text=winner_text, textposition="top center",
            textfont=dict(color=VERDE_OSCURO, size=16),
            showlegend=False, hoverinfo="none"
        ))

        fig.update_layout(
            barmode="group",
            height=400, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            title="MAE por producto (★ = Híbrido fue mejor)",
            title_font=dict(color=VERDE_OSCURO), font=dict(color=CARBON),
            legend=dict(orientation="h", y=1.1, x=0),
            yaxis=dict(title="Unidades de error", showgrid=True, gridcolor="#E0DDD3"),
            margin=dict(t=50, b=40, l=10, r=10),
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧮 Tabla Completa de Métricas")
        mostrar_cols = [
            "producto", "modelo_usado",
            "mae_prophet", "mae_hibrido", "mejora_mae",
            "rmse_hibrido", "smape_hibrido", "sesgo_hibrido"
        ]

        # Format table
        tabla_metricas = df_res_sorted[mostrar_cols].copy()
        for col in ["mae_prophet", "mae_hibrido", "mejora_mae", "rmse_hibrido", "sesgo_hibrido"]:
            if col in tabla_metricas:
                tabla_metricas[col] = tabla_metricas[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")

        if "smape_hibrido" in tabla_metricas:
            tabla_metricas["smape_hibrido"] = tabla_metricas["smape_hibrido"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")

        tabla_metricas.columns = [
            "Plato", "Modelo Elegido",
            "MAE Prophet", "MAE Híbrido", "Mejora MAE",
            "RMSE Híbrido", "SMAPE Híbrido", "Sesgo Híbrido"
        ]

        st.dataframe(tabla_metricas, use_container_width=True, hide_index=True)

        st.info("💡 **RMSE** castiga errores grandes (picos). **SMAPE** mide error porcentual. **Sesgo** negativo significa que tiende a subestimar.")
# ═══════════════════════════════════════════════════════════════════
# TAB: PRODUCTOS
# ═══════════════════════════════════════════════════════════════════
with tab_productos:
    st.markdown("## 🏆 Productos más vendidos")
    n_top = st.slider("Cantidad de productos a mostrar", 5, 25, 12)

    top = (df.groupby(["Nombre", "TipoPlato"])
           .agg(Ingresos=("Total", "sum"), Unidades=("Cantidad", "sum"),
                Facturas=("Consecutivo", "nunique"),
                Precio_prom=("Precio", "mean"))
           .reset_index().sort_values("Ingresos", ascending=False).head(n_top))

    t1, t2 = st.columns(2)
    with t1:
        fig = px.bar(top.sort_values("Ingresos"), x="Ingresos", y="Nombre",
                     orientation="h", color="TipoPlato",
                     color_discrete_sequence=PALETA_CATEGORIAS,
                     title="Top por ingresos",
                     text=top.sort_values("Ingresos")["Ingresos"].apply(lambda x: f"${x/1e6:.2f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=440, margin=dict(t=40, b=10, l=10, r=80),
                          plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                          title_font=dict(color=VERDE_OSCURO),
                          font=dict(color=CARBON),
                          xaxis=dict(tickformat="$,.0f", showgrid=True, gridcolor="#E0DDD3"),
                          yaxis=dict(title=""),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig = px.bar(top.sort_values("Unidades"), x="Unidades", y="Nombre",
                     orientation="h", color="TipoPlato",
                     color_discrete_sequence=PALETA_CATEGORIAS,
                     title="Top por unidades",
                     text=top.sort_values("Unidades")["Unidades"].astype(int))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=440, margin=dict(t=40, b=10, l=10, r=60),
                          plot_bgcolor=CREMA, paper_bgcolor=CREMA,
                          title_font=dict(color=VERDE_OSCURO),
                          font=dict(color=CARBON),
                          xaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
                          yaxis=dict(title=""),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Concentración de ingresos (regla 80/20)")
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
        tabla = (df.groupby(["Nombre", "TipoPlato"])
                 .agg(Ingresos=("Total", "sum"), Unidades=("Cantidad", "sum"),
                      Facturas=("Consecutivo", "nunique"),
                      Precio_prom=("Precio", "mean"))
                 .reset_index().sort_values("Ingresos", ascending=False))
        tabla["%"] = (tabla["Ingresos"] / tabla["Ingresos"].sum() * 100).round(1)
        tabla["Ingresos"] = tabla["Ingresos"].apply(lambda x: f"${x:,.0f}")
        tabla["Precio_prom"] = tabla["Precio_prom"].apply(lambda x: f"${x:,.0f}")
        tabla["%"] = tabla["%"].apply(lambda x: f"{x}%")
        st.dataframe(tabla, use_container_width=True, height=450, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB: PATRONES
# ═══════════════════════════════════════════════════════════════════
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
                    marker_color=VERDE,
                    text=vdia["Ingresos"].apply(lambda x: f"${x/1e6:.1f}M"),
                    textposition="outside")
        fig.update_layout(
            title="Ingresos por día de la semana",
            height=330, plot_bgcolor=CREMA, paper_bgcolor=CREMA,
            yaxis_title="Millones COP",
            font=dict(color=CARBON), title_font=dict(color=VERDE_OSCURO),
            margin=dict(t=40, b=10, l=60, r=10),
            yaxis=dict(showgrid=True, gridcolor="#E0DDD3"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ Sábado: operación mínima (estudiantes no van).")

    with c2:
        pivot = (df.groupby(["DiaSemana", "Mes"])["Total"]
                 .sum().unstack(fill_value=0))
        pivot = pivot.reindex([d for d in ORDEN_DIAS if d in pivot.index])
        pivot.index = [NOMBRES[d] for d in pivot.index]
        fig = px.imshow(
            pivot / 1e3, aspect="auto",
            color_continuous_scale=[[0, CREMA], [0.5, VERDE_CLARO], [1, VERDE_OSCURO]],
            title="Calor: día × mes (miles COP)",
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

    st.markdown("### Ingresos por establecimiento")
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


# ═══════════════════════════════════════════════════════════════════
# TAB: COMPRAS
# ═══════════════════════════════════════════════════════════════════
with tab_compras:
    st.markdown("## 🛒 Compras y proveedores")
    st.caption(
        f"ℹ️ Solo transacciones tipo **Compra/Gasto** + **Documento soporte** "
        f"(se excluyen 'Pagos' para evitar doble conteo)."
    )

    if costos_f.empty:
        st.info("No hay datos de compras en el periodo seleccionado.")
    else:
        g_c = st.radio("Granularidad", ["Día", "Semana", "Mes"],
                       horizontal=True, index=2, key="g_compras")
        dc_agrup = agrupar_por_granularidad(costos_f, "Fecha", "Valor", g_c)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                dc_agrup, x="Periodo", y="Valor",
                color_discrete_sequence=[AMARILLO],
                title=f"Gastos por {g_c.lower()}",
                text=dc_agrup["Valor"].apply(lambda x: f"${x/1e6:.1f}M"),
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
            if "Proveedor" in costos_f.columns:
                top_prov = (costos_f.groupby("Proveedor")["Valor"].sum()
                            .reset_index().sort_values("Valor", ascending=False).head(10))
                fig = px.bar(
                    top_prov.sort_values("Valor"), x="Valor", y="Proveedor",
                    orientation="h",
                    color_discrete_sequence=[VERDE],
                    title="Top 10 proveedores (acumulado)",
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
