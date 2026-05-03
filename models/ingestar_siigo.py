# -*- coding: utf-8 -*-
"""
Descarga facturas de venta desde la API de Siigo y las agrega a data/raw/ventas.csv.

Uso:
    python -m models.ingestar_siigo                        # últimos 7 días
    python -m models.ingestar_siigo --desde 2026-04-01 --hasta 2026-04-30

Credenciales (cualquiera de las dos formas):
    - Variables de entorno: SIIGO_USERNAME, SIIGO_ACCESS_KEY
    - Archivo .env en la raíz del proyecto
    - Streamlit Secrets (cuando corre dentro del dashboard)
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.generar_predicciones import limpiar_nombre_final

# ── Columnas esperadas en ventas.csv
COLUMNAS_VENTAS = [
    "Consecutivo", "Fecha", "Tipo_reg", "Tipo_clas", "Codigo",
    "Nombre", "Vendedor", "Cantidad", "Precio", "Impuesto",
    "Total", "Forma_pago", "Num_comp", "Establecimiento",
]

TOKEN_URL = "https://siigonube.siigo.com:50050/connect/token"
API_BASE  = "https://api.siigo.com/v1"


def _cargar_credenciales():
    """Lee SIIGO_USERNAME y SIIGO_ACCESS_KEY desde .env, entorno o st.secrets."""
    # Intentar cargar .env si existe
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    username = os.environ.get("SIIGO_USERNAME")
    access_key = os.environ.get("SIIGO_ACCESS_KEY")

    # Fallback: Streamlit Secrets (si corre dentro de Streamlit)
    if not username or not access_key:
        try:
            import streamlit as st
            username = username or st.secrets.get("SIIGO_USERNAME")
            access_key = access_key or st.secrets.get("SIIGO_ACCESS_KEY")
        except Exception:
            pass

    if not username or not access_key:
        raise EnvironmentError(
            "Credenciales de Siigo no encontradas. "
            "Define SIIGO_USERNAME y SIIGO_ACCESS_KEY en el archivo .env o como variables de entorno."
        )
    return username, access_key


def obtener_token(username: str, access_key: str) -> str:
    """Autentica con Siigo (OAuth2 password grant) y retorna el Bearer token."""
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "password",
            "username": username,
            "password": access_key,
            "scope": "WebApi offline_access",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def descargar_facturas(token: str, desde: str, hasta: str) -> list[dict]:
    """
    Descarga todas las páginas de GET /v1/invoices en el rango de fechas.
    Retorna lista de dicts con los items expandidos (una entrada por línea de producto).
    """
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {
        "created_start": desde,  # yyyy-MM-dd
        "created_end":   hasta,
        "page":          1,
        "page_size":     25,
    }

    filas = []
    while True:
        resp = requests.get(f"{API_BASE}/invoices", headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        data = resp.json()
        resultados = data.get("results", [])
        if not resultados:
            break

        for factura in resultados:
            fecha     = factura.get("date", "")
            consec    = factura.get("number", "")
            vendedor  = factura.get("seller", "")
            total_fac = float(factura.get("total", 0) or 0)
            forma_pago = ""
            pagos = factura.get("payments", [])
            if pagos:
                forma_pago = str(pagos[0].get("payment_method_id", ""))
            # Establecimiento: cost_center → name, o "Principal" por defecto
            cc = factura.get("cost_center") or {}
            establecimiento = cc.get("name", "Principal") if isinstance(cc, dict) else "Principal"

            items = factura.get("items", []) or []
            if not items:
                # Factura sin items detallados: una fila con totales
                filas.append({
                    "Consecutivo":    consec,
                    "Fecha":          fecha,
                    "Tipo_reg":       "Secuencia",
                    "Tipo_clas":      "",
                    "Codigo":         "",
                    "Nombre":         "",
                    "Vendedor":       vendedor,
                    "Cantidad":       1,
                    "Precio":         total_fac,
                    "Impuesto":       0,
                    "Total":          total_fac,
                    "Forma_pago":     forma_pago,
                    "Num_comp":       consec,
                    "Establecimiento": establecimiento,
                })
                continue

            for item in items:
                cantidad = float(item.get("quantity", 1) or 1)
                precio   = float(item.get("price", 0) or 0)
                impuesto = sum(
                    float(t.get("value", 0) or 0)
                    for t in (item.get("taxes") or [])
                )
                total_item = cantidad * precio + impuesto
                filas.append({
                    "Consecutivo":    consec,
                    "Fecha":          fecha,
                    "Tipo_reg":       "Secuencia",
                    "Tipo_clas":      "",
                    "Codigo":         item.get("code", ""),
                    "Nombre":         item.get("description", ""),
                    "Vendedor":       vendedor,
                    "Cantidad":       cantidad,
                    "Precio":         precio,
                    "Impuesto":       impuesto,
                    "Total":          total_item,
                    "Forma_pago":     forma_pago,
                    "Num_comp":       consec,
                    "Establecimiento": establecimiento,
                })

        # Paginación
        pagination = data.get("pagination", {})
        if params["page"] * params["page_size"] >= pagination.get("total_results", 0):
            break
        params["page"] += 1

    return filas


def normalizar_y_guardar(filas: list[dict], ventas_path: Path) -> int:
    """
    Normaliza los campos, deduplica contra el CSV existente y hace append.
    Retorna el número de filas nuevas escritas.
    """
    if not filas:
        return 0

    nuevo = pd.DataFrame(filas, columns=COLUMNAS_VENTAS)
    nuevo["Fecha"]    = pd.to_datetime(nuevo["Fecha"], errors="coerce")
    nuevo["Cantidad"] = pd.to_numeric(nuevo["Cantidad"], errors="coerce").fillna(0)
    nuevo["Precio"]   = pd.to_numeric(nuevo["Precio"], errors="coerce").fillna(0)
    nuevo["Total"]    = pd.to_numeric(nuevo["Total"], errors="coerce").fillna(0)
    nuevo["Nombre"]   = nuevo["Nombre"].apply(limpiar_nombre_final)
    nuevo = nuevo.dropna(subset=["Fecha"])

    if ventas_path.exists():
        existente = pd.read_csv(ventas_path, encoding="utf-8", header=0)
        existente.columns = COLUMNAS_VENTAS
        existente["Consecutivo"] = existente["Consecutivo"].astype(str)
        nuevo["Consecutivo"]     = nuevo["Consecutivo"].astype(str)
        # Deduplicar: excluir filas cuyo Consecutivo ya está en el CSV
        consecutivos_existentes = set(existente["Consecutivo"].unique())
        nuevo = nuevo[~nuevo["Consecutivo"].isin(consecutivos_existentes)]

        if nuevo.empty:
            return 0

        combinado = pd.concat([existente, nuevo], ignore_index=True)
    else:
        ventas_path.parent.mkdir(parents=True, exist_ok=True)
        combinado = nuevo

    combinado.to_csv(ventas_path, index=False, encoding="utf-8")
    return len(nuevo)


def main(desde: str | None = None, hasta: str | None = None):
    if hasta is None:
        hasta = datetime.today().strftime("%Y-%m-%d")
    if desde is None:
        desde = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"Descargando facturas Siigo del {desde} al {hasta}...")

    username, access_key = _cargar_credenciales()
    token  = obtener_token(username, access_key)
    filas  = descargar_facturas(token, desde, hasta)

    print(f"  {len(filas)} líneas de producto descargadas.")

    ventas_path = ROOT / "data" / "raw" / "ventas.csv"
    n_nuevas = normalizar_y_guardar(filas, ventas_path)
    print(f"  {n_nuevas} filas nuevas agregadas a {ventas_path}.")
    return n_nuevas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de ventas desde Siigo API")
    parser.add_argument("--desde", type=str, default=None, help="Fecha inicio yyyy-MM-dd")
    parser.add_argument("--hasta", type=str, default=None, help="Fecha fin yyyy-MM-dd")
    args = parser.parse_args()
    main(args.desde, args.hasta)
