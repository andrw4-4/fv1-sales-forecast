# -*- coding: utf-8 -*-
"""
Script: entrena el pipeline hibrido para los top-10 productos y guarda:
  data/predicciones/predicciones_top10.parquet       — prediccion semana 1 (resumen)
  data/predicciones/predicciones_4_semanas.parquet   — prediccion para sem 1..4 con confianza
  data/predicciones/historial_walkforward.parquet    — real vs pred del test (grafico)
  data/predicciones/precios_unitarios.parquet        — precio promedio por producto
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.pipeline import pipeline_producto
from models.vacaciones import construir_vacaciones



import re

def limpiar_nombre_final(text):
    if pd.isna(text): return text

    patrones_a_remover = [
        r'\(.*\)', r'\d+k', r'\d+oz',
        r'Mediana', r'Mediano', r'Grande', r'Veggie',
        r'400ml', r'600ml', r'250ml', r'450ml'
    ]

    temp_name = text
    for patron in patrones_a_remover:
        temp_name = re.sub(patron, '', temp_name, flags=re.IGNORECASE)

    temp_name = " ".join(temp_name.split()).strip()

    mapeo_especifico = {
        'Hatsu amarillo': 'Hatsu Amarillo',
        'Chocolate caliente': 'Chocolate Caliente',
        'Pastel de pollo': 'Pastel de Pollo',
        'Agua cristal': 'Agua sin Gas',
        'Agua sin gas cristal': 'Agua sin Gas',
        'Agua con gas cristal': 'Agua con Gas',
        'Soda rosada': 'Soda Hatsu Rosada',
        'Parfaits': 'Parfait Frutos Rojos',
        'Parfaits frutos rojos': 'Parfait Frutos Rojos'
    }

    nombre_limpio = mapeo_especifico.get(temp_name, temp_name)
    return nombre_limpio if len(nombre_limpio) > 0 else text


def obtener_clase_A(ventas):
    """Calcula el 80% de ventas y transacciones dinamicamente."""
    # Filtrar solo 'Principal' y desde '2024-01-01' como en el notebook
    ventas = ventas[
        (ventas["Establecimiento"] == "Principal") &
        (ventas["Fecha"] >= "2024-01-01")
    ].copy()

    info = ventas.groupby("Nombre").agg(
        transacciones=("Cantidad", "count"),
        cantidad_total=("Cantidad", "sum"),
        plata_generada=("Total", "sum")
    ).reset_index()

    # 1. Filtro Estrella (Matriz BCG)
    corte_x = info["cantidad_total"].median()
    corte_y = info["plata_generada"].median()

    info["estrella"] = (info["cantidad_total"] >= corte_x) & (info["plata_generada"] >= corte_y)
    info = info[info["estrella"]].copy()

    # Clase A Ventas
    info_v = info.sort_values("plata_generada", ascending=False).copy()
    info_v["pct_plata"] = info_v["plata_generada"] / info_v["plata_generada"].sum()
    info_v["pct_acum_plata"] = info_v["pct_plata"].cumsum()
    clase_a_ventas = info_v[info_v["pct_acum_plata"] - info_v["pct_plata"] < 0.80]["Nombre"].tolist()

    # Clase A Transacciones
    info_t = info.sort_values("transacciones", ascending=False).copy()
    info_t["pct_trans"] = info_t["transacciones"] / info_t["transacciones"].sum()
    info_t["pct_acum_trans"] = info_t["pct_trans"].cumsum()
    clase_a_trans = info_t[info_t["pct_acum_trans"] - info_t["pct_trans"] < 0.80]["Nombre"].tolist()

    return list(set(clase_a_ventas) & set(clase_a_trans))


def cargar_ventas() -> pd.DataFrame:
    v = pd.read_csv(ROOT / "data" / "raw" / "ventas.csv",
                    encoding="utf-8", header=0)
    v.columns = ["Consecutivo", "Fecha", "Tipo_reg", "Tipo_clas", "Codigo",
                 "Nombre", "Vendedor", "Cantidad", "Precio", "Impuesto",
                 "Total", "Forma_pago", "Num_comp", "Establecimiento"]
    v["Fecha"] = pd.to_datetime(v["Fecha"], errors="coerce")
    v = v.dropna(subset=["Fecha"])
    v["Cantidad"] = pd.to_numeric(v["Cantidad"], errors="coerce").fillna(0)
    v["Precio"] = pd.to_numeric(v["Precio"], errors="coerce").fillna(0)
    v["Total"] = pd.to_numeric(v["Total"], errors="coerce")
    v["Total"] = v["Total"].fillna(v["Cantidad"] * v["Precio"])

    v["Nombre"] = v["Nombre"].apply(limpiar_nombre_final)
    return v



def calcular_precios(ventas: pd.DataFrame, productos: list[str]) -> pd.DataFrame:
    """Precio promedio ponderado por producto (ultimos 90 dias)."""
    fecha_corte = ventas["Fecha"].max() - pd.Timedelta(days=90)
    recientes = ventas[ventas["Fecha"] >= fecha_corte]
    filas = []
    for p in productos:
        df_p = recientes[recientes["Nombre"] == p]
        if len(df_p) == 0:
            df_p = ventas[ventas["Nombre"] == p]
        if len(df_p) == 0:
            filas.append({"producto": p, "precio_unitario": 0.0})
            continue
        # precio ponderado por cantidad vendida
        precio = (df_p["Precio"] * df_p["Cantidad"]).sum() / max(df_p["Cantidad"].sum(), 1)
        filas.append({"producto": p, "precio_unitario": float(round(precio, 0))})
    return pd.DataFrame(filas)


def main(n_trials_prophet: int = 20, n_trials_xgb: int = 30):
    vacaciones = construir_vacaciones()
    ventas = cargar_ventas()

    productos_a_correr = obtener_clase_A(ventas)

    resumen = []
    preds_4sem = []
    historial_por_producto = {}
    start = time.time()

    for i, producto in enumerate(productos_a_correr, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(productos_a_correr)}] {producto}")
        try:
            ventas_p = ventas
            out = pipeline_producto(
                ventas_p, producto, vacaciones,
                n_trials_prophet=n_trials_prophet,
                n_trials_xgb=n_trials_xgb,
            )
            if "error" in out:
                print(f"  SKIP: {out['error']}")
                continue
            dur = time.time() - t0
            print(f"  MAE Hibrido: {out['mae_test_walkforward']:.2f} | SMAPE: {out.get('smape_hibrido', 0):.1f}% "
                  f"(Prophet solo: {out['mae_test_prophet_solo']:.2f})  "
                  f"[+{out['mejora_mae']:.2f}]  "
                  f"Proxima semana: {out['prediccion_proxima_semana']}  "
                  f"[{dur:.0f}s]")

            resumen.append({
                "producto": out["producto"],
                "n_semanas": out["n_semanas"],
                "fecha_proxima_semana": out["fecha_proxima_semana"],
                "prediccion": out["prediccion_proxima_semana"],
                "prophet_solo": out["prophet_proxima_semana"],
                "mae_hibrido": out["mae_test_walkforward"],
                "smape_hibrido": out.get("smape_hibrido", 0),
                "mae_prophet": out["mae_test_prophet_solo"],
                "mejora_mae": out["mejora_mae"],
                "std_residual": out.get("std_residual_test", 0),
            })

            # Predicciones 4 semanas
            for pf in out.get("predicciones_4_semanas", []):
                preds_4sem.append({
                    "producto": out["producto"],
                    "semana_offset": pf["semana_offset"],
                    "fecha": pf["fecha"],
                    "prediccion": pf["prediccion"],
                    "prediccion_lower": pf["prediccion_lower"],
                    "prediccion_upper": pf["prediccion_upper"],
                    "prophet_solo": pf["prophet_solo"],
                    "confianza_pct": pf["confianza_pct"],
                })

            historial_por_producto[out["producto"]] = out["historial_test"]
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    # ── Precios unitarios (para estimar ingresos)
    productos_para_precio = [r["producto"] for r in resumen]
    df_precios = []
    for p in productos_para_precio:
        sub = ventas[ventas["Nombre"] == p]
        precio = ((sub["Precio"] * sub["Cantidad"]).sum()
                  / max(sub["Cantidad"].sum(), 1)) if len(sub) else 0
        df_precios.append({"producto": p, "precio_unitario": float(round(precio, 0))})
    df_precios = pd.DataFrame(df_precios)

    # ── Guardar
    df_resumen = pd.DataFrame(resumen)
    df_preds_4 = pd.DataFrame(preds_4sem)

    out_dir = ROOT / "data" / "predicciones"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_resumen.to_parquet(out_dir / "predicciones_top10.parquet")
    df_preds_4.to_parquet(out_dir / "predicciones_4_semanas.parquet")
    df_precios.to_parquet(out_dir / "precios_unitarios.parquet")

    if historial_por_producto:
        hist = pd.concat(
            [h.assign(producto=p) for p, h in historial_por_producto.items()],
            ignore_index=True,
        )
        hist.to_parquet(out_dir / "historial_walkforward.parquet")
    else:
        print("⚠️  No hay historial que guardar (todos los productos fallaron).")

    print(f"\n{'=' * 60}")
    print(f"Tiempo total: {(time.time() - start) / 60:.1f} min")
    print(f"Guardado en: {out_dir}")
    print("\n── Resumen semana 1 ──")
    print(df_resumen.to_string(index=False))
    print("\n── Precios unitarios ──")
    print(df_precios.to_string(index=False))

    # ── Metadata (timestamp del último reentrenamiento)
    import json
    from datetime import datetime
    metadata = {
        "ultima_actualizacion": datetime.now().isoformat(timespec="seconds"),
        "n_productos": len(resumen),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-prophet", type=int, default=20)
    parser.add_argument("--trials-xgb", type=int, default=30)
    args = parser.parse_args()
    main(args.trials_prophet, args.trials_xgb)
