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


TOP_10_PRODUCTOS = [
    "Arma tu plato Grande(21k)",
    "Arma tu plato mediano (18k)",
    "Sándwich Romano",
    "Bowl Pollo tostada Mediana",
    "Ensalada Chefsito Mediana",
    "Sándwich Criollo",
    "Bowl Lomo alto Mediana",
    "Bowl Pasta Buona Mediana",
    "Bowl Colombianito Mediana",
    "Ensalada Pollo miel mostaza Mediana",
]


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

    # Combinar las dos variantes de "Arma tu plato" en una sola serie
    ventas_combinadas = ventas.copy()
    mapa_arma = {
        "Arma tu plato Grande(21k)": "Arma tu plato (combinado)",
        "Arma tu plato mediano (18k)": "Arma tu plato (combinado)",
    }
    ventas_combinadas["Nombre"] = ventas_combinadas["Nombre"].replace(mapa_arma)

    productos_a_correr = [p for p in TOP_10_PRODUCTOS if p not in mapa_arma]
    productos_a_correr.insert(0, "Arma tu plato (combinado)")

    resumen = []
    preds_4sem = []
    historial_por_producto = {}
    start = time.time()

    for i, producto in enumerate(productos_a_correr, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(productos_a_correr)}] {producto}")
        try:
            ventas_p = ventas_combinadas if producto == "Arma tu plato (combinado)" else ventas
            out = pipeline_producto(
                ventas_p, producto, vacaciones,
                n_trials_prophet=n_trials_prophet,
                n_trials_xgb=n_trials_xgb,
            )
            if "error" in out:
                print(f"  SKIP: {out['error']}")
                continue
            dur = time.time() - t0
            print(f"  MAE Híbrido: {out['mae_hibrido']:.2f}  "
                  f"(Prophet: {out['mae_prophet']:.2f})  "
                  f"[Modelo: {out['modelo_usado']}]  "
                  f"Proxima semana: {out['prediccion_proxima_semana']}  "
                  f"[{dur:.0f}s]")

            resumen.append({
                "producto": out["producto"],
                "n_semanas": out["n_semanas"],
                "fecha_proxima_semana": out["fecha_proxima_semana"],
                "modelo_usado": out["modelo_usado"],
                "prediccion": out["prediccion_proxima_semana"],
                "prophet_solo": out["prophet_proxima_semana"],
                "mae_hibrido": out["mae_hibrido"],
                "mae_prophet": out["mae_prophet"],
                "mejora_mae": out["mejora_mae"],
                "rmse_hibrido": out["rmse_hibrido"],
                "rmse_prophet": out["rmse_prophet"],
                "mape_hibrido": out["mape_hibrido"],
                "mape_prophet": out["mape_prophet"],
                "smape_hibrido": out["smape_hibrido"],
                "smape_prophet": out["smape_prophet"],
                "sesgo_hibrido": out["sesgo_hibrido"],
                "sesgo_prophet": out["sesgo_prophet"],
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
    # Para la serie combinada, usar el promedio de las dos originales
    df_precios = []
    for p in productos_para_precio:
        if p == "Arma tu plato (combinado)":
            sub = ventas[ventas["Nombre"].isin(list(mapa_arma.keys()))]
            precio = ((sub["Precio"] * sub["Cantidad"]).sum()
                      / max(sub["Cantidad"].sum(), 1)) if len(sub) else 0
            df_precios.append({"producto": p, "precio_unitario": float(round(precio, 0))})
        else:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-prophet", type=int, default=20)
    parser.add_argument("--trials-xgb", type=int, default=30)
    args = parser.parse_args()
    main(args.trials_prophet, args.trials_xgb)
