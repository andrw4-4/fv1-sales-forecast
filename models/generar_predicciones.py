# -*- coding: utf-8 -*-
"""
Script: entrena el pipeline hibrido para los top-10 productos y guarda las
predicciones en data/predicciones_top10.parquet.

Uso:
    python -m models.generar_predicciones

El dashboard.py lee este parquet. Re-ejecutar cuando haya datos nuevos.
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
    return v


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

    productos_a_correr = list(TOP_10_PRODUCTOS)
    # usar la combinada en lugar de las dos originales
    productos_a_correr = [p for p in productos_a_correr if p not in mapa_arma]
    productos_a_correr.insert(0, "Arma tu plato (combinado)")

    resumen = []
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
            print(f"  MAE walk-forward: {out['mae_test_walkforward']:.2f}  "
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
                "mae_prophet": out["mae_test_prophet_solo"],
                "mejora_mae": out["mejora_mae"],
            })
            historial_por_producto[out["producto"]] = out["historial_test"]
        except Exception as exc:
            print(f"  ERROR: {exc}")

    df_resumen = pd.DataFrame(resumen)
    out_dir = ROOT / "data" / "predicciones"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_resumen.to_parquet(out_dir / "predicciones_top10.parquet")

    # historial walkforward (para graficar real vs pred en el dashboard)
    hist = pd.concat(
        [h.assign(producto=p) for p, h in historial_por_producto.items()],
        ignore_index=True,
    )
    hist.to_parquet(out_dir / "historial_walkforward.parquet")

    print(f"\n{'=' * 60}")
    print(f"Tiempo total: {(time.time() - start) / 60:.1f} min")
    print(f"Guardado en: {out_dir}")
    print(df_resumen.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-prophet", type=int, default=20)
    parser.add_argument("--trials-xgb", type=int, default=30)
    args = parser.parse_args()
    main(args.trials_prophet, args.trials_xgb)
