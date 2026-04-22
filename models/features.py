# -*- coding: utf-8 -*-
"""Feature engineering para el modelo hibrido Prophet + XGBoost."""
import numpy as np
import pandas as pd

from models.vacaciones import HITOS_INICIO


FEATURES_MODELO = [
    # Prophet
    "yhat", "yhat_lower", "yhat_upper",
    # Calendario generico
    "mes", "semana_año", "trimestre",
    "mes_sin", "mes_cos", "semana_sin", "semana_cos",
    # Calendario academico Uniandes
    "semana_academica",
    "pre_evento",
    "inicio_parciales_1", "fin_parciales_1",
    "inicio_parciales_2", "fin_parciales_2",
    "vuelta_a_clases", "post_vacaciones",
    # Historia reciente
    "lag_1", "lag_2", "lag_4",
    "rolling_4", "rolling_8",
    "lag_1_clean",
    # Vacaciones / eventos
    "hol_semana_santa", "hol_semana_receso_verano",
    "hol_examenes_finales", "hol_vacaciones_verano",
    "hol_semana_receso_invierno", "hol_vacaciones_invierno",
    "hol_cierre_oct_dic_2024", "hol_induccion_pregrado",
    "semanas_antes_verano",
]


def preparar_serie_semanal(ventas: pd.DataFrame, producto: str) -> pd.DataFrame:
    """Agrega ventas de un producto a semanas lunes-lunes, rellenando con 0."""
    df = ventas.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    f_min, f_max = df["Fecha"].min(), df["Fecha"].max()
    lunes_ini = f_min - pd.Timedelta(days=f_min.weekday())
    lunes_fin = f_max - pd.Timedelta(days=f_max.weekday())
    calendario = pd.date_range(lunes_ini, lunes_fin, freq="W-MON")

    df["ds"] = df["Fecha"].dt.floor("D") - df["Fecha"].dt.weekday.map(
        lambda x: pd.Timedelta(days=x)
    )
    df_prod = df[df["Nombre"] == producto]
    serie = df_prod.groupby("ds")["Cantidad"].sum().reset_index()
    serie = pd.merge(
        serie, pd.DataFrame({"ds": calendario}), on="ds", how="right"
    ).fillna(0)
    return serie.rename(columns={"Cantidad": "y"}).sort_values("ds").reset_index(drop=True)


def _semana_academica(ds: pd.Timestamp, mapa: dict) -> int:
    if ds in HITOS_INICIO:
        return 1
    etiqueta = mapa.get(ds, "")
    if "finales" in etiqueta:
        return 17
    if any(h in etiqueta for h in ["receso", "santa", "induccion"]):
        return 0
    if "vacaciones" in etiqueta:
        return -1
    inicio = max([h for h in HITOS_INICIO if h <= ds], default=pd.Timestamp("1900-01-01"))
    if inicio == pd.Timestamp("1900-01-01"):
        return 0
    return len([
        l for l in pd.date_range(inicio, ds, freq="W-MON")
        if not any(h in mapa.get(l, "") for h in ["receso", "santa", "finales", "induccion"])
    ])


def crear_features(serie: pd.DataFrame, vacaciones: pd.DataFrame) -> pd.DataFrame:
    df = serie.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Calendario generico
    df["mes"] = df["ds"].dt.month
    df["semana_año"] = df["ds"].dt.isocalendar().week.astype(int)
    df["trimestre"] = df["ds"].dt.quarter
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_año"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_año"] / 52)

    # Historia
    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_4"] = df["y"].shift(4)
    df["rolling_4"] = df["y"].shift(1).rolling(4).mean()
    df["rolling_8"] = df["y"].shift(1).rolling(8).mean()

    # Holidays (flags semana-contiene-holiday)
    for holiday in vacaciones["holiday"].unique():
        fechas = vacaciones.loc[vacaciones["holiday"] == holiday, "ds"].values
        fechas = pd.to_datetime(fechas)
        df[f"hol_{holiday}"] = df["ds"].apply(
            lambda x: int(((fechas >= x) & (fechas < x + pd.Timedelta(days=7))).any())
        )

    # Semanas antes de vacaciones de verano (efecto anticipacion)
    try:
        fecha_verano = vacaciones.loc[vacaciones["holiday"] == "vacaciones_verano", "ds"].min()
        df["semanas_antes_verano"] = df["ds"].apply(
            lambda x: max(0, 4 - (fecha_verano - x).days // 7) if (fecha_verano - x).days > 0 else 0
        )
    except Exception:
        df["semanas_antes_verano"] = 0

    # Semana academica Uniandes
    mapa_vacas = vacaciones.set_index("ds")["holiday"].to_dict()
    df["semana_academica"] = df["ds"].apply(lambda x: _semana_academica(x, mapa_vacas))

    df["inicio_parciales_1"] = (df["semana_academica"] == 7).astype(int)
    df["fin_parciales_1"] = (df["semana_academica"] == 8).astype(int)
    df["inicio_parciales_2"] = (df["semana_academica"] == 12).astype(int)
    df["fin_parciales_2"] = (df["semana_academica"] == 13).astype(int)
    df["pre_evento"] = (df["semana_academica"].shift(-1) == 0).astype(int)
    df["vuelta_a_clases"] = df["semana_academica"].isin([1, 2]).astype(int)
    df["post_vacaciones"] = (
        (df["semana_academica"] == 1) | (df["semana_academica"].shift(1) == -1)
    ).astype(int)

    # Lag limpio (usa la media historica por semana_academica cuando lag=0 en vuelta a clases)
    def _lag_clean(row):
        if row["lag_1"] == 0 and row["semana_academica"] in [1, 2]:
            mask = df["semana_academica"] == row["semana_academica"]
            promedio = df.loc[mask, "y"].mean()
            return promedio if pd.notna(promedio) else row["lag_1"]
        return row["lag_1"]

    df["lag_1_clean"] = df.apply(_lag_clean, axis=1)

    # Garantia: todas las columnas objetivo existen (rellenar con 0 si alguna vacacion no aparece)
    # EXCEPTO las de Prophet (yhat*) que se mergean despues.
    columnas_prophet = {"yhat", "yhat_lower", "yhat_upper"}
    for col in FEATURES_MODELO:
        if col not in df.columns and col not in columnas_prophet:
            df[col] = 0

    return df


def expandir_calendario(serie: pd.DataFrame, semanas_adicionales: int) -> pd.DataFrame:
    """Agrega filas futuras a la serie con y=NaN para poder calcular features."""
    df = serie.copy().sort_values("ds").reset_index(drop=True)
    ultima = df["ds"].max()
    futuras = pd.date_range(
        ultima + pd.Timedelta(weeks=1),
        periods=semanas_adicionales, freq="W-MON"
    )
    extra = pd.DataFrame({"ds": futuras, "y": np.nan})
    return pd.concat([df, extra], ignore_index=True)
