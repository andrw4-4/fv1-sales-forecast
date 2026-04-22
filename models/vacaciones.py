# -*- coding: utf-8 -*-
"""Calendario academico Uniandes + cierre remodelacion 2024."""
import pandas as pd


def _h(holiday, ini, fin):
    return pd.DataFrame({
        "holiday": holiday,
        "ds": pd.date_range(ini, fin, freq="D"),
        "lower_window": 0, "upper_window": 0,
    })


def construir_vacaciones() -> pd.DataFrame:
    bloques = [
        # 2024
        _h("semana_santa",           "2024-03-24", "2024-03-30"),
        _h("semana_receso_verano",   "2024-03-18", "2024-03-23"),
        _h("examenes_finales",       "2024-05-27", "2024-06-01"),
        _h("vacaciones_verano",      "2024-06-02", "2024-08-04"),
        _h("semana_receso_invierno", "2024-09-30", "2024-10-05"),
        _h("examenes_finales",       "2024-12-02", "2024-12-07"),
        _h("vacaciones_invierno",    "2024-12-08", "2025-01-20"),
        # 2025
        _h("semana_receso_verano",   "2025-03-17", "2025-03-22"),
        _h("semana_santa",           "2025-04-13", "2025-04-19"),
        _h("examenes_finales",       "2025-05-26", "2025-05-31"),
        _h("vacaciones_verano",      "2025-06-01", "2025-08-03"),
        _h("semana_receso_invierno", "2025-09-29", "2025-10-04"),
        _h("examenes_finales",       "2025-12-01", "2025-12-06"),
        _h("vacaciones_invierno",    "2025-12-07", "2026-01-19"),
        # 2026
        _h("semana_receso_verano",   "2026-03-16", "2026-03-21"),
        _h("semana_santa",           "2026-03-29", "2026-04-04"),
        _h("examenes_finales",       "2026-05-25", "2026-05-30"),
        _h("vacaciones_verano",      "2026-05-31", "2026-08-02"),
        _h("semana_receso_invierno", "2026-09-28", "2026-10-03"),
        _h("examenes_finales",       "2026-11-30", "2026-12-05"),
        _h("vacaciones_invierno",    "2026-12-06", "2027-01-18"),
        # Cierre remodelacion
        _h("cierre_oct_dic_2024",    "2024-10-01", "2024-12-31"),
    ]
    induccion = pd.DataFrame({
        "holiday": "induccion_pregrado",
        "ds": pd.to_datetime([
            "2024-01-17", "2024-01-18", "2024-01-19",
            "2024-07-31", "2024-08-01", "2024-08-02",
            "2025-01-15", "2025-01-16", "2025-01-17",
            "2025-07-30", "2025-07-31", "2025-08-01",
            "2026-01-13", "2026-01-14", "2026-01-15",
            "2026-07-28", "2026-07-29", "2026-07-30",
        ]),
        "lower_window": 0, "upper_window": 0,
    })
    return pd.concat(bloques + [induccion]).reset_index(drop=True)


HITOS_INICIO = [
    pd.Timestamp("2024-01-22"), pd.Timestamp("2024-08-05"),
    pd.Timestamp("2025-01-20"), pd.Timestamp("2025-08-04"),
    pd.Timestamp("2026-01-19"), pd.Timestamp("2026-08-03"),
]
