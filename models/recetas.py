# -*- coding: utf-8 -*-
"""Parser del Excel 'Recetas Vivo (1).xlsx' y matcher con nombres de ventas."""
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
RUTA_EXCEL = ROOT / "Recetas" / "Recetas Vivo (1).xlsx"


def _normalizar(texto: str) -> str:
    """Minusculas, sin acentos, sin espacios extra."""
    if pd.isna(texto):
        return ""
    s = str(texto).lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def cargar_recetas(ruta: Path | str | None = None) -> pd.DataFrame:
    """Devuelve DataFrame long: [categoria, plato, porcion, ingrediente, cantidad, unidad]."""
    ruta = Path(ruta) if ruta else RUTA_EXCEL
    if not ruta.exists():
        return pd.DataFrame(columns=[
            "categoria", "plato", "porcion", "ingrediente", "cantidad", "unidad"
        ])

    xls = pd.ExcelFile(ruta)
    registros = []
    for hoja in xls.sheet_names:
        df = pd.read_excel(ruta, sheet_name=hoja, header=None)
        # Asumimos columnas: col1=Porcion, col2=Receta, col3=Ingrediente, col4=Cantidad, col5=Unidad
        plato_actual = None
        porcion_actual = None
        for _, row in df.iterrows():
            porcion = row.iloc[1] if len(row) > 1 else None
            receta = row.iloc[2] if len(row) > 2 else None
            ingrediente = row.iloc[3] if len(row) > 3 else None
            cantidad = row.iloc[4] if len(row) > 4 else None
            unidad = row.iloc[5] if len(row) > 5 else None

            # Saltar encabezados
            if str(receta).strip().lower() == "receta":
                continue
            if pd.notna(receta) and str(receta).strip() not in ("", "nan"):
                plato_actual = str(receta).strip()
                porcion_actual = str(porcion).strip() if pd.notna(porcion) else None
            if plato_actual and pd.notna(ingrediente) and str(ingrediente).strip() not in ("", "nan"):
                registros.append({
                    "categoria": hoja,
                    "plato": plato_actual,
                    "porcion": porcion_actual,
                    "ingrediente": str(ingrediente).strip(),
                    "cantidad": cantidad,
                    "unidad": str(unidad).strip() if pd.notna(unidad) else "",
                })
    return pd.DataFrame(registros)


# Prefijos/sufijos que aparecen en nombres de venta pero no en nombres de receta
_PREFIJOS = ["sandwich ", "sándwich ", "bowl ", "ensalada ", "smoothie ", "pancakes "]
_SUFIJOS_TAMANO = [" mediana", " mediano", " grande", " veggie mediana",
                   " veggie grande", " (mediano)", " (grande)"]


def _limpiar_nombre_venta(nombre: str) -> str:
    s = _normalizar(nombre)
    # quitar precios tipo (21k), (18k)
    s = re.sub(r"\(\d+k\)", "", s)
    s = re.sub(r"\(.*?\)", "", s).strip()
    # quitar prefijos
    for p in _PREFIJOS:
        if s.startswith(p):
            s = s[len(p):]
            break
    return s.strip()


def _variantes_clave(s: str) -> list[str]:
    """Devuelve variantes del nombre para intentar matchear."""
    s = _normalizar(s)
    variantes = [s]
    for suf in _SUFIJOS_TAMANO:
        if s.endswith(suf):
            variantes.append(s[: -len(suf)].strip())
    # intercambiar mediano<->mediana y grande (genero)
    for a, b in [("mediano", "mediana"), ("mediana", "mediano")]:
        if a in s:
            variantes.append(s.replace(a, b))
    return list(dict.fromkeys(variantes))


def construir_mapeo(recetas_df: pd.DataFrame, nombres_venta: list[str]) -> dict[str, str]:
    """Mapea cada nombre de venta al nombre de receta mas probable. Valor None si no hay match."""
    platos_receta = recetas_df["plato"].dropna().unique().tolist()
    plato_norm = {p: _normalizar(p) for p in platos_receta}

    mapeo = {}
    for venta in nombres_venta:
        limpio = _limpiar_nombre_venta(venta)
        variantes = _variantes_clave(limpio)
        match = None
        # exact match
        for v in variantes:
            for plato, norm in plato_norm.items():
                if v == norm:
                    match = plato
                    break
            if match:
                break
        # substring match (en ambas direcciones)
        if not match:
            for v in variantes:
                for plato, norm in plato_norm.items():
                    if v and (v in norm or norm in v):
                        match = plato
                        break
                if match:
                    break
        mapeo[venta] = match
    return mapeo


def insumos_para_demanda(recetas_df: pd.DataFrame, plato_receta: str,
                         unidades: float) -> pd.DataFrame:
    """Multiplica la receta por la cantidad de unidades pronosticadas."""
    r = recetas_df[recetas_df["plato"] == plato_receta].copy()
    if r.empty:
        return pd.DataFrame(columns=["ingrediente", "cantidad", "unidad"])
    r["cantidad_num"] = pd.to_numeric(r["cantidad"], errors="coerce").fillna(0)
    r["cantidad_total"] = r["cantidad_num"] * unidades
    agg = (r.groupby(["ingrediente", "unidad"], as_index=False)["cantidad_total"]
           .sum()
           .rename(columns={"cantidad_total": "cantidad"}))
    return agg.sort_values("cantidad", ascending=False).reset_index(drop=True)
