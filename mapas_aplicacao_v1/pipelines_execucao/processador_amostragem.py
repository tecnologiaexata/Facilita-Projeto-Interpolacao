"""
processador_amostragem.py

Mapas de Aplicação — Passo 2 (ingestão + validação de amostras de solo)

Requisitos do projeto:
- Entrada: tabela de solo em CSV ou vetor (SHP/GPKG)
- Mínimo obrigatório: colunas 'Data' e 'Talhão' (padrão da plataforma)
- Coordenadas podem vir como:
  - lon/lat (graus)  -> CRS 4326
  - x/y em UTM (m)   -> CRS = crs_out do job
  (decisão automática por faixa de valores)
- Saída: GeoDataFrame em CRS UTM (crs_out) + lista de atributos numéricos candidatos
- Validação: pontos devem estar dentro do contorno (com tolerância opcional via buffer)

Obs:
- Exige coordenadas
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


@dataclass(frozen=True)
class AmostragemOutput:
    gdf_utm: gpd.GeoDataFrame
    atributos_numericos: List[str]
    field_name_from_table: str
    data_from_table: str


# ----------------------------
# helpers
# ----------------------------
def _norm_col(s: str) -> str:
    """Normaliza nome de coluna (minúsculo, sem espaços repetidos)."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Encontra coluna por match case-insensitive (exato), priorizando candidatos na ordem."""
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _read_csv_flexible(path: str) -> pd.DataFrame:
    """Lê CSV com fallback de encoding e separador."""
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    last_err = None
    for enc in encodings:
        try:
            # sep=None + engine python -> inferência de separador
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"Falha ao ler CSV ({path}). Último erro: {last_err}")


def _coords_look_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    """Heurística robusta para decidir se x/y parecem lon/lat (graus)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return False

    # usa percentis pra ignorar outliers óbvios
    x_lo, x_hi = np.nanpercentile(x, [2, 98])
    y_lo, y_hi = np.nanpercentile(y, [2, 98])

    return (abs(x_lo) <= 180 and abs(x_hi) <= 180) and (abs(y_lo) <= 90 and abs(y_hi) <= 90)


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Converte strings numéricas com vírgula/ponto para float, coerce erros para NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    s2 = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def _select_numeric_attributes(
    gdf: gpd.GeoDataFrame,
    exclude_cols: List[str],
    min_valid_fraction: float = 0.05,
) -> List[str]:
    """Seleciona colunas numéricas candidatas para interpolação."""
    attrs = []
    for c in gdf.columns:
        if c in exclude_cols or c == "geometry":
            continue
        ser = _coerce_numeric_series(gdf[c])
        valid_frac = ser.notna().mean() if len(ser) else 0.0
        if valid_frac >= min_valid_fraction:
            attrs.append(c)
            gdf[c] = ser  # atualiza para numérico
    return attrs


# ----------------------------
# API do módulo
# ----------------------------
def processar_amostragem(
    table_path: str,
    contorno_utm: gpd.GeoDataFrame,
    crs_out: str,
    logger=None,
    enforce_within: bool = True,
    within_tolerance_m: float = 0.0,
    # nomes preferidos para coordenadas
    x_candidates: Optional[List[str]] = None,
    y_candidates: Optional[List[str]] = None,
) -> AmostragemOutput:
    """
    Processa amostragem (CSV ou vetor) e retorna GeoDataFrame em crs_out.

    Parameters
    ----------
    table_path : str
        Caminho da tabela (CSV) ou vetor (SHP/GPKG).
    contorno_utm : GeoDataFrame
        Contorno em UTM (crs_out).
    crs_out : str
        CRS UTM da lavoura (ex.: 'EPSG:32722').
    enforce_within : bool
        Se True, falha caso existam pontos fora do contorno (considerando tolerância).
    within_tolerance_m : float
        Se >0, aplica buffer positivo ao contorno antes de testar "within".
        Útil para evitar falso negativo na borda por precisão numérica.
    """

    def _log(level: str, msg: str):
        if logger is None:
            print(f"[{level}] {msg}")
        else:
            fn = getattr(logger, level.lower(), None)
            if callable(fn):
                fn(msg)
            else:
                logger.info(msg)

    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Tabela de solo não encontrada: {table_path}")

    x_candidates = x_candidates or ["x", "lon", "long", "longitude", "easting"]
    y_candidates = y_candidates or ["y", "lat", "latitude", "northing"]

    ext = os.path.splitext(table_path)[1].lower()

    # ------------------ read ------------------
    if ext in [".csv", ".txt"]:
        _log("INFO", f"Lendo CSV: {table_path}")
        df = _read_csv_flexible(table_path)
        df.columns = [_norm_col(c) for c in df.columns]
    else:
        _log("INFO", f"Lendo vetor: {table_path}")
        gdf_in = gpd.read_file(table_path)
        if gdf_in.empty:
            raise ValueError("Arquivo vetorial lido, porém sem feições.")
        # se já vier com geometria, transforma para df padrão
        df = gdf_in.copy()
        df.columns = [_norm_col(c) for c in df.columns]

    # ------------------ required columns ------------------
    col_data = _find_col(df, ["Data"])
    col_talhao = _find_col(df, ["Talhão", "Talhao"])

    if col_data is None:
        raise ValueError("Coluna obrigatória não encontrada: 'Data'")
    if col_talhao is None:
        raise ValueError("Coluna obrigatória não encontrada: 'Talhão'")

    field_name = str(df[col_talhao].dropna().iloc[0]) if df[col_talhao].notna().any() else ""
    data_val = str(df[col_data].dropna().iloc[0]) if df[col_data].notna().any() else ""

    # ------------------ build GeoDataFrame ------------------
    if "geometry" in df.columns and isinstance(df, gpd.GeoDataFrame):
        gdf = df
        if gdf.crs is None:
            # se vier sem CRS, assumimos que já está em crs_out (mais seguro para shp/gpkg local)
            _log("WARNING", f"Vetor sem CRS. Assumindo CRS de saída: {crs_out}")
            gdf = gdf.set_crs(crs_out)
        gdf_utm = gdf.to_crs(crs_out)

    else:
        # CSV (sem geometry)
        col_x = _find_col(df, x_candidates)
        col_y = _find_col(df, y_candidates)
        if col_x is None or col_y is None:
            raise ValueError(
                "Coordenadas não encontradas no CSV. "
                "Esperado colunas tipo x/y ou lon/lat."
            )

        x_raw = _coerce_numeric_series(df[col_x])
        y_raw = _coerce_numeric_series(df[col_y])

        if x_raw.isna().all() or y_raw.isna().all():
            raise ValueError("Coordenadas inválidas: x/y não possuem valores numéricos.")

        is_lonlat = _coords_look_like_lonlat(x_raw.values, y_raw.values)

        if is_lonlat:
            _log("INFO", "Coordenadas detectadas como lon/lat (EPSG:4326).")
            gdf = gpd.GeoDataFrame(
                df.copy(),
                geometry=gpd.points_from_xy(x_raw, y_raw),
                crs="EPSG:4326",
            )
            gdf_utm = gdf.to_crs(crs_out)
        else:
            _log("INFO", f"Coordenadas detectadas como UTM (assumindo CRS={crs_out}).")
            gdf = gpd.GeoDataFrame(
                df.copy(),
                geometry=gpd.points_from_xy(x_raw, y_raw),
                crs=crs_out,
            )
            gdf_utm = gdf

    # ------------------ within contorno ------------------
    if contorno_utm is None or contorno_utm.empty:
        raise ValueError("contorno_utm vazio/nulo. Forneça o contorno em UTM.")

    geom_contorno = contorno_utm.geometry.unary_union
    if within_tolerance_m and within_tolerance_m > 0:
        geom_contorno = geom_contorno.buffer(float(within_tolerance_m))

    if enforce_within:
        inside = gdf_utm.geometry.within(geom_contorno)
        n_out = int((~inside).sum())
        if n_out > 0:
            # pega alguns índices para facilitar debug
            idxs = gdf_utm.index[~inside].tolist()[:10]
            raise ValueError(
                f"{n_out} ponto(s) fora do contorno (tol={within_tolerance_m} m). "
                f"Exemplos de índices: {idxs}"
            )

    # ------------------ numeric attributes ------------------
    # excluir colunas de coord + IDs e campos padrão que não entram como variável espacial
    exclude = set()
    exclude.update([col_data, col_talhao])
    exclude.update([c for c in df.columns if c.lower() in ("x", "y", "lon", "lat", "longitude", "latitude")])
    # outros campos “provavelmente metadata” (não obrigatórios, mas comuns)
    for meta in ["Cliente", "Fazenda", "Gleba", "Ponto", "Profundidade", "Camada"]:
        c = _find_col(gdf_utm, [meta])
        if c:
            exclude.add(c)

    attrs = _select_numeric_attributes(gdf_utm, exclude_cols=list(exclude), min_valid_fraction=0.05)
    _log("INFO", f"Atributos numéricos candidatos: {attrs}")

    return AmostragemOutput(
        gdf_utm=gdf_utm,
        atributos_numericos=attrs,
        field_name_from_table=field_name,
        data_from_table=data_val,
    )
