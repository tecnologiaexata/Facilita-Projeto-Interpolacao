from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

from .logger import LoggerAgricola, NivelLog


@dataclass
class ConfigFiltroYield:
    aplicar_global: bool = True
    aplicar_local: bool = True

    global_pct: float = 0.5      # variação global ±50% da mediana
    local_pct: float = 0.3       # variação local ±30% da mediana local
    raio_local_m: float = 30.0   # raio em metros para filtragem local

    min_pontos_pos_filtro: int = 30


class FiltroYield:
    def __init__(
        self,
        logger: Optional[LoggerAgricola] = None,
        config: Optional[ConfigFiltroYield] = None,
    ) -> None:
        self.logger = logger
        self.config = config or ConfigFiltroYield()

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        if self.logger is not None:
            self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)
        else:
            print(f"[{nivel.value}] {etapa}: {mensagem}")

    # ------------------------------------------------------------------
    # Filtros básicos
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_global(gdf: gpd.GeoDataFrame, value_col: str, pct: float) -> gpd.GeoDataFrame:
        med = gdf[value_col].median()
        lower, upper = med * (1 - pct), med * (1 + pct)
        return gdf[(gdf[value_col] >= lower) & (gdf[value_col] <= upper)].copy()

    @staticmethod
    def _filter_local_by_radius(
        gdf: gpd.GeoDataFrame,
        value_col: str,
        local_pct: float,
        radius: float,
    ) -> gpd.GeoDataFrame:
        coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
        tree = BallTree(coords, leaf_size=20)
        idxs = tree.query_radius(coords, r=radius)

        mask = []
        for i, neighbors in enumerate(idxs):
            vals = gdf.iloc[neighbors][value_col].values
            med_loc = np.median(vals)
            lower, upper = med_loc * (1 - local_pct), med_loc * (1 + local_pct)
            mask.append(lower <= gdf.iloc[i][value_col] <= upper)

        return gdf[mask].copy()

    @staticmethod
    def _compute_stats(gdf: gpd.GeoDataFrame, value_col: str) -> Dict[str, float]:
        vals = gdf[value_col].dropna().values
        if vals.size == 0:
            return {
                "count": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "cv": np.nan,
            }

        stats = {
            "count": float(len(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        }
        stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else np.nan
        return stats

    def _corrigir_por_balanca(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
    ) -> Tuple[gpd.GeoDataFrame, Optional[Dict[str, float]]]:
        col_bal = "Média de produção Kg/Ha (Balança)"

        if col_bal not in gdf.columns:
            # Sem dado de balança → nada a corrigir
            self._log(
                NivelLog.INFO,
                "filtro_yield",
                f"Coluna '{col_bal}' não encontrada. Ajuste por balança não será aplicado.",
            )
            return gdf, None

        # Série de yield (monitor) e balança
        serie_y = gdf[value_col]
        serie_bal = gdf[col_bal]

        # Coerção para numérico (vírgula decimal etc.)
        def _to_numeric(s: pd.Series) -> np.ndarray:
            if s.dtype == "O":
                s = (
                    s.astype(str)
                     .str.strip()
                     .str.replace(",", ".", regex=False)
                )
            return pd.to_numeric(s, errors="coerce").values

        y_vals = _to_numeric(serie_y)
        bal_vals = _to_numeric(serie_bal)

        # Médias
        mask_y = np.isfinite(y_vals)
        if mask_y.sum() == 0:
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                "Não há valores numéricos válidos em yield após filtragem. "
                "Ajuste por balança não aplicado.",
                mostrar_usuario=True,
            )
            return gdf, None

        mean_monitor = float(y_vals[mask_y].mean())

        bal_valid = bal_vals[np.isfinite(bal_vals)]
        if bal_valid.size == 0:
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                "Coluna de balança não possui valores numéricos válidos. "
                "Ajuste por balança não aplicado.",
                mostrar_usuario=True,
            )
            return gdf, None

        mean_balanca = float(bal_valid.mean())

        # Checagens básicas
        if mean_monitor <= 0 or mean_balanca <= 0:
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                f"Médias inválidas para ajuste por balança "
                f"(monitor={mean_monitor:.3f}, balança={mean_balanca:.3f}). "
                "Ajuste não aplicado.",
                mostrar_usuario=True,
            )
            return gdf, None

        fator = mean_balanca / mean_monitor

        gdf_corrigido = gdf.copy()
        y_corr = np.where(np.isfinite(y_vals), y_vals * fator, np.nan)
        gdf_corrigido[value_col] = y_corr

        self._log(
            NivelLog.INFO,
            "filtro_yield",
            f"Ajuste por balança aplicado. "
            f"Média monitor (pós-filtro) = {mean_monitor:.2f}, "
            f"Média balança = {mean_balanca:.2f}, "
            f"fator = {fator:.4f}.",
            mostrar_usuario=True,
        )

        info = {
            "mean_monitor": mean_monitor,
            "mean_balanca": mean_balanca,
            "fator": fator,
        }

        return gdf_corrigido, info


    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def filtrar(
        self,
        gdf_utm: gpd.GeoDataFrame,
        value_col: str,
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        if gdf_utm.crs is None or not gdf_utm.crs.is_projected:
            raise ValueError("FiltroYield espera um GeoDataFrame em CRS projetado (UTM, metros).")

        if value_col not in gdf_utm.columns:
            raise ValueError(f"Coluna de valor '{value_col}' não encontrada no GeoDataFrame.")

        cfg = self.config

        # 0) Estatísticas brutas
        raw_stats = self._compute_stats(gdf_utm, value_col)

        # 1) Filtragem global
        gdf_glob = gdf_utm.copy()
        glob_stats = raw_stats
        if cfg.aplicar_global:
            gdf_glob = self._filter_global(
                gdf_utm,
                value_col,
                cfg.global_pct,
            )
            glob_stats = self._compute_stats(gdf_glob, value_col)
            self._log(
                NivelLog.INFO,
                "filtro_yield",
                f"Filtragem GLOBAL aplicada (±{cfg.global_pct*100:.0f}%). "
                f"Pontos restantes: {glob_stats['count']:.0f}",
            )

        # 2) Filtragem local
        gdf_loc = gdf_glob.copy()
        loc_stats = glob_stats
        if cfg.aplicar_local:
            gdf_loc = self._filter_local_by_radius(
                gdf_glob,
                value_col,
                cfg.local_pct,
                cfg.raio_local_m,
            )
            loc_stats = self._compute_stats(gdf_loc, value_col)
            self._log(
                NivelLog.INFO,
                "filtro_yield",
                f"Filtragem LOCAL aplicada (raio={cfg.raio_local_m}m, "
                f"±{cfg.local_pct*100:.0f}%). Pontos restantes: {loc_stats['count']:.0f}",
            )

        # 3) Checagem de quantidade mínima
        if loc_stats is not None:
            n_final = int(loc_stats["count"])
        elif glob_stats is not None:
            n_final = int(glob_stats["count"])
        else:
            n_final = int(raw_stats["count"])

        if n_final < cfg.min_pontos_pos_filtro:
            msg = (
                f"Apenas {n_final} pontos restantes após filtragem de yield. "
                f"Mínimo configurado: {cfg.min_pontos_pos_filtro}. "
                "Considere ajustar parâmetros de filtro."
            )
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                msg,
                mostrar_usuario=True,
            )

        resumo_stats = {
            "raw": raw_stats,
            "global": glob_stats,
            "local": loc_stats,
            "n_final": n_final,
        }

        # 4) Ajuste por balança (se possível)
        gdf_corrigido, info_bal = self._corrigir_por_balanca(gdf_loc, value_col)
        if info_bal is not None:
            resumo_stats["ajuste_balanca"] = info_bal

        return gdf_corrigido, resumo_stats
