"""
analise_espacial.py

Decisão IDW vs Kriging (por atributo), com fallback conservador.
- Se libs opcionais não estiverem disponíveis, tende a IDW (com diagnóstico).
- Se config.interpolation.method != "auto", respeita (forced).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd

try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    HAS_MORAN = True
except Exception:
    HAS_MORAN = False

try:
    from skgstat import Variogram
    HAS_VARIO = True
except Exception:
    HAS_VARIO = False


def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[{level}] {msg}")
        return
    fn = getattr(logger, level.lower(), None)
    if callable(fn):
        fn(msg)
    else:
        # fallback
        logger.info(msg)


@dataclass
class SpatialDecisionConfig:
    moran_alpha: float = 0.05
    min_n_kriging: int = 30
    max_n_kriging: int = 2000
    max_nugget_sill_ratio: float = 0.7
    min_range_frac: float = 0.1
    k_moran: int = 8
    permutations: int = 499
    n_lags: int = 10
    variogram_model: str = "spherical"
    maxlag: Optional[float] = None


class AnalisadorEspacial:
    def __init__(self, logger=None):
        self.logger = logger
        if not HAS_MORAN:
            _log(self.logger, "WARNING",
                 "libpysal/esda não encontrados. Moran I desabilitado. Decisão ficará mais conservadora (tende a IDW).")
        if not HAS_VARIO:
            _log(self.logger, "WARNING",
                 "scikit-gstat não encontrado. Variograma desabilitado. Decisão ficará mais conservadora (tende a IDW).")

    def avaliar_moran_global(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        k: int,
        permutations: int,
    ) -> Dict[str, Any]:
        if not HAS_MORAN:
            return {"available": False, "reason": "Moran deps missing"}

        gdf_valid = gdf.dropna(subset=[value_col]).copy()
        n = len(gdf_valid)
        if n < 5:
            return {"available": False, "reason": f"n={n} < 5"}

        coords = np.column_stack([gdf_valid.geometry.x.values, gdf_valid.geometry.y.values])
        values = gdf_valid[value_col].values

        w = KNN(coords, k=k)
        w.transform = "r"
        mi = Moran(values, w, permutations=permutations)

        return {
            "available": True,
            "I": float(mi.I),
            "EI": float(mi.EI),
            "p_value": float(mi.p_sim),
            "z_score": float(mi.z_sim),
            "n": int(n),
            "k": int(k),
        }

    def ajustar_variograma(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        n_lags: int,
        model: str,
        maxlag: Optional[float],
    ) -> Dict[str, Any]:
        if not HAS_VARIO:
            raise RuntimeError("Variogram deps missing")

        gdf_valid = gdf.dropna(subset=[value_col]).copy()
        n = len(gdf_valid)
        if n < 10:
            raise RuntimeError(f"n={n} < 10 for variogram")

        coords = np.column_stack([gdf_valid.geometry.x.values, gdf_valid.geometry.y.values])
        values = gdf_valid[value_col].values

        V = Variogram(
            coords,
            values,
            n_lags=n_lags,
            model=model,
            maxlag=maxlag,
            normalize=False,
            use_nugget=True,
        )
        psill, rang, nugget = V.parameters
        sill_total = psill + nugget

        minx, miny, maxx, maxy = gdf_valid.total_bounds
        diagonal = float(np.hypot(maxx - minx, maxy - miny))

        nsr = (nugget / sill_total) if sill_total > 0 else np.nan
        rf = (rang / diagonal) if diagonal > 0 else np.nan

        return {
            "psill": float(psill),
            "range": float(rang),
            "nugget": float(nugget),
            "sill_total": float(sill_total),
            "nugget_sill_ratio": float(nsr),
            "diag": float(diagonal),
            "range_frac": float(rf),
            "n_lags": int(n_lags),
            "model": str(model),
            "n": int(n),
        }

    def decidir_metodo_por_atributo(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        cfg: SpatialDecisionConfig,
    ) -> Tuple[str, Dict[str, Any]]:
        diag: Dict[str, Any] = {"atributo": value_col}

        gdf_valid = gdf.dropna(subset=[value_col]).copy()
        n = len(gdf_valid)
        diag["n_validos"] = int(n)

        if n < cfg.min_n_kriging:
            diag["motivo"] = f"n={n} < {cfg.min_n_kriging} (amostra pequena) -> IDW"
            return "idw", diag

        if n > cfg.max_n_kriging:
            diag["motivo"] = f"n={n} > {cfg.max_n_kriging} (kriging pesado) -> IDW"
            return "idw", diag

        moran = self.avaliar_moran_global(gdf_valid, value_col, k=cfg.k_moran, permutations=cfg.permutations)
        diag["moran"] = moran

        if moran.get("available", False):
            I = moran["I"]
            p = moran["p_value"]
            if (p > cfg.moran_alpha) or (I <= 0):
                diag["motivo"] = f"Autocorr fraca/não sig (I={I:.3f}, p={p:.3f}) -> IDW"
                return "idw", diag
        else:
            _log(self.logger, "WARNING", f"{value_col}: Moran indisponível; decisão seguirá conservadora.")

        if not HAS_VARIO:
            diag["motivo"] = "Variograma indisponível (deps missing) -> IDW"
            return "idw", diag

        try:
            vario = self.ajustar_variograma(gdf_valid, value_col, n_lags=cfg.n_lags, model=cfg.variogram_model, maxlag=cfg.maxlag)
            diag["variograma"] = vario
        except Exception as e:
            diag["motivo"] = f"Falha no variograma ({e}) -> IDW"
            return "idw", diag

        nsr = vario["nugget_sill_ratio"]
        rf = vario["range_frac"]

        if np.isnan(nsr) or nsr > cfg.max_nugget_sill_ratio:
            diag["motivo"] = f"Nugget/Sill alto (ratio={nsr:.2f}) -> IDW"
            return "idw", diag

        if np.isnan(rf) or rf < cfg.min_range_frac:
            diag["motivo"] = f"Range_frac baixo ({rf:.2f}) -> IDW"
            return "idw", diag

        diag["motivo"] = f"Passou critérios (nsr={nsr:.2f}, rf={rf:.2f}) -> Kriging"
        return "kriging", diag


def decidir_metodos_interpolacao(
    gdf_utm: gpd.GeoDataFrame,
    atributos: List[str],
    interpolation_method: str = "auto",  # auto|idw|kriging
    logger=None,
    cfg: Optional[SpatialDecisionConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Decide método por atributo, apenas para a lista 'atributos' fornecida.
    Se interpolation_method != 'auto', respeita e marca diag como forced.
    """
    cfg = cfg or SpatialDecisionConfig()
    analyzer = AnalisadorEspacial(logger=logger)

    method = str(interpolation_method).lower().strip()
    if method not in ("auto", "idw", "kriging"):
        raise ValueError("interpolation_method deve ser: auto, idw ou kriging")

    results: Dict[str, Dict[str, Any]] = {}

    if method != "auto":
        for col in atributos:
            results[col] = {
                "metodo": method,
                "diag": {"atributo": col, "motivo": f"Forced by config: {method}"},
            }
        _log(logger, "INFO", f"Método forçado via config: {method} (aplicado a {len(atributos)} atributo(s))")
        return results

    # auto
    _log(logger, "INFO", f"Decidindo método automaticamente para {len(atributos)} atributo(s).")
    for col in atributos:
        m, diag = analyzer.decidir_metodo_por_atributo(gdf_utm, col, cfg)
        results[col] = {"metodo": m, "diag": diag}
        _log(logger, "INFO", f"{col}: {m} | {diag.get('motivo','')}")
    return results
