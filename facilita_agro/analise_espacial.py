"""
analise_espacial.py - Análise de dependência espacial e decisão IDW vs Krigagem
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import geopandas as gpd

from .logger import LoggerAgricola, NivelLog


try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    HAS_MORAN = True
except ImportError:
    HAS_MORAN = False

try:
    from skgstat import Variogram
    HAS_VARIO = True
except ImportError:
    HAS_VARIO = False


class AnalisadorEspacial:
    """
    Responsável por:
      - avaliar autocorrelação espacial (Moran I global)
      - ajustar variograma simples
      - decidir automaticamente entre IDW e Kriging para um atributo
    """

    def __init__(self, logger: Optional[LoggerAgricola] = None) -> None:
        self.logger = logger

        if not HAS_MORAN:
            self._log(
                NivelLog.WARNING,
                "init_analise_espacial",
                "libpysal/esda não encontrados. Moran I desabilitado. "
                "Decisão espacial ficará mais conservadora (tende a IDW).",
                mostrar_usuario=False,
            )
        if not HAS_VARIO:
            self._log(
                NivelLog.WARNING,
                "init_analise_espacial",
                "scikit-gstat não encontrado. Ajuste de variograma desabilitado. "
                "Decisão espacial ficará mais conservadora (tende a IDW).",
                mostrar_usuario=False,
            )

    # ---------------------------------------------------------
    # Logging helper
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Moran global
    # ---------------------------------------------------------

    def avaliar_moran_global(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        k: int = 8,
        permutations: int = 499,
    ) -> Dict[str, Any]:
        """
        Calcula Moran I global para um atributo.

        Retorna dict com:
          - I, EI, p_value, z_score, n, k
        Se Moran não estiver disponível, retorna dict com flag 'disponivel=False'.
        """
        if not HAS_MORAN:
            return {"disponivel": False}

        gdf_valid = gdf.dropna(subset=[value_col]).copy()
        n = len(gdf_valid)

        if n < 5:
            return {
                "disponivel": False,
                "motivo": f"n={n} < 5 pontos válidos para Moran",
            }

        coords = np.column_stack(
            [gdf_valid.geometry.x.values, gdf_valid.geometry.y.values]
        )
        valores = gdf_valid[value_col].values

        w = KNN(coords, k=k)
        w.transform = "r"  

        mi = Moran(valores, w, permutations=permutations)

        return {
            "disponivel": True,
            "I": mi.I,
            "EI": mi.EI,
            "p_value": mi.p_sim,
            "z_score": mi.z_sim,
            "n": n,
            "k": k,
        }

    # ---------------------------------------------------------
    # Variograma
    # ---------------------------------------------------------

    def ajustar_variograma(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        n_lags: int = 10,
        model: str = "spherical",
        maxlag: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ajusta um variograma simples usando scikit-gstat.

        Retorna dict com:
          - psill, range, nugget, sill_total,
          - nugget_sill_ratio,
          - diag (diagonal da área),
          - range_frac (range/diag),
          - n_lags, model

        Se não for possível ajustar, lança Exception.
        """
        if not HAS_VARIO:
            raise RuntimeError("scikit-gstat não disponível para ajuste de variograma.")

        gdf_valid = gdf.dropna(subset=[value_col]).copy()
        n = len(gdf_valid)
        if n < 10:
            raise RuntimeError(f"Amostra muito pequena para variograma (n={n})")

        coords = np.column_stack(
            [gdf_valid.geometry.x.values, gdf_valid.geometry.y.values]
        )
        valores = gdf_valid[value_col].values

        V = Variogram(
            coords,
            valores,
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

        nugget_sill_ratio = nugget / sill_total if sill_total > 0 else np.nan
        range_frac = rang / diagonal if diagonal > 0 else np.nan

        return {
            "psill": float(psill),
            "range": float(rang),
            "nugget": float(nugget),
            "sill_total": float(sill_total),
            "nugget_sill_ratio": float(nugget_sill_ratio),
            "diag": float(diagonal),
            "range_frac": float(range_frac),
            "n_lags": int(n_lags),
            "model": model,
            "n": int(n),
        }

    # ---------------------------------------------------------
    # Decisão IDW vs Kriging (por atributo)
    # ---------------------------------------------------------

    def decidir_metodo_interpolacao(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        moran_alpha: float = 0.05,
        min_n_kriging: int = 30,
        max_n_kriging: int = 2000,
        max_nugget_sill_ratio: float = 0.7,
        min_range_frac: float = 0.1,
        k_moran: int = 8,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Decide automaticamente entre 'idw' e 'kriging' para um atributo.

        Retorna:
          metodo: 'idw' ou 'kriging'
          diag: dict com diagnóstico detalhado
        """
        diag: Dict[str, Any] = {"atributo": value_col}
        gdf_valid = gdf.dropna(subset=[value_col]).copy()
        n = len(gdf_valid)
        diag["n_validos"] = n

        if n < min_n_kriging:
            motivo = f"n={n} < {min_n_kriging}, amostra pequena para kriging"
            diag["motivo"] = motivo
            self._log(NivelLog.INFO, "decidir_interp", f"{value_col}: {motivo}")
            return "idw", diag

        if n > max_n_kriging:
            motivo = f"n={n} > {max_n_kriging}, kriging muito pesado → IDW"
            diag["motivo"] = motivo
            self._log(NivelLog.INFO, "decidir_interp", f"{value_col}: {motivo}")
            return "idw", diag

        moran_info = self.avaliar_moran_global(
            gdf_valid, value_col, k=k_moran
        )
        diag["moran"] = moran_info

        if not moran_info.get("disponivel", False):
            self._log(
                NivelLog.WARNING,
                "decidir_interp",
                f"{value_col}: Moran indisponível, decisão baseada só em variograma (se disponível).",
            )
        else:
            I = moran_info["I"]
            p = moran_info["p_value"]

            if (p > moran_alpha) or (I <= 0):
                motivo = (
                    f"Autocorrelação global fraca/não significativa "
                    f"(I={I:.3f}, p={p:.3f}) → IDW"
                )
                diag["motivo"] = motivo
                self._log(NivelLog.INFO, "decidir_interp", f"{value_col}: {motivo}")
                return "idw", diag

        if not HAS_VARIO:
            motivo = (
                "scikit-gstat não disponível; usando IDW mesmo com autocorrelação "
                "detetada (se houver)."
            )
            diag["motivo"] = motivo
            self._log(NivelLog.WARNING, "decidir_interp", f"{value_col}: {motivo}")
            return "idw", diag

        try:
            vario = self.ajustar_variograma(gdf_valid, value_col)
            diag["variograma"] = vario
        except Exception as e:
            motivo = f"Falha ao ajustar variograma ({e}) → IDW"
            diag["motivo"] = motivo
            self._log(NivelLog.WARNING, "decidir_interp", f"{value_col}: {motivo}")
            return "idw", diag

        nsr = vario["nugget_sill_ratio"]
        rf = vario["range_frac"]

        if np.isnan(nsr) or nsr > max_nugget_sill_ratio:
            motivo = (
                f"Nugget/Sill alto (ratio={nsr:.2f}) → estrutura fraca / muito ruído → IDW"
            )
            diag["motivo"] = motivo
            self._log(NivelLog.INFO, "decidir_interp", f"{value_col}: {motivo}")
            return "idw", diag

        if np.isnan(rf) or rf < min_range_frac:
            motivo = (
                f"Alcance relativo baixo (range_frac={rf:.2f}) → correlação em escala "
                "muito pequena → IDW"
            )
            diag["motivo"] = motivo
            self._log(NivelLog.INFO, "decidir_interp", f"{value_col}: {motivo}")
            return "idw", diag

        # Se passou em tudo, kriging é plausível
        motivo = (
            "Autocorrelação espacial significativa e variograma com estrutura "
            f"razoável (nugget/sill={nsr:.2f}, range_frac={rf:.2f}) → Kriging"
        )
        diag["motivo"] = motivo
        self._log(NivelLog.INFO, "decidir_interp", f"{value_col}: {motivo}")

        return "kriging", diag

    # ---------------------------------------------------------
    # Decisão em lote para vários atributos
    # ---------------------------------------------------------

    def analisar_atributos(
        self,
        gdf: gpd.GeoDataFrame,
        atributos: List[str],
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Roda decidir_metodo_interpolacao para uma lista de atributos.

        Retorna dict:
          { atributo: {"metodo": "idw"/"kriging", "diag": {...}}, ... }
        """
        resultados: Dict[str, Dict[str, Any]] = {}

        for col in atributos:
            metodo, diag = self.decidir_metodo_interpolacao(gdf, col, **kwargs)
            resultados[col] = {
                "metodo": metodo,
                "diag": diag,
            }

        return resultados
