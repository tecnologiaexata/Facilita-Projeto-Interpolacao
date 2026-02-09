"""
mapa_aplicacao.py

Mapas de Aplicação — Passo 6 (álgebra de mapas e geração do raster de aplicação)

Responsabilidade:
- Ler rasters temporários de solo (GeoTIFF) do job_dir/tmp/rasters
- Montar a tabela pixel-a-pixel com as variáveis requeridas
- Aplicar a função de recomendação do motor_calculos.py (ex.: calcular_calcario)
- Salvar raster temporário de dose (GeoTIFF) em job_dir/tmp/maps
- Retornar metadados/paths para o export.py

Constraints (min/max/arredondamento):
- NÃO aplicado aqui (ficará no export, na etapa raster->vetor).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    from pipelines.motor_calculos import (
        get_recommendation_fn,
        get_required_soil_vars,
        get_required_params,
        get_recommendation_spec,
    )
except Exception:
    from pipelines_execucao.motor_calculos import (
        get_recommendation_fn,
        get_required_soil_vars,
        get_required_params,
        get_recommendation_spec,
    )


def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[{level}] {msg}")
        return
    fn = getattr(logger, level.lower(), None)
    if callable(fn):
        fn(msg)
    else:
        logger.info(msg)


@dataclass(frozen=True)
class MapaAplicacaoConfig:
    tmp_subdir: str = "tmp"
    rasters_subdir: str = "rasters"
    maps_subdir: str = "maps"
    nodata: float = -9999.0

    # comportamento de alinhamento:
    # "error" = aborta se grid divergir
    # "reproject" = reamostra/reprojeta para o grid de referência
    align_mode: str = "error"

    # resampling só usado se align_mode="reproject"
    resampling: str = "nearest"


def _read_raster(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é necessário para ler/escrever GeoTIFFs.")
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
        }
    # converte nodata numérico em NaN
    nd = meta["nodata"]
    if nd is not None and np.isfinite(nd):
        arr[arr == float(nd)] = np.nan
    return arr, meta


def _align_to_reference(
    arr: np.ndarray,
    meta: Dict[str, Any],
    ref_meta: Dict[str, Any],
    mode: str,
    resampling: str,
    logger=None,
) -> np.ndarray:
    """Garante que (arr, meta) fique no mesmo grid do raster de referência."""
    same = (
        meta["crs"] == ref_meta["crs"]
        and meta["transform"] == ref_meta["transform"]
        and meta["width"] == ref_meta["width"]
        and meta["height"] == ref_meta["height"]
    )
    if same:
        return arr

    if mode == "error":
        raise ValueError(
            "Raster desalinhado do grid de referência. "
            f"ref: (crs={ref_meta['crs']}, shape={ref_meta['height']}x{ref_meta['width']}) | "
            f"cur: (crs={meta['crs']}, shape={meta['height']}x{meta['width']})"
        )

    if mode != "reproject":
        raise ValueError(f"align_mode inválido: {mode}")

    _log(logger, "WARNING", "Raster desalinhado — reamostrando/reprojetando para o grid de referência.")

    if not HAS_RASTERIO:
        raise RuntimeError("rasterio necessário para reproject.")

    # escolhe resampling
    res_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    res = res_map.get(str(resampling).lower(), Resampling.nearest)

    dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=float)

    reproject(
        source=arr,
        destination=dst,
        src_transform=meta["transform"],
        src_crs=meta["crs"],
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        resampling=res,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _write_raster(
    path: str,
    arr: np.ndarray,
    ref_meta: Dict[str, Any],
    nodata: float,
):
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é necessário para escrever GeoTIFF.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.where(np.isfinite(arr), arr, float(nodata)).astype("float32")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ref_meta["height"],
        width=ref_meta["width"],
        count=1,
        dtype="float32",
        crs=ref_meta["crs"],
        transform=ref_meta["transform"],
        nodata=float(nodata),
        compress="lzw",
    ) as dst:
        dst.write(data, 1)


def gerar_mapa_aplicacao(
    rec_key: str,
    params: Dict[str, Any],
    job_dir: str,
    rasters_info: Optional[Dict[str, Dict[str, Any]]] = None,
    logger=None,
    cfg: Optional[MapaAplicacaoConfig] = None,
) -> Dict[str, Any]:
    """
    Gera o raster (GeoTIFF temporário) do projeto de aplicação para a recomendação rec_key.


    Retorna:
      {
        "rec_key": ...,
        "required_soil_vars": [...],
        "required_params": [...],
        "dose_tif": ".../job_dir/tmp/maps/<rec_key>_dose.tif",
        "grid_ref": {...},
        "stats": {...},
        "inputs": {var: {"tif_path":..., "method_used":..., ...}, ...}
      }
    """
    cfg = cfg or MapaAplicacaoConfig()

    required_vars = get_required_soil_vars(rec_key)
    required_params = get_required_params(rec_key)

    spec = get_recommendation_spec(rec_key)

    # valida params (só presença; regra e ranges ficam fora daqui)
    missing_params = [p for p in required_params if params.get(p, None) is None]
    if missing_params:
        raise ValueError(f"Parâmetros obrigatórios ausentes para '{rec_key}': {missing_params}")

    # resolve paths dos rasters de entrada
    if rasters_info is None:
        raise ValueError(
            "rasters_info é obrigatório neste pipeline (evita ambiguidade de nomes). "
            "Passe o retorno do interpolador."
        )

    missing_vars = [v for v in required_vars if v not in rasters_info]
    if missing_vars:
        raise ValueError(f"Rasters ausentes em rasters_info para '{rec_key}': {missing_vars}")

    # lê raster de referência = primeiro var requerida
    ref_var = required_vars[0]
    ref_path = rasters_info[ref_var]["tif_path"]
    _log(logger, "INFO", f"Mapa aplicação '{rec_key}': usando '{ref_var}' como referência de grid.")

    ref_arr, ref_meta = _read_raster(ref_path)

    # lê/alinha todas as variáveis requeridas
    inputs_arrays: Dict[str, np.ndarray] = {}
    inputs_meta: Dict[str, Dict[str, Any]] = {}

    for v in required_vars:
        p = rasters_info[v]["tif_path"]
        arr, meta = _read_raster(p)
        arr = _align_to_reference(arr, meta, ref_meta, mode=cfg.align_mode, resampling=cfg.resampling, logger=logger)
        inputs_arrays[v] = arr
        inputs_meta[v] = {
            "tif_path": p,
            "metodo_usado": rasters_info[v].get("metodo_usado"),
            "metodo_solicitado": rasters_info[v].get("metodo_solicitado"),
            "fallback": rasters_info[v].get("fallback"),
        }

    # máscara: qualquer NaN em qualquer var => NaN na saída
    nan_mask = np.zeros_like(ref_arr, dtype=bool)
    for v in required_vars:
        nan_mask |= ~np.isfinite(inputs_arrays[v])

    fn = get_recommendation_fn(rec_key)

    flat = {}
    for v in required_vars:
        flat[v] = inputs_arrays[v].ravel()

    df = pd.DataFrame(flat)

    # onde tem NaN em alguma var, a função pode dar NaN/erro;
    # nós forçamos NaN no resultado depois.
    dose = fn(df, params)
    dose_arr = np.asarray(dose, dtype=float).reshape(ref_meta["height"], ref_meta["width"])

    # impõe NaN onde faltou dado de entrada (fora contorno/buracos)
    dose_arr[nan_mask] = np.nan

    stats = {
        "n_pixels_total": int(dose_arr.size),
        "n_validos": int(np.isfinite(dose_arr).sum()),
        "min": float(np.nanmin(dose_arr)) if np.isfinite(dose_arr).any() else None,
        "max": float(np.nanmax(dose_arr)) if np.isfinite(dose_arr).any() else None,
        "mean": float(np.nanmean(dose_arr)) if np.isfinite(dose_arr).any() else None,
    }
    if stats["min"] is not None and stats["min"] < 0:
        _log(logger, "WARNING", f"Dose negativa detectada (min={stats['min']:.3f}). Sem clamp aqui (fica pro export).")

    # salva raster temporário
    maps_dir = Path(job_dir) / cfg.tmp_subdir / cfg.maps_subdir
    dose_path = maps_dir / f"{rec_key}_dose.tif"
    _write_raster(str(dose_path), dose_arr, ref_meta, nodata=cfg.nodata)

    _log(logger, "INFO", f"Projeto de aplicação (dose) salvo em tmp: {dose_path}")

    return {
        "rec_key": rec_key,
        "required_soil_vars": list(required_vars),
        "required_params": list(required_params),
        "dose_tif": str(dose_path),
        "dose_unit": spec.units,
        "dose_decimals": int(spec.decimals),
        "suggested_round_step": spec.suggested_round_step,
        "suggested_min": spec.suggested_min,
        "suggested_max": spec.suggested_max,
        "grid_ref": {
            "crs": ref_meta["crs"],
            "transform": ref_meta["transform"],
            "width": ref_meta["width"],
            "height": ref_meta["height"],
        },
        "stats": stats,
        "inputs": inputs_meta,
        "params_used": dict(params),
    }
