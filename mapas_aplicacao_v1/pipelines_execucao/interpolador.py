"""
interpolador.py

Mapas de Aplicação — Passo 5 (interpolação espacial dos atributos de solo)

Contrato do projeto:
- Interpola VARIÁVEIS (ex.: "T", "V %"), não a dose final.
- Consome decisão por atributo (idw/kriging) vinda do analise_espacial.py.
- Exporta rasters temporários (GeoTIFF) dentro do job_dir/tmp/rasters.
- Fora do contorno = NaN.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping

# ----------------------------
# Optional deps
# ----------------------------
try:
    import rasterio
    from rasterio.transform import from_origin, Affine
    from rasterio.features import geometry_mask
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except Exception:
    HAS_PYKRIGE = False


# ----------------------------
# Configs
# ----------------------------
@dataclass(frozen=True)
class InterpolacaoConfig:
    pixel_size_m: float = 10.0
    nodata: float = -9999.0
    tmp_subdir: str = "tmp"
    rasters_subdir: str = "rasters"

    # IDW
    idw_power: float = 2.0
    idw_k: int = 12

    # Kriging
    kriging_variogram_model: str = "spherical"


# ----------------------------
# Helpers
# ----------------------------
def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[{level}] {msg}")
        return
    fn = getattr(logger, level.lower(), None)
    if callable(fn):
        fn(msg)
    else:
        logger.info(msg)


def _sanitizar_nome(texto: str) -> str:
    """Nome seguro para arquivo: remove acentos, troca não-alfanumérico por '_'."""
    s = unicodedata.normalize("NFKD", str(texto))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _criar_grade(contorno_utm: gpd.GeoDataFrame, pixel_size_m: float) -> Tuple[np.ndarray, np.ndarray, "Affine"]:
    """
    Grade de centros de pixel (xs, ys) e transform affine.
    xs crescente; ys decrescente (top->down).
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é necessário para criar GeoTIFF/transform. Instale rasterio.")

    minx, miny, maxx, maxy = contorno_utm.total_bounds
    px = float(pixel_size_m)

    xs = np.arange(minx + px / 2.0, maxx, px)
    ys = np.arange(maxy - px / 2.0, miny, -px)
    transform = from_origin(minx, maxy, px, px)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Grade vazia. Verifique contorno/pixel_size_m.")
    return xs, ys, transform


def _mask_outside_contorno(
    contorno_union,
    out_shape: Tuple[int, int],
    transform: "Affine",
) -> np.ndarray:
    """
    Retorna máscara booleana com True para pixels FORA do contorno.
    """
    mask_outside = geometry_mask(
        [mapping(contorno_union)],
        out_shape=out_shape,
        transform=transform,
        invert=False,
        all_touched=False,
    )
    return mask_outside


def _recortar_bbox_valido(grid: np.ndarray, transform: "Affine") -> Tuple[np.ndarray, "Affine"]:
    """
    Recorta para o menor bbox contendo valores finitos.
    Mantém NaN fora do contorno. Ajusta transform.
    """
    valid = np.isfinite(grid)
    if not valid.any():
        return grid, transform

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)

    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]

    r_min, r_max = int(r_idx[0]), int(r_idx[-1])
    c_min, c_max = int(c_idx[0]), int(c_idx[-1])

    cropped = grid[r_min : r_max + 1, c_min : c_max + 1]

    offset = Affine.translation(c_min, r_min)
    new_transform = transform * offset
    return cropped, new_transform


def _salvar_geotiff(
    path: Path,
    grid: np.ndarray,
    transform: "Affine",
    crs: Any,
    nodata: float,
):
    """Salva GeoTIFF 1 banda com nodata numérico."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.where(np.isfinite(grid), grid, nodata).astype("float32")
    height, width = data.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float(nodata),
        compress="lzw",
    ) as dst:
        dst.write(data, 1)


# ----------------------------
# Interpoladores
# ----------------------------
def _interpolar_idw(
    gdf_utm: gpd.GeoDataFrame,
    value_col: str,
    xs: np.ndarray,
    ys: np.ndarray,
    power: float,
    k: int,
) -> np.ndarray:
    if not HAS_SCIPY:
        raise RuntimeError("scipy não disponível para IDW (cKDTree). Instale scipy.")

    g = gdf_utm.dropna(subset=[value_col]).copy()
    if g.empty:
        raise RuntimeError(f"Nenhum valor válido em '{value_col}' para IDW.")

    x = g.geometry.x.values.astype(float)
    y = g.geometry.y.values.astype(float)
    z = g[value_col].values.astype(float)

    n = len(z)
    if n == 1:
        return np.full((len(ys), len(xs)), float(z[0]), dtype=float)

    k_eff = int(min(max(1, k), n))
    tree = cKDTree(np.column_stack([x, y]))

    grid_x, grid_y = np.meshgrid(xs, ys)  # ys decrescente
    q = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    dist, idx = tree.query(q, k=k_eff)
    dist = np.asarray(dist)
    idx = np.asarray(idx)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / (dist ** float(power))
    w[~np.isfinite(w)] = 0.0

    zn = z[idx]
    num = np.sum(w * zn, axis=1)
    den = np.sum(w, axis=1)

    out = np.where(den > 0, num / den, np.nan).astype(float)
    return out.reshape(grid_x.shape)


def _interpolar_kriging(
    gdf_utm: gpd.GeoDataFrame,
    value_col: str,
    xs: np.ndarray,
    ys_desc: np.ndarray,
    variogram_model: str,
) -> np.ndarray:
    if not HAS_PYKRIGE:
        raise RuntimeError("pykrige não disponível para Kriging. Instale pykrige.")

    g = gdf_utm.dropna(subset=[value_col]).copy()
    if len(g) < 10:
        raise RuntimeError(f"Amostra muito pequena para Kriging (n={len(g)}).")

    x = g.geometry.x.values.astype(float)
    y = g.geometry.y.values.astype(float)
    z = g[value_col].values.astype(float)

    ys_asc = np.sort(ys_desc)

    OK = OrdinaryKriging(
        x, y, z,
        variogram_model=str(variogram_model),
        verbose=False,
        enable_plotting=False,
    )
    z_grid, _ss = OK.execute("grid", xs, ys_asc)

    grid_vals = np.asarray(z_grid, dtype=float)[::-1, :]  
    return grid_vals


# ----------------------------
# API do módulo
# ----------------------------
def interpolar_atributos(
    gdf_utm: gpd.GeoDataFrame,
    contorno_utm: gpd.GeoDataFrame,
    decisao_metodos: Dict[str, Dict[str, Any]],
    atributos: List[str],
    job_dir: str,
    logger=None,
    cfg: Optional[InterpolacaoConfig] = None,
    recortar_bbox: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Interpola lista de atributos e salva GeoTIFFs temporários no job_dir.

    Parameters
    ----------
    gdf_utm:
        GeoDataFrame de pontos (UTM).
    contorno_utm:
        Contorno (UTM).
    decisao_metodos:
        Saída do decidir_metodos_interpolacao (analise_espacial.py).
        Ex.: {"T": {"metodo":"idw", "diag": {...}}, ...}
    atributos:
        Lista de colunas a interpolar.
    job_dir:
        Pasta do job (outputs/<field>/<job_id>).
    cfg:
        Configurações de interpolação.
    recortar_bbox:
        Se True, recorta cada raster ao menor bbox válido (reduz tamanho).

    Returns
    -------
    dict por atributo:
      {
        "T": {
            "metodo_solicitado": "kriging",
            "metodo_usado": "idw",
            "tif_path": "...",
            "transform": Affine(...),
            "shape": (ny, nx),
            "diag": {...}
        },
        ...
      }
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é obrigatório neste pipeline (export temporário em GeoTIFF). Instale rasterio.")

    cfg = cfg or InterpolacaoConfig()

    tmp_dir = Path(job_dir) / cfg.tmp_subdir
    rasters_dir = tmp_dir / cfg.rasters_subdir
    rasters_dir.mkdir(parents=True, exist_ok=True)

    # prepara grade comum para todos
    xs, ys, transform = _criar_grade(contorno_utm, cfg.pixel_size_m)
    contorno_union = contorno_utm.geometry.unary_union
    mask_outside = _mask_outside_contorno(contorno_union, out_shape=(len(ys), len(xs)), transform=transform)

    results: Dict[str, Dict[str, Any]] = {}

    _log(logger, "INFO", f"Interpolando {len(atributos)} atributo(s) em grade {len(xs)}x{len(ys)} (px={cfg.pixel_size_m} m).")

    for col in atributos:
        info_dec = decisao_metodos.get(col, {})
        metodo = str(info_dec.get("metodo", "idw")).lower().strip()
        if metodo not in ("idw", "kriging"):
            metodo = "idw"

        nome = _sanitizar_nome(col) or "atributo"
        tif_path = rasters_dir / f"{nome}.tif"

        meta = {
            "atributo": col,
            "metodo_solicitado": metodo,
            "metodo_usado": metodo,
            "tif_path": str(tif_path),
            "shape": None,
            "diag": info_dec.get("diag", {}),
        }

        _log(logger, "INFO", f"{col}: executando {metodo.upper()}...")

        grid = None
        metodo_usado = metodo

        if metodo == "kriging":
            try:
                grid = _interpolar_kriging(
                    gdf_utm=gdf_utm,
                    value_col=col,
                    xs=xs,
                    ys_desc=ys,
                    variogram_model=cfg.kriging_variogram_model,
                )
            except Exception as e:
                _log(logger, "WARNING", f"{col}: kriging falhou/indisponível ({e}). Fallback -> IDW.")
                metodo_usado = "idw"
                grid = _interpolar_idw(
                    gdf_utm=gdf_utm,
                    value_col=col,
                    xs=xs,
                    ys=ys,
                    power=cfg.idw_power,
                    k=cfg.idw_k,
                )
                meta["fallback"] = {"from": "kriging", "to": "idw", "reason": str(e)}
        else:
            grid = _interpolar_idw(
                gdf_utm=gdf_utm,
                value_col=col,
                xs=xs,
                ys=ys,
                power=cfg.idw_power,
                k=cfg.idw_k,
            )

        grid = grid.astype(float, copy=False)
        grid[mask_outside] = np.nan

        out_transform = transform
        if recortar_bbox:
            grid, out_transform = _recortar_bbox_valido(grid, transform)

        _salvar_geotiff(
            path=tif_path,
            grid=grid,
            transform=out_transform,
            crs=contorno_utm.crs,
            nodata=cfg.nodata,
        )

        meta["metodo_usado"] = metodo_usado
        meta["shape"] = (int(grid.shape[0]), int(grid.shape[1]))
        meta["transform"] = out_transform

        results[col] = meta
        _log(logger, "INFO", f"{col}: salvo (tmp) em {tif_path.name} | metodo_usado={metodo_usado.upper()} | shape={meta['shape']}")

    return results
