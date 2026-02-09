# -*- coding: utf-8 -*-
"""
export.py (V2 - Download)

Etapas implementadas:
- taxa fora do mapa (ring fill por vizinho mais próximo)
- dose mínima
- dose máxima
- arredondamento de dose (step)
- vetorização + dissolve por dose
- explode de multipolígonos (ilhas)
- englobar polígonos menores do que X ha pelo vizinho dominante
- dissolve final por dose
- export em árvores de pastas para monitores (ou shp cru, ou all)

Saídas:
- (tmp/exports) GeoTIFF tratado + GPKG dissolvido (para QA/depuração)
- (export_final) árvore pronta para download (shapefile copiado nas pastas)
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import geopandas as gpd
from shapely.geometry import shape

import rasterio
from rasterio.transform import Affine
from rasterio.features import geometry_mask, shapes


# -------------------------
# Logger helper
# -------------------------
def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[{level}] {msg}")
        return
    fn = getattr(logger, level.lower(), None)
    if callable(fn):
        fn(msg)
    else:
        logger.info(msg)


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class ExportConfig:
    tmp_subdir: str = "tmp"
    exports_subdir: str = "exports"
    export_root_name: str = "export_final"
    nodata: float = -9999.0

    # metadados da dose (para export/relatório)
    dose_unit: Optional[str] = None   # ex.: 't/ha', 'kg/ha', 'L/ha'
    dose_decimals: int = 2
    default_round_step: Optional[float] = None  # se round_step=None, usa este

    # taxa fora
    enable_taxa_fora: bool = True
    outside_mode: str = 'nearest'           # "nearest" | "zero" | "fixed"
    fixed_rate: Optional[float] = None # valor na unidade da dose (dose_unit)
    buffer_out_m: float = 10.0
    all_touched: bool = False

    # constraints (pós taxa fora)
    dose_min: Optional[float] = None
    dose_max: Optional[float] = None

    # arredondamento (step)
    round_step: Optional[float] = None
    round_mode: str = "nearest"  # "nearest" | "floor" | "ceil"

    # vetorização
    simplify_tolerance_m: Optional[float] = None

    # merge de polígonos pequenos
    min_polygon_area_ha: float = 0.5  # 0.0 desativa
    merge_small_mode: str = "largest_area"  # ou "largest_shared_border"

    # QA: manter outputs intermediários?
    save_tmp_outputs: bool = True


# -------------------------
# Registry de monitores
# -------------------------
FORMATOS_MONITORES = {
    "trimble_cfx_750": {"display": "Trimble_CFX_750", "inner": "AgGPS/Prescriptions"},
    "trimble_gfx_750": {"display": "Trimble_GFX_750", "inner": "AgDATA/Prescriptions"},
    "stara":           {"display": "Stara",           "inner": "Dados/Mapas"},
    "john_deere":      {"display": "JohnDeere",       "inner": "Rx"},
    "hexagon":         {"display": "Hexagon",         "inner": ""},  
    "arvus":           {"display": "Arvus",           "inner": ""},
    "agres":           {"display": "Agres",           "inner": ""},
    "jacto":           {"display": "Jacto",           "inner": ""},
    "new_holland":     {"display": "New_Holland",     "inner": "Shapefile"},
    "raven":           {"display": "Raven",           "inner": "ePro/WorkOrders/rxMaps"},
    "verion":          {"display": "Verion",          "inner": "{nome_talhao}"},
}

# -------------------------
# Helpers
# -------------------------
def _norm_key(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def _sanitizar_nome(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "talhao"

def _zip_dir(folder_path: str, zip_path_no_ext: str) -> str:
    """
    Cria zip a partir de uma pasta.
    zip_path_no_ext: caminho completo SEM .zip (padrão do make_archive)
    Retorna o caminho do zip gerado.
    """
    return shutil.make_archive(zip_path_no_ext, "zip", folder_path)

# -------------------------
# Raster IO
# -------------------------
def _read_tif_as_nan(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    nd = meta["nodata"]
    if nd is not None and np.isfinite(nd):
        arr[arr == float(nd)] = np.nan
    return arr, meta


def _write_tif_from_nan(path: str, arr: np.ndarray, ref_meta: Dict[str, Any], nodata: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


def _pad_raster(arr: np.ndarray, meta: Dict[str, Any], pad_px: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    if pad_px <= 0:
        return arr, meta

    transform: Affine = meta["transform"]
    px_w = transform.a
    px_h = -transform.e

    arr2 = np.pad(arr, pad_width=pad_px, mode="constant", constant_values=np.nan)

    new_transform = Affine(
        transform.a, transform.b, transform.c - pad_px * px_w,
        transform.d, transform.e, transform.f + pad_px * px_h,
    )

    meta2 = dict(meta)
    meta2["transform"] = new_transform
    meta2["width"] = arr2.shape[1]
    meta2["height"] = arr2.shape[0]
    return arr2, meta2


# -------------------------
# Masks
# -------------------------
def _mask_inside(geom, out_shape, transform, all_touched=False) -> np.ndarray:
    return geometry_mask([geom], out_shape=out_shape, transform=transform, invert=True, all_touched=all_touched)


# -------------------------
# Taxa fora (ring fill)
# -------------------------
def _fill_ring_nearest(dose: np.ndarray, inside_mask: np.ndarray, ring_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    out = dose.copy()
    valid_inside = inside_mask & np.isfinite(dose)
    if not valid_inside.any():
        return out

    _, (iy, ix) = distance_transform_edt(~valid_inside, return_indices=True)
    out[ring_mask] = dose[iy[ring_mask], ix[ring_mask]]
    return out


# -------------------------
# Constraints e rounding
# -------------------------
def _apply_minmax_round(arr: np.ndarray, cfg: ExportConfig) -> np.ndarray:
    out = arr.copy()
    finite = np.isfinite(out)

    step_value = cfg.round_step if cfg.round_step is not None else cfg.default_round_step
    if step_value is not None:
        step = float(step_value)

    if cfg.dose_min is not None:
        out[finite] = np.maximum(out[finite], float(cfg.dose_min))
    if cfg.dose_max is not None:
        out[finite] = np.minimum(out[finite], float(cfg.dose_max))

    if cfg.round_step is not None:
        step = float(cfg.round_step)
        if step <= 0:
            raise ValueError("round_step deve ser > 0.")
        x = out[finite] / step

        mode = str(cfg.round_mode).lower()
        if mode == "nearest":
            x2 = np.round(x)
        elif mode == "floor":
            x2 = np.floor(x)
        elif mode == "ceil":
            x2 = np.ceil(x)
        else:
            raise ValueError("round_mode inválido: use 'nearest', 'floor' ou 'ceil'.")

        out[finite] = x2 * step

    return out


# -------------------------
# Vectorize + merge small
# -------------------------
def _englobar_poligonos_pequenos(
    gdf: gpd.GeoDataFrame,
    min_area_ha: float,
    mode: str = "largest_area",
) -> gpd.GeoDataFrame:
    if gdf.empty or min_area_ha <= 0:
        return gdf

    gdf = gdf.copy()
    gdf["area_ha"] = gdf.geometry.area / 10000.0

    changed = True
    while changed:
        changed = False

        small_idx = gdf.index[gdf["area_ha"] < float(min_area_ha)].tolist()
        if not small_idx:
            break

        sidx = gdf.sindex

        for i in small_idx:
            if i not in gdf.index:
                continue

            geom_i = gdf.at[i, "geometry"]

            cand = list(sidx.intersection(geom_i.bounds))
            cand = [j for j in cand if j != i and j in gdf.index]
            if not cand:
                continue

            touching = [j for j in cand if gdf.at[j, "geometry"].touches(geom_i)]

            if not touching:
                dists = [(j, gdf.at[j, "geometry"].distance(geom_i)) for j in cand]
                j_best = min(dists, key=lambda t: t[1])[0]
            else:
                if mode == "largest_shared_border":
                    scores = []
                    for j in touching:
                        shared = gdf.at[j, "geometry"].boundary.intersection(geom_i.boundary).length
                        scores.append((j, shared, gdf.at[j, "area_ha"]))
                    j_best = max(scores, key=lambda t: (t[1], t[2]))[0]
                else:
                    j_best = max(touching, key=lambda j: gdf.at[j, "area_ha"])

            gdf.at[j_best, "geometry"] = gdf.at[j_best, "geometry"].union(geom_i)
            gdf = gdf.drop(index=i)
            changed = True
            gdf.at[j_best, "area_ha"] = gdf.at[j_best, "geometry"].area / 10000.0

    return gdf


def _vectorize_prescricao(arr: np.ndarray, meta: Dict[str, Any], cfg: ExportConfig) -> gpd.GeoDataFrame:
    transform = meta["transform"]
    crs = meta["crs"]

    mask = np.isfinite(arr)
    if not mask.any():
        return gpd.GeoDataFrame({"dose": []}, geometry=[], crs=crs)

    feats = []
    for geom, val in shapes(arr.astype("float32"), mask=mask, transform=transform):
        feats.append({"geometry": shape(geom), "dose": float(val)})

    gdf = gpd.GeoDataFrame(feats, crs=crs)

    # dissolve por dose (adjacentes)
    gdf = gdf.dissolve(by="dose", as_index=False)

    # explode para separar ilhas (checagem por polígono individual)
    gdf = gdf.explode(index_parts=False, ignore_index=True)

    if cfg.simplify_tolerance_m is not None:
        tol = float(cfg.simplify_tolerance_m)
        if tol > 0:
            gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)

    gdf["area_ha"] = gdf.geometry.area / 10000.0

    min_ha = float(cfg.min_polygon_area_ha) if cfg.min_polygon_area_ha is not None else 0.0
    if min_ha > 0 and len(gdf) > 1:
        gdf = _englobar_poligonos_pequenos(gdf, min_area_ha=min_ha, mode=str(cfg.merge_small_mode))

    # dissolve final por dose
    gdf = gdf.dissolve(by="dose", as_index=False)
    gdf["area_ha"] = gdf.geometry.area / 10000.0

    return gdf


# -------------------------
# Shapefile export helpers
# -------------------------
def _shp_components(shp_path: str) -> List[str]:
    base = os.path.splitext(shp_path)[0]
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix", ".sbn", ".sbx"]
    out = []
    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            out.append(p)
    return out


def _copy_shapefile(src_shp: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for f in _shp_components(src_shp):
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))


def _write_base_shapefile(gdf: gpd.GeoDataFrame, out_dir: str, base_name: str = "prescricao") -> str:
    os.makedirs(out_dir, exist_ok=True)
    shp_path = os.path.join(out_dir, f"{base_name}.shp")

    gdf_out = gdf.copy()
    if "dose" not in gdf_out.columns:
        raise ValueError("GeoDataFrame não possui coluna 'dose' para export.")

    gdf_out["dose"] = gdf_out["dose"].astype(float)

    cols = ["dose"]
    if "area_ha" in gdf_out.columns:
        cols.append("area_ha")

    gdf_out = gdf_out[cols + ["geometry"]]
    gdf_out.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    return shp_path


def _exportar_shp_para_monitores(
    gdf_prescricao: gpd.GeoDataFrame,
    job_dir: str,
    nome_talhao: str,
    formato: str,
    logger=None,
    export_root_name: str = "export_final",
) -> Dict[str, Any]:
    formato_key = _norm_key(formato)
    nome_talhao = (nome_talhao or "talhao").strip()

    out_root = Path(job_dir) / export_root_name
    out_root.mkdir(parents=True, exist_ok=True)

    base_dir = out_root / "_base_shp"
    shp_base = _write_base_shapefile(gdf_prescricao, str(base_dir), base_name="prescricao")

    created = []

    def _resolve_target_dir(mkey: str) -> Path:
        info = FORMATOS_MONITORES[mkey]
        display = info["display"]
        inner = info["inner"].format(nome_talhao=_sanitizar_nome(nome_talhao)).strip("/")
    
        # nível 1: pasta do monitor
        base = out_root / display
    
        # nível 2: estrutura interna (pode ser vazio)
        return base / inner if inner else base

    if formato_key == "shp":
        dst = out_root / "shp"
        _copy_shapefile(shp_base, str(dst))
        created.append(str(dst))

    elif formato_key == "all":
        dst = out_root / "shp"
        _copy_shapefile(shp_base, str(dst))
        created.append(str(dst))

        for mkey in FORMATOS_MONITORES.keys():
            dst = _resolve_target_dir(mkey)
            _copy_shapefile(shp_base, str(dst))
            created.append(str(dst))

    else:
        if formato_key not in FORMATOS_MONITORES:
            raise ValueError(
                f"Formato '{formato}' não reconhecido. Use 'shp', 'all' ou um destes: "
                f"{sorted(FORMATOS_MONITORES.keys())}"
            )
        dst = _resolve_target_dir(formato_key)
        _copy_shapefile(shp_base, str(dst))
        created.append(str(dst))

    _log(logger, "INFO", f"Export final criado em: {out_root}")
    return {
        "export_root": str(out_root),
        "created_dirs": created,
        "base_shp": shp_base,
        "formato": formato_key,
    }



def exportar_prescricao_download(
    dose_tif_path: str,
    contorno_utm: gpd.GeoDataFrame,
    job_dir: str,
    nome_talhao: str,
    formato: str = "shp",  # "shp" | "<monitor>" | "all"
    logger=None,
    cfg: Optional[ExportConfig] = None,
) -> Dict[str, Any]:
    """
    Pipeline completo de export:
    - raster dose -> taxa fora -> min/max/round
    - (opcional) salva tmp/dose_tratada.tif e tmp/prescricao_dissolve.gpkg
    - vetorização + merge de ilhas pequenas
    - export shapefile em árvore do monitor (download)

    Retorna paths + stats.
    """
    cfg = cfg or ExportConfig()

    dose, meta = _read_tif_as_nan(dose_tif_path)
    talhao = contorno_utm.geometry.unary_union

    transform: Affine = meta["transform"]
    px = float(transform.a)
    pad_px = 0
    if cfg.enable_taxa_fora and cfg.buffer_out_m and cfg.buffer_out_m > 0:
        pad_px = int(np.ceil(cfg.buffer_out_m / px)) + 2

    if pad_px > 0:
        _log(logger, "INFO", f"Padding raster: {pad_px} px (buffer_out={cfg.buffer_out_m} m).")
        dose, meta = _pad_raster(dose, meta, pad_px)

    out_shape = (meta["height"], meta["width"])
    tr = meta["transform"]

    inside = _mask_inside(talhao, out_shape=out_shape, transform=tr, all_touched=cfg.all_touched)
    
    # ring = fora do contorno, mas apenas até buffer_out_m (em metros)
    ring = np.zeros_like(inside, dtype=bool)
    if cfg.enable_taxa_fora and cfg.buffer_out_m and cfg.buffer_out_m > 0:
        from scipy.ndimage import distance_transform_edt
    
        # distância (em pixels) até o contorno: para pixels fora, mede até o pixel inside mais próximo
        dist_px = distance_transform_edt(~inside)  # fora do contorno
        px = float(meta["transform"].a)            # pixel size em metros 
        dist_m = dist_px * px
    
        ring = (~inside) & (dist_m <= float(cfg.buffer_out_m))
    
    dose2 = dose

    if cfg.enable_taxa_fora and cfg.buffer_out_m and cfg.buffer_out_m > 0:
        if cfg.outside_mode == "nearest":
            _log(logger, "INFO", "Aplicando taxa fora (vizinho mais próximo).")
            dose2 = _fill_ring_nearest(dose, inside_mask=inside, ring_mask=ring)
        
        elif cfg.outside_mode == "zero":
            _log(logger, "INFO", "Aplicando taxa fora = 0.")
            dose2 = dose.copy()
            dose2[ring] = 0.0
        
        elif cfg.outside_mode == "fixed":
            if cfg.fixed_rate is None:
                raise ValueError("outside_buffer.mode='fixed' exige processing.outside_buffer.fixed_rate.value (não pode ser null).")
            _log(logger, "INFO", f"Aplicando taxa fora fixa: {cfg.fixed_rate}.")
            dose2 = dose.copy()
            dose2[ring] = float(cfg.fixed_rate)
        
        else:
            raise ValueError(f"outside_buffer.mode inválido: {cfg.outside_mode} (use nearest|zero|fixed)")
        

    _log(logger, "INFO", "Aplicando min/max/round (pós taxa-fora).")
    dose3 = _apply_minmax_round(dose2, cfg)

    finite = np.isfinite(dose3)
    stats = {
        "n_total": int(dose3.size),
        "n_validos": int(finite.sum()),
        "min": float(np.nanmin(dose3)) if finite.any() else None,
        "max": float(np.nanmax(dose3)) if finite.any() else None,
        "mean": float(np.nanmean(dose3)) if finite.any() else None,
        "n_ring_pixels": int(ring.sum()),
    }

    # tmp outputs (QA)
    tmp_tif = None
    tmp_gpkg = None
    if cfg.save_tmp_outputs:
        out_dir = Path(job_dir) / cfg.tmp_subdir / cfg.exports_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_tif = out_dir / "dose_tratada.tif"
        tmp_gpkg = out_dir / "prescricao_dissolve.gpkg"

        meta_out = dict(meta)
        _write_tif_from_nan(str(tmp_tif), dose3, meta_out, nodata=cfg.nodata)

    # vetorização
    _log(logger, "INFO", "Vetorizando prescrição (dissolve/explode/merge small/dissolve).")
    gdf = _vectorize_prescricao(dose3, meta, cfg)

    if cfg.save_tmp_outputs and tmp_gpkg is not None:
        if len(gdf) > 0:
            gdf.to_file(tmp_gpkg, layer="prescricao", driver="GPKG")
        else:
            _log(logger, "WARNING", "Prescrição vazia após processamento (sem pixels válidos).")

    # export para download
    out_dl = _exportar_shp_para_monitores(
        gdf_prescricao=gdf,
        job_dir=job_dir,
        nome_talhao=nome_talhao,
        formato=formato,
        logger=logger,
        export_root_name=cfg.export_root_name,
    )

    # zip do export_final para download
    zip_out = None
    try:
        export_root = out_dl["export_root"]
        zip_base = str(Path(export_root).with_suffix(""))  
        zip_out = _zip_dir(export_root, zip_base)          
        _log(logger, "INFO", f"ZIP criado: {zip_out}")
    except Exception as e:
        _log(logger, "WARNING", f"Falha ao criar ZIP do export_final: {e}")
    
    return {
        "stats": stats,
        "gdf_prescricao": gdf,  
        "tmp_dose_tif_tratada": str(tmp_tif) if tmp_tif else None,
        "tmp_prescricao_gpkg": str(tmp_gpkg) if tmp_gpkg else None,
        "zip_path": zip_out,
        **out_dl
    }
