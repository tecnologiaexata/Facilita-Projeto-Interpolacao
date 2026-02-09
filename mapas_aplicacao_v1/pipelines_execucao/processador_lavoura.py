"""processador_lavoura.py

Mapas de Aplicação — Passo 1 (pré-processamento do contorno)

Requisitos do projeto:
- Entrada: KML (geralmente sem CRS explícito no arquivo)
- Garantir CRS 4326 ao ler (se vier ausente)
- Determinar UTM automaticamente via centróide (lon/lat)
- Reprojetar contorno para UTM (métrico)
- Salvar contorno reprojetado em GPKG para uso downstream no pipeline

Observação:
- Portanto, o GPKG é um artefato temporário do job (salvo dentro do job_dir).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import geopandas as gpd


@dataclass(frozen=True)
class LavouraOutput:
    """Saída do processador de lavoura."""
    gpkg_path: str
    layer: str
    crs_out: str 


def _detectar_epsg_utm(lat: float, lon: float) -> str:
    """Retorna EPSG UTM para um ponto (lat/lon em graus).

    Regras:
    - zona = floor((lon + 180)/6) + 1
    - hemisfério sul => EPSG:327{zona:02d}
      hemisfério norte => EPSG:326{zona:02d}
    """
    zona = int((lon + 180) // 6) + 1
    if lat < 0:
        return f"EPSG:327{zona:02d}"
    return f"EPSG:326{zona:02d}"


def processar_lavoura_kml_para_utm_gpkg(
    kml_path: str,
    job_dir: str,
    logger=None,
    tmp_subdir: str = "tmp",
    out_filename: str = "contorno_utm.gpkg",
    layer: str = "contorno",
) -> LavouraOutput:
    """Processa um KML de contorno e salva em GPKG reprojetado para UTM.

    Parameters
    ----------
    kml_path:
        Caminho para o KML do contorno.
    job_dir:
        Diretório base do job (ex.: outputs/pivo/job_YYYYMMDD_HHMMSS).
    logger:
        Logger opcional (JobLogger). Se None, usa prints.
    tmp_subdir:
        Subpasta onde o GPKG temporário será salvo (default: "tmp").
    out_filename:
        Nome do arquivo GPKG de saída (default: "contorno_utm.gpkg").
    layer:
        Nome da layer no GPKG (default: "contorno").

    Returns
    -------
    LavouraOutput: (gpkg_path, layer, crs_out)
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

    if not os.path.exists(kml_path):
        raise FileNotFoundError(f"KML não encontrado: {kml_path}")

    tmp_dir = os.path.join(job_dir, tmp_subdir)
    os.makedirs(tmp_dir, exist_ok=True)

    _log("INFO", f"Lendo contorno KML: {kml_path}")
    gdf = gpd.read_file(kml_path)

    if gdf.empty:
        raise ValueError("KML lido, porém não contém geometrias.")

    # Garantir CRS 4326 se ausente
    if gdf.crs is None:
        _log("WARNING", "CRS ausente no KML. Assumindo EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    else:
        # Se vier em outro CRS, reprojeta para 4326 antes de detectar UTM
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception as e:
            raise ValueError(f"Falha ao reprojetar contorno para EPSG:4326: {e}")

    # Detectar UTM pelo centróide (em 4326)
    centroid = gdf.geometry.unary_union.centroid
    lon, lat = float(centroid.x), float(centroid.y)
    crs_out = _detectar_epsg_utm(lat=lat, lon=lon)
    _log("INFO", f"CRS UTM detectado (centróide): {crs_out} (lat={lat:.6f}, lon={lon:.6f})")

    # Reprojetar para UTM
    gdf_utm = gdf.to_crs(crs_out)

    # Salvar em GPKG (temporário)
    gpkg_path = os.path.join(tmp_dir, out_filename)
    if os.path.exists(gpkg_path):
        try:
            os.remove(gpkg_path)
        except Exception:
            # Se existir e não puder remover, salva com sufixo
            base, ext = os.path.splitext(gpkg_path)
            gpkg_path = f"{base}_1{ext}"

    _log("INFO", f"Salvando contorno UTM em GPKG: {gpkg_path} (layer='{layer}')")
    gdf_utm.to_file(gpkg_path, layer=layer, driver="GPKG")

    return LavouraOutput(gpkg_path=gpkg_path, layer=layer, crs_out=crs_out)

def processar_lavoura(
    contorno_path: str,
    job_dir: str,
    logger=None,
    tmp_subdir: str = "tmp",
):
    """
    Wrapper compatível com pipeline.py.
    Hoje suporta KML; se amanhã vocês aceitarem SHP/GPKG, expande aqui.
    """
    ext = os.path.splitext(str(contorno_path).lower())[1]

    if ext == ".kml":
        return processar_lavoura_kml_para_utm_gpkg(
            kml_path=contorno_path,
            job_dir=job_dir,
            logger=logger,
            tmp_subdir=tmp_subdir,
        )

    raise ValueError(
        f"Formato de contorno não suportado ainda: '{ext}'. "
        "Atualmente o pipeline suporta apenas KML."
    )
