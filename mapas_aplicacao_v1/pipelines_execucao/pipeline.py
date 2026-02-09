# -*- coding: utf-8 -*-
"""
pipeline.py

Orquestrador do pipeline Mapas de Aplicação.

- Lê um config YAML/JSON (config_example.yml) e executa o fluxo:
  1) processador_lavoura
  2) processador_amostragem
  3) analise_espacial
  4) interpolador
  5) mapa_aplicacao
  6) export
  7) relatorio

UNIDADES
- A unidade da dose é definida pela recomendação no motor_calculos (ex.: t/ha, kg/ha, L/ha).
- O config pode declarar limites/round_step em outra unidade.
  Nesse caso, o pipeline converte explicitamente quando possível (kg/ha <-> t/ha).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
import geopandas as gpd


# ------------------------
# Import helpers
# ------------------------
def _ensure_paths():
    here = Path(__file__).resolve().parent
    parent = here.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def _import_attr(mod_candidates: List[str], attr: str):
    last = None
    for m in mod_candidates:
        try:
            mod = __import__(m, fromlist=[attr])
            return getattr(mod, attr)
        except Exception as e:
            last = e
    raise ImportError(f"Não consegui importar '{attr}' de {mod_candidates}. Último erro: {last}")


def _import_mod(mod_candidates: List[str]):
    last = None
    for m in mod_candidates:
        try:
            return __import__(m, fromlist=["*"])
        except Exception as e:
            last = e
    raise ImportError(f"Não consegui importar módulo de {mod_candidates}. Último erro: {last}")


_ensure_paths()

# logger
PipelineLogger = _import_attr(["pipelines.logger", "pipelines_execucao.logger", "logger"], "PipelineLogger")
LoggerConfig = _import_attr(["pipelines.logger", "pipelines_execucao.logger", "logger"], "LoggerConfig")

# módulos de execução
processar_lavoura = _import_attr(
    ["pipelines.processador_lavoura", "pipelines_execucao.processador_lavoura", "processador_lavoura"],
    "processar_lavoura",
)
processar_amostragem = _import_attr(
    ["pipelines.processador_amostragem", "pipelines_execucao.processador_amostragem", "processador_amostragem"],
    "processar_amostragem",
)

decidir_metodos_interpolacao = _import_attr(
    ["pipelines.analise_espacial", "pipelines_execucao.analise_espacial", "analise_espacial"],
    "decidir_metodos_interpolacao",
)
SpatialDecisionConfig = _import_attr(
    ["pipelines.analise_espacial", "pipelines_execucao.analise_espacial", "analise_espacial"],
    "SpatialDecisionConfig",
)

interpolar_atributos = _import_attr(
    ["pipelines.interpolador", "pipelines_execucao.interpolador", "interpolador"],
    "interpolar_atributos",
)
InterpolacaoConfig = _import_attr(
    ["pipelines.interpolador", "pipelines_execucao.interpolador", "interpolador"],
    "InterpolacaoConfig",
)

gerar_mapa_aplicacao = _import_attr(
    ["pipelines.mapa_aplicacao", "pipelines_execucao.mapa_aplicacao", "mapa_aplicacao"],
    "gerar_mapa_aplicacao",
)
MapaAplicacaoConfig = _import_attr(
    ["pipelines.mapa_aplicacao", "pipelines_execucao.mapa_aplicacao", "mapa_aplicacao"],
    "MapaAplicacaoConfig",
)

exportar_prescricao_download = _import_attr(
    ["pipelines.export", "pipelines_execucao.export", "export"],
    "exportar_prescricao_download",
)
ExportConfig = _import_attr(
    ["pipelines.export", "pipelines_execucao.export", "export"],
    "ExportConfig",
)

gerar_relatorio_prescricao = _import_attr(
    ["pipelines.relatorio", "pipelines_execucao.relatorio", "relatorio"],
    "gerar_relatorio_prescricao",
)
RelatorioConfig = _import_attr(
    ["pipelines.relatorio", "pipelines_execucao.relatorio", "relatorio"],
    "RelatorioConfig",
)

# motor de cálculos
motor = _import_mod(["pipelines.motor_calculos", "pipelines_execucao.motor_calculos", "motor_calculos"])


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _infer_name(path: str, default: str) -> str:
    stem = Path(path).stem.strip() or default
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)
    safe = "_".join([p for p in safe.split("_") if p])
    return safe or default


def _write_artifacts_json(path: str, data: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _convert_rate(value: Optional[float], from_unit: str, to_unit: str) -> Optional[float]:
    """
    Converte taxa entre unidades quando possível.
    Suporte: kg/ha <-> t/ha (massa).
    """
    if value is None:
        return None

    fu = (from_unit or "").strip()
    tu = (to_unit or "").strip()
    if fu == tu:
        return float(value)

    fu = fu.lower()
    tu = tu.lower()

    if "/ha" not in fu or "/ha" not in tu:
        raise ValueError(f"Conversão de unidade não suportada: {from_unit} -> {to_unit}")

    base_f = fu.replace("/ha", "")
    base_t = tu.replace("/ha", "")

    if base_f == "kg" and base_t == "t":
        return float(value) / 1000.0
    if base_f == "t" and base_t == "kg":
        return float(value) * 1000.0

    # volume (L) e outros não convertem aqui
    if base_f == base_t:
        return float(value)

    raise ValueError(f"Conversão de unidade não suportada (semântica diferente): {from_unit} -> {to_unit}")


def _resolve_logo_path(project_root: str) -> str:
    """
    Logo é hard-coded por decisão de produto.
    Pode-se inserir uma ou duas opções aqui e ligar/desligar elas.
    """
    base = Path(project_root) / "support_files" / "imagens_relatorio"
    c1 = base / "logo_exata.png"
    c2 = base / "logo exata alternativo.png"
    if c1.exists():
        return str(c1)
    return str(c2)  # mesmo se não existir, deixa o relatório acusar com erro claro


# ------------------------
# Config do pipeline
# ------------------------
@dataclass
class PipelineRunConfig:
    project_root: str
    contorno_path: str
    amostragem_path: str

    rec_key: str
    rec_params: Dict[str, Any]

    interpolation_method: str = "auto"
    pixel_size_m: float = 10.0

    # limites/arredondamento na unidade do config
    config_rate_unit: str = "kg/ha"
    dose_min_cfg: Optional[float] = None
    dose_max_cfg: Optional[float] = None
    round_step_cfg: Optional[float] = None
    round_mode_cfg: str = "nearest"

    # taxa fora
    enable_taxa_fora: bool = True
    buffer_out_m: float = 10.0
    outside_mode: str = "nearest"  # nearest | zero | fixed
    fixed_rate_unit: str = "kg/ha"
    fixed_rate_value: Optional[float] = None

    # pós-processamento / export (vindo do YAML)
    simplify_tolerance_m: Optional[float] = None
    min_polygon_area_ha: float = 0.5
    merge_small_mode: str = "largest_area"
    save_tmp_outputs: bool = False
    export_root_name: str = "export_final"

    export_format: str = "all"

    report_enabled: bool = True
    report_info: Dict[str, Any] = field(default_factory=dict)

    log_level: str = "INFO"


def run_pipeline(cfg: PipelineRunConfig) -> Dict[str, Any]:
    project_root = Path(cfg.project_root)
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    nome_lavoura = _infer_name(cfg.contorno_path, "lavoura")
    job_dir = outputs_dir / nome_lavoura / cfg.rec_key / _ts()
    job_dir.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger(
        LoggerConfig(
            job_dir=str(job_dir),
            level=str(cfg.log_level).upper(),
            to_console=True,
            to_file=True,
        )
    )

    logger.info(f"Job: {job_dir}")
    logger.info(f"rec_key={cfg.rec_key}")

    # ---- 1) lavoura
    lav = processar_lavoura(contorno_path=cfg.contorno_path, job_dir=str(job_dir), logger=logger, tmp_subdir="tmp")
    contorno_utm = gpd.read_file(lav.gpkg_path, layer=lav.layer)
    if contorno_utm.empty:
        raise ValueError("Contorno vazio após processar_lavoura.")

    # ---- 2) amostragem
    am = processar_amostragem(
        table_path=cfg.amostragem_path,
        contorno_utm=contorno_utm,
        crs_out=lav.crs_out,
        logger=logger,
        enforce_within=True,
        within_tolerance_m=0.0,
    )
    gdf_pts = am.gdf_utm

    # ---- 3) motor: required vars/params + spec (unidades)
    spec = motor.get_recommendation_spec(cfg.rec_key)
    required_vars = list(spec.required_soil_vars)
    required_params = list(spec.required_params)

    missing_params = [p for p in required_params if cfg.rec_params.get(p, None) is None]
    if missing_params:
        raise ValueError(f"Parâmetros ausentes para '{cfg.rec_key}': {missing_params}")

    missing_vars = [v for v in required_vars if v not in gdf_pts.columns]
    if missing_vars:
        raise ValueError(f"Variáveis de solo ausentes para '{cfg.rec_key}': {missing_vars}")

    logger.info(f"Unidade da recomendação: {spec.units} | suggested_round_step={spec.suggested_round_step}")

    # ---- 4) decisão espacial
    spatial_cfg = SpatialDecisionConfig()
    decisao = decidir_metodos_interpolacao(
        gdf_utm=gdf_pts,
        atributos=required_vars,
        interpolation_method=str(cfg.interpolation_method),
        logger=logger,
        cfg=spatial_cfg,
    )

    # ---- 5) interpolação
    interp_cfg = InterpolacaoConfig(pixel_size_m=float(cfg.pixel_size_m))
    rasters_info = interpolar_atributos(
        gdf_utm=gdf_pts,
        contorno_utm=contorno_utm,
        decisao_metodos=decisao,
        atributos=required_vars,
        job_dir=str(job_dir),
        logger=logger,
        cfg=interp_cfg,
        recortar_bbox=True,
    )

    # ---- 6) mapa aplicação (gera dose.tif)
    mapa_cfg = MapaAplicacaoConfig()
    out_map = gerar_mapa_aplicacao(
        rec_key=cfg.rec_key,
        params=cfg.rec_params,
        job_dir=str(job_dir),
        rasters_info=rasters_info,
        logger=logger,
        cfg=mapa_cfg,
    )

    # metadados de dose
    dose_unit = out_map.get("dose_unit", spec.units)
    dose_decimals = int(out_map.get("dose_decimals", spec.decimals))
    default_round_step = out_map.get("suggested_round_step", spec.suggested_round_step)

    # ---- 7) export (converte limites config_unit -> dose_unit)
    dose_min = _convert_rate(cfg.dose_min_cfg, cfg.config_rate_unit, dose_unit)
    dose_max = _convert_rate(cfg.dose_max_cfg, cfg.config_rate_unit, dose_unit)
    round_step = _convert_rate(cfg.round_step_cfg, cfg.config_rate_unit, dose_unit)
    fixed_rate = _convert_rate(cfg.fixed_rate_value, cfg.fixed_rate_unit, dose_unit)

    cfg_exp = ExportConfig(
        enable_taxa_fora=bool(cfg.enable_taxa_fora),
        buffer_out_m=float(cfg.buffer_out_m),
        outside_mode=str(cfg.outside_mode),
        fixed_rate=fixed_rate,
        dose_min=dose_min,
        dose_max=dose_max,
        round_step=round_step,
        round_mode=str(cfg.round_mode_cfg),
        simplify_tolerance_m=cfg.simplify_tolerance_m,
        min_polygon_area_ha=float(cfg.min_polygon_area_ha),
        merge_small_mode=str(cfg.merge_small_mode),
        save_tmp_outputs=bool(cfg.save_tmp_outputs),
        export_root_name=str(cfg.export_root_name or "export_final"),
        dose_unit=dose_unit,
        dose_decimals=dose_decimals,
        default_round_step=default_round_step,
    )

    out_dl = exportar_prescricao_download(
        dose_tif_path=out_map["dose_tif"],
        contorno_utm=contorno_utm,
        job_dir=str(job_dir),
        nome_talhao=_infer_name(cfg.contorno_path, "talhao"),
        formato=str(cfg.export_format),
        logger=logger,
        cfg=cfg_exp,
    )

    # ---- 8) relatório
    report_path = None
    if cfg.report_enabled:
        cfg_rep = RelatorioConfig(
            dose_unit=dose_unit,
            dose_decimals=dose_decimals,
            legend_n_classes=5,
            legend_mode="quantiles_area",
            legend_decimals=None,
        )

        report_dir = job_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = str(report_dir / f"relatorio_{cfg.rec_key}.pdf")

        info = dict(cfg.report_info or {})
        info.setdefault("produto", cfg.rec_key)
        info.setdefault("talhao", _infer_name(cfg.contorno_path, "talhao"))

        gdf_presc = out_dl.get("gdf_prescricao")
        if gdf_presc is not None and len(gdf_presc) > 0:
            gerar_relatorio_prescricao(
                gdf_prescricao=gdf_presc,
                contorno_utm=contorno_utm,
                output_pdf_path=report_path,
                logo_path=_resolve_logo_path(str(project_root)),
                info=info,
                cfg=cfg_rep,
                logger=logger,
            )
        else:
            logger.warning("Relatório não gerado: gdf_prescricao vazio.")

    artifacts = {
        "job_dir": str(job_dir),
        "dose_unit": dose_unit,
        "dose_decimals": dose_decimals,
        "dose_tif": out_map.get("dose_tif"),
        "export_root": out_dl.get("export_root"),
        "zip_path": out_dl.get("zip_path"),
        "report_path": report_path,
        "log_path": logger.get_log_path(),
    }
    _write_artifacts_json(str(job_dir / "artifacts.json"), artifacts)
    logger.info("Pipeline finalizado.")
    return artifacts


def run_from_yaml(config_path: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_y = yaml.safe_load(f)

    project_root = project_root or str(Path(config_path).resolve().parent)
    inputs = cfg_y.get("inputs", {}) or {}
    processing = cfg_y.get("processing", {}) or {}
    interpolation = cfg_y.get("interpolation", {}) or {}
    recommendation = cfg_y.get("recommendation", {}) or {}
    export = cfg_y.get("export", {}) or {}
    report = cfg_y.get("report", {}) or {}
    logging = cfg_y.get("logging", {}) or {}
    outputs = cfg_y.get("outputs", {}) or {}
    postprocess = cfg_y.get("postprocess", {}) or {}

    rate_limits = processing.get("rate_limits", {}) or {}
    rate_unit_cfg = str(rate_limits.get("unit") or cfg_y.get("project", {}).get("units", {}).get("rate", "kg/ha"))
    dose_min_cfg = rate_limits.get("min", None)
    dose_max_cfg = rate_limits.get("max", None)
    round_step_cfg = rate_limits.get("round_step", None)
    round_mode_cfg = str(rate_limits.get("round_mode", "nearest"))

    # compat: formato antigo processing.limits
    if (dose_min_cfg is None and dose_max_cfg is None and round_step_cfg is None) and ("limits" in processing):
        limits = processing.get("limits", {}) or {}
        dose_min_cfg = limits.get("dose_min_kg_ha", None)
        dose_max_cfg = limits.get("dose_max_kg_ha", None)
        round_step_cfg = limits.get("rounding_step_kg_ha", None)
        round_mode_cfg = "nearest"

    # outside buffer
    outside = processing.get("outside_buffer", {}) or {}
    enable_taxa_fora = bool(outside.get("enabled", True))
    buffer_out_m = float(outside.get("width_m", 10.0))
    outside_mode = str(outside.get("mode", "nearest"))

    fixed_rate = outside.get("fixed_rate", {}) or {}
    fixed_rate_unit = str(fixed_rate.get("unit") or rate_unit_cfg)
    fixed_rate_value = fixed_rate.get("value", None)

    # inputs
    contorno_path = str(Path(project_root) / (inputs.get("field_boundary", {}) or {}).get("path"))
    amostragem_path = str(Path(project_root) / (inputs.get("soil_table", {}) or {}).get("path"))

    # interpolation/grid
    pixel = float((processing.get("grid", {}) or {}).get("pixel_size_m", 10))
    method = str(interpolation.get("method", "auto"))

    # postprocess 
    geom_pp = (postprocess.get("geometry", {}) or {})
    simplify_tol = geom_pp.get("simplify_tolerance_m", None)
    min_poly_ha = float(geom_pp.get("min_polygon_area_ha", 0.5))
    merge_small_mode = str(geom_pp.get("merge_small_mode", "largest_area"))

    # outputs
    keep_intermediate = bool(outputs.get("keep_intermediate", False))

    # export package name_prefix -> export_root_name
    export_pkg = (export.get("output_package", {}) or {})
    name_prefix = export_pkg.get("name_prefix", None)
    export_root_name = str(name_prefix) if name_prefix else "export_final"

    # recommendations selection
    selection = recommendation.get("selection", ["calcario"])
    if selection == "all":
        rec_keys = motor.list_recommendations()
    else:
        rec_keys = list(selection)

    out_all = {}
    for rec_key in rec_keys:
        params = (recommendation.get("params", {}) or {}).get(rec_key, {})

        cfg_run = PipelineRunConfig(
            project_root=project_root,
            contorno_path=contorno_path,
            amostragem_path=amostragem_path,
            rec_key=rec_key,
            rec_params=params,
            interpolation_method=method,
            pixel_size_m=pixel,
            config_rate_unit=rate_unit_cfg,
            dose_min_cfg=dose_min_cfg,
            dose_max_cfg=dose_max_cfg,
            round_step_cfg=round_step_cfg,
            round_mode_cfg=round_mode_cfg,
            enable_taxa_fora=enable_taxa_fora,
            buffer_out_m=buffer_out_m,
            outside_mode=outside_mode,
            fixed_rate_unit=fixed_rate_unit,
            fixed_rate_value=fixed_rate_value,
            simplify_tolerance_m=simplify_tol,
            min_polygon_area_ha=min_poly_ha,
            merge_small_mode=merge_small_mode,
            save_tmp_outputs=keep_intermediate,
            export_root_name=export_root_name,
            export_format=str(export.get("format", "all")),
            report_enabled=bool(report.get("enabled", True)),
            report_info=(report.get("info", {}) or {}),
            log_level=str(logging.get("level", "INFO")),
        )
        out_all[rec_key] = run_pipeline(cfg_run)

    return out_all


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Caminho do YAML (ex.: config_example.yml)")
    ap.add_argument("--project_root", default=None, help="Raiz do projeto (default: pasta do YAML)")
    args = ap.parse_args()

    out = run_from_yaml(args.config, project_root=args.project_root)
    print(json.dumps(out, ensure_ascii=False, indent=2))
