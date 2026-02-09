# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# Import do pipeline (ele está em pipelines_execucao)
REPO_ROOT = Path(__file__).resolve().parents[1]  
PIPE_DIR = REPO_ROOT / "pipelines_execucao"

import sys
if str(PIPE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import run_from_yaml  


app = FastAPI(
    title="Mapas de Aplicação API",
    version="0.1",
    description="API para gerar mapas de aplicação e export/relatório via pipeline.",
)

def _api_artifacts_path(job_id: str) -> Path:
    return _job_dir(job_id) / "api_artifacts.json"


def _save_api_artifacts(job_id: str, out: dict):
    p = _api_artifacts_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_api_artifacts(job_id: str) -> dict:
    p = _api_artifacts_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Job não encontrado ou artifacts ausentes.")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifacts do job corrompidos: {e}")


def _resolve_artifact_path(artifacts_for_rec: dict, artifact: str) -> Optional[str]:
    artifact = (artifact or "").lower()
    if artifact in ("zip", "package"):
        return artifacts_for_rec.get("zip_path")
    if artifact in ("report", "pdf"):
        return artifacts_for_rec.get("report_path")
    if artifact in ("log", "logs"):
        return artifacts_for_rec.get("log_path")
    return None


def _slugify_name(name: str, default: str = "boundary") -> str:
    name = (name or "").strip()
    if not name:
        return default
    # troca espaços e caracteres estranhos por underscore
    name = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or default


def _job_dir(job_id: str) -> Path:
    return REPO_ROOT / "outputs" / "api_jobs" / job_id


def _safe_write_upload(dst: Path, up: UploadFile):
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = up.file.read()
    dst.write_bytes(data)


def _build_yaml_config(
    *,
    kml_rel: str,
    soil_rel: str,
    rec_selection: list[str],
    rec_params: dict[str, Any],
    cfg_user: dict[str, Any],
) -> dict[str, Any]:
    """
    Constrói um config YAML no formato que o pipeline entende (unit-aware).
    `cfg_user` pode trazer overrides (pixel_size, interpolation.method, rate_limits, outside_buffer, postprocess, etc.)
    """

    # Defaults “seguros”
    base = {
        "project": {
            "name": "mapas_aplicacao",
            "version": "0.1",
            "crs_out": "auto",
            "units": {"rate": "kg/ha", "pixel": "m"},
        },
        "inputs": {
            "field_boundary": {"format": "kml", "path": kml_rel, "layer": None},
            "soil_table": {
                "format": "auto",
                "path": soil_rel,
                "x_col": "x",
                "y_col": "y",
                "id_col": None,
                "delimiter": ",",
                "decimal": ".",
                "encoding": "utf-8",
            },
        },
        "processing": {
            "grid": {"pixel_size_m": 10, "snap_to_boundary": True},
            "masking": {"nodata_value": -9999, "clip_to_boundary": True},
            # unit-aware:
            "rate_limits": {
                "unit": "t/ha",
                "min": None,
                "max": None,
                "round_step": None,
                "round_mode": "nearest",
            },
            "outside_buffer": {
                "enabled": True,
                "width_m": 10,
                "mode": "nearest",  # nearest | zero | fixed
                "fixed_rate": {"unit": "t/ha", "value": None},
            },
        },
        "interpolation": {"method": "auto"},
        "recommendation": {
            "selection": rec_selection,
            "params": rec_params,
            "missing_variables": "error",
            "missing_params": "error",
        },
        "postprocess": {
            "geometry": {
                "simplify_tolerance_m": 0.0,
                "min_polygon_area_ha": 0.5,
                "merge_small_mode": "largest_area",
            },
            "metadata_fields": {"add_fields": True},
        },
        "export": {
            "format": "all",
            "output_package": {"enabled": True, "type": "zip", "name_prefix": "export_final"},
        },
        "report": {"enabled": True, "format": "pdf", "info": {}},
        "logging": {"level": "INFO", "structured": True, "save_to_file": True, "folder": "logs"},
        "outputs": {"folder": "outputs", "keep_intermediate": False},
    }

    # Você pode mandar blocos completos: processing, interpolation, postprocess, export, report, logging, outputs.
    for k in ("project", "inputs", "processing", "interpolation", "postprocess", "export", "report", "logging", "outputs"):
        if k in cfg_user and isinstance(cfg_user[k], dict):
            # merge 1 nível
            base[k].update(cfg_user[k])

    return base


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_job(
    # config em JSON string
    config_json: str = Form(...),
    boundary_kml: UploadFile = File(...),
    soil_table: UploadFile = File(...),
):
    """
    Executa o pipeline de forma síncrona.

    Espera:
    - config_json: string JSON com, no mínimo:
        {
          "recommendation": {"selection": ["calcario"], "params": {"calcario": {...}}},
          "processing": {"rate_limits": {...}, "outside_buffer": {...}},
          ...
        }
    - boundary_kml: arquivo .kml
    - soil_table: .csv (ou .shp/.gpkg se seu processador suportar)
    """
    try:
        cfg_user = json.loads(config_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"config_json inválido: {e}")

    # validação mínima do contrato
    rec = (cfg_user.get("recommendation") or {})
    selection = rec.get("selection")
    params = rec.get("params") or {}

    if not selection:
        raise HTTPException(status_code=400, detail="config_json.recommendation.selection é obrigatório.")
    if selection != "all" and not isinstance(selection, list):
        raise HTTPException(status_code=400, detail="recommendation.selection deve ser 'all' ou lista.")
    if selection == "all":
        rec_selection = "all"
    else:
        rec_selection = selection

    # job_id e pastas
    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # salva uploads dentro de outputs/
    meta = (cfg_user.get("meta") or {})
    field_name = meta.get("field_name") or meta.get("talhao") or meta.get("field_id")
    
    # fallback: nome do arquivo enviado
    if not field_name:
        field_name = Path(boundary_kml.filename or "boundary.kml").stem
    
    field_slug = _slugify_name(str(field_name), default="boundary")
    
    kml_path = inputs_dir / f"{field_slug}.kml"
    
    soil_ext = Path(soil_table.filename or "soil.csv").suffix.lower() or ".csv"
    soil_path = inputs_dir / f"soil{soil_ext}"


    _safe_write_upload(kml_path, boundary_kml)
    _safe_write_upload(soil_path, soil_table)

    kml_rel = str(kml_path.relative_to(REPO_ROOT)).replace("\\", "/")
    soil_rel = str(soil_path.relative_to(REPO_ROOT)).replace("\\", "/")

    # monta YAML final
    yaml_cfg = _build_yaml_config(
        kml_rel=kml_rel,
        soil_rel=soil_rel,
        rec_selection=rec_selection,
        rec_params=params,
        cfg_user=cfg_user,
    )

    cfg_path = job_dir / "config.yml"
    cfg_path.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # roda pipeline
    try:
        out = run_from_yaml(str(cfg_path), project_root=str(REPO_ROOT))
    except Exception as e:
        # grava erro e devolve
        (job_dir / "error.txt").write_text(str(e), encoding="utf-8")
        raise HTTPException(status_code=500, detail=f"Falha no pipeline: {e}")

    # cria links de download por rec_key (se selection foi lista)
    downloads = {}
    if isinstance(out, dict):
        for rec_key, artifacts in out.items():
            zip_path = artifacts.get("zip_path")
            report_path = artifacts.get("report_path")
            downloads[rec_key] = {
                "zip": f"/download/{job_id}/{rec_key}/zip" if zip_path else None,
                "report": f"/download/{job_id}/{rec_key}/report" if report_path else None,
                "log": f"/download/{job_id}/{rec_key}/log" if artifacts.get("log_path") else None,
                "job_dir": artifacts.get("job_dir"),
            }


    _save_api_artifacts(job_id, out)

    return JSONResponse(
        {
            "job_id": job_id,
            "outputs": out,
            "downloads": downloads,
            "config_path": str(cfg_path),
        }
    )


@app.get("/download/{job_id}/{rec_key}/{artifact}")
def download_artifact(job_id: str, rec_key: str, artifact: str):
    """
    Baixa artefatos gerados por job, sem expor paths internos.
    artifact: zip | report | log
    """
    out = _load_api_artifacts(job_id)

    if rec_key not in out:
        raise HTTPException(status_code=404, detail=f"Recomendação '{rec_key}' não encontrada no job {job_id}.")

    artifacts_for_rec = out[rec_key] or {}
    path = _resolve_artifact_path(artifacts_for_rec, artifact)
    if not path:
        raise HTTPException(status_code=404, detail=f"Artefato '{artifact}' não disponível para '{rec_key}'.")

    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()

    allowed_root = (REPO_ROOT / "outputs").resolve()
    if allowed_root not in p.parents and p != allowed_root:
        raise HTTPException(status_code=403, detail="Path fora de outputs/ não permitido.")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado no disco.")

    return FileResponse(str(p), filename=p.name)
