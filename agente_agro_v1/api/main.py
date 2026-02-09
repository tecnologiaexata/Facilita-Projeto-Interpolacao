# -*- coding: utf-8 -*-
"""
agente_agro/api/main.py

API mínima para expor o Agente de Inteligência Agronômica.

Dois endpoints:
- POST /run/temporal    -> roda o pipeline temporal
- POST /run/individual  -> roda o pipeline intra-safra por Safra_prod

Saída:
- report_text
- bundle_json extraído do audit do observability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agente_agro.pipelines import observability
from agente_agro.pipelines import runner


APP_NAME = "Agente Agro API"
app = FastAPI(title=APP_NAME, version="0.2.0")


# ---------------------
# Helpers
# ---------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _uploads_dir() -> Path:
    d = _project_root() / "outputs" / "_api_uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_upload_csv(upload: UploadFile, run_id: str) -> Path:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Arquivo sem nome.")
    if not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

    out = _uploads_dir() / f"grid_completo__{run_id}.csv"

    try:
        content = upload.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")

        max_mb = 150
        if len(content) > max_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande (> {max_mb} MB).",
            )

        out.write_bytes(content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao salvar upload: {type(e).__name__}: {e}")
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    return out


# ----------------------
# Audit helpers (bundle_json)
# -----------------------

def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(p.read_bytes().decode("utf-8", errors="replace"))
        except Exception:
            return {}


def _find_last_decision_metrics(events: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    last = None
    for ev in events:
        if ev.get("type") == "decision" and ev.get("name") == name:
            last = ev.get("metrics")
    if isinstance(last, dict):
        return last
    return None


def _split_events_by_runner_season(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Segmenta o audit em blocos por safra.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    cur_season: Optional[str] = None

    for ev in events:
        if ev.get("type") == "step_start" and ev.get("step") == "runner_individual_season":
            ctx = ev.get("context") or {}
            cur_season = str(ctx.get("season") or "").strip() or None
            if cur_season and cur_season not in out:
                out[cur_season] = []
            continue

        if cur_season:
            out[cur_season].append(ev)

    return out


def _bundle_from_audit_temporal(audit_path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Extrai o bundle temporal do audit.
    Fonte: recommendation_engine_temporal_full (metrics)
    """
    audit = _read_json(audit_path)
    events = list(audit.get("events") or [])
    metrics = _find_last_decision_metrics(events, "recommendation_engine_temporal_full")
    if not metrics:
        return None
    return metrics


def _bundle_from_audit_individual(audit_path: str | Path, season: str) -> Optional[Dict[str, Any]]:
    """
    Extrai o bundle intra-safra (por season) do audit.
    Fonte: recommendation_engine_full (metrics) dentro do bloco da season.
    """
    audit = _read_json(audit_path)
    events = list(audit.get("events") or [])
    by_season = _split_events_by_runner_season(events)

    block = by_season.get(season)
    if not block:
        return None

    metrics = _find_last_decision_metrics(block, "recommendation_engine_full")
    if not metrics:
        return None
    return metrics


# ---------
# Routes
# ---------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": APP_NAME}


@app.post("/run/temporal")
def run_temporal(
    file: UploadFile = File(...),
    talhao: Optional[str] = Form(default=None),
    season_col: str = Form(default="Safra_prod"),
    enable_cross_season_filter: bool = Form(default=True),
    top_k: int = Form(default=5),
    min_support_rate: float = Form(default=0.50),
    include_bundle_json: bool = Form(default=True),
) -> JSONResponse:
    """
    Roda o pipeline temporal e retorna:
      - report_text
      - bundle_json (opcional) extraído do audit
    """
    run_id = observability.new_run_id()
    if talhao:
        observability.set_context(Talhao=str(talhao))

    csv_path = _save_upload_csv(file, run_id)

    try:
        ok, report_text, warnings = runner.run_temporal(
            csv_path,
            season_col=season_col,
            enable_cross_season_filter=enable_cross_season_filter,
            top_k=int(top_k),
            min_support_rate=float(min_support_rate),
            observability=observability,
        )
    except Exception as e:
        try:
            observability.record_decision(
                "api_temporal_error",
                decision="fail",
                reason=str(e),
                metrics={"exc_type": type(e).__name__},
            )
            observability.export_audit(step_name="api_temporal__error")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Erro no temporal: {type(e).__name__}: {e}")

    try:
        audit_path = observability.export_audit(step_name="api_temporal")
        audit_path_str = str(audit_path)
    except Exception:
        audit_path_str = ""

    bundle_json = None
    if include_bundle_json and audit_path_str:
        bundle_json = _bundle_from_audit_temporal(audit_path_str)

    payload = {
        "ok": bool(ok),
        "run_id": run_id,
        "talhao": talhao or "unknown_talhao",
        "source_csv": str(csv_path),
        "season_col": season_col,
        "report_text": report_text,
        "bundle_json": bundle_json,  # <- NOVO
        "warnings": list(warnings or []),
        "audit_path": audit_path_str,
    }
    return JSONResponse(content=payload)


@app.post("/run/individual")
def run_individual(
    file: UploadFile = File(...),
    talhao: Optional[str] = Form(default=None),
    season_col: str = Form(default="Safra_prod"),
    enable_cross_season_filter: bool = Form(default=True),
    top_k: int = Form(default=5),
    include_bundle_json: bool = Form(default=True),
) -> JSONResponse:
    """
    Individualiza por Safra_prod e roda intra-safra para cada safra.
    Retorna:
      - results: lista por safra com report_text + bundle_json (opcional)
    """
    run_id = observability.new_run_id()
    if talhao:
        observability.set_context(Talhao=str(talhao))

    csv_path = _save_upload_csv(file, run_id)

    try:
        results, warnings = runner.run_individual_seasons(
            csv_path,
            season_col=season_col,
            enable_cross_season_filter=enable_cross_season_filter,
            observability=observability,
            top_k=int(top_k),
        )
    except Exception as e:
        try:
            observability.record_decision(
                "api_individual_error",
                decision="fail",
                reason=str(e),
                metrics={"exc_type": type(e).__name__},
            )
            observability.export_audit(step_name="api_individual__error")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Erro no individual: {type(e).__name__}: {e}")

    try:
        audit_path = observability.export_audit(step_name="api_individual")
        audit_path_str = str(audit_path)
    except Exception:
        audit_path_str = ""

    seasons_payload: List[Dict[str, Any]] = []
    ok_any = False

    for r in results:
        ok_any = ok_any or bool(r.ok)

        bundle_json = None
        if include_bundle_json and audit_path_str:
            bundle_json = _bundle_from_audit_individual(audit_path_str, season=str(r.season))

        seasons_payload.append(
            {
                "season": r.season,
                "ok": bool(r.ok),
                "culture": r.culture,
                "n_rows": int(r.n_rows),
                "report_text": r.report_text,
                "bundle_json": bundle_json,  # <- NOVO
                "warnings": list(r.warnings or []),
                "metadata": dict(r.metadata or {}),
            }
        )

    payload = {
        "ok": bool(ok_any) if results else True,
        "run_id": run_id,
        "talhao": talhao or "unknown_talhao",
        "source_csv": str(csv_path),
        "season_col": season_col,
        "warnings": list(warnings or []),
        "results": seasons_payload,
        "audit_path": audit_path_str,
    }
    return JSONResponse(content=payload)
