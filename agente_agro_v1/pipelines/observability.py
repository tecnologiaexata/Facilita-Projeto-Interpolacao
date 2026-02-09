# -*- coding: utf-8 -*-
"""
observability.py

Instrumentação mínima e científica do pipeline de Inteligência Agronômica.

Objetivos:
- Rastrear decisões e etapas do pipeline
- Organizar auditorias por Talhão
- Diferenciar execuções automaticamente por run_id
"""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


# ------------------------
# Estado interno do run 
# ------------------------

_RUN_CONTEXT: Dict[str, Any] = {
    "run_id": None,
    "started_at": None,
    "events": [],
    "context": {},  
}



def new_run_id() -> str:
    run_id = str(uuid4())
    _RUN_CONTEXT["run_id"] = run_id
    _RUN_CONTEXT["started_at"] = datetime.utcnow().isoformat()
    _RUN_CONTEXT["events"] = []
    _RUN_CONTEXT["context"] = {}

    log(f"New run started: {run_id}")
    return run_id


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] [{level}] {message}")


def start_step(step_name: str, **context: Any) -> None:
    event = {
        "type": "step_start",
        "step": step_name,
        "timestamp": datetime.utcnow().isoformat(),
        "context": context,
    }
    _RUN_CONTEXT["events"].append(event)
    log(f"Starting step: {step_name}")


def record_decision(
    name: str,
    decision: Any,
    reason: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    event = {
        "type": "decision",
        "name": name,
        "decision": decision,
        "reason": reason,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }
    _RUN_CONTEXT["events"].append(event)
    log(f"Decision recorded: {name} -> {decision}")


# ----------------
# Contexto do run 
# ----------------

def set_context(**kwargs: Any) -> None:
    _RUN_CONTEXT["context"].update(kwargs)
    log(f"Run context updated: {list(kwargs.keys())}")


def get_context() -> Dict[str, Any]:
    """Retorna uma cópia do contexto atual do run."""
    return dict(_RUN_CONTEXT.get("context", {}))


# --------------------
# Utilidades internas
# --------------------

def _sanitize_for_path(value: str) -> str:
    value = str(value).strip()
    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_\-]+", "", value)
    return value or "unknown"


def hash_file(path: str | Path, algo: str = "md5") -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para hash: {path}")

    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    digest = h.hexdigest()
    log(f"Computed {algo} hash for {path.name}: {digest}")
    return digest


# ------------------
# Export do audit
# ------------------

def export_audit(
    *,
    step_name: str,
    group_by: str = "Talhao",
) -> Path:
    """
    Exporta o audit do run atual para JSON.

    Convenção:
    - 1 run = 1 arquivo
    - Sempre inclui run_id + step_name
    - Organiza por Talhão (se disponível no contexto) e sem acento para evitar problemas de codificação.

    Estrutura final:
    agente_agro/audit/<Talhão>/audit_<run_id>__<step_name>.json
    """
    if _RUN_CONTEXT["run_id"] is None:
        raise RuntimeError("Nenhuma execução ativa. Chame new_run_id() primeiro.")

    package_root = Path(__file__).resolve().parents[1]
    audit_root = package_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    ctx = _RUN_CONTEXT.get("context", {}) or {}

    group_value = ctx.get(group_by) or ctx.get("Talhao") or ctx.get("Talhao_id")
    if group_value is None:
        group_value = "unknown_talhao"

    group_folder = _sanitize_for_path(str(group_value))
    out_dir = audit_root / group_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = _RUN_CONTEXT["run_id"]
    step_safe = _sanitize_for_path(step_name)

    out_name = f"audit_{run_id}__{step_safe}.json"
    out_path = out_dir / out_name

    payload = {
        "run_id": run_id,
        "started_at": _RUN_CONTEXT["started_at"],
        "exported_at": datetime.utcnow().isoformat(),
        "context": ctx,
        "events": _RUN_CONTEXT["events"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log(f"Audit exported to {out_path.resolve()}")
    return out_path
