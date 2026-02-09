# -*- coding: utf-8 -*-
"""
io_checks.py

Importa o grid_completo.csv gerado pela plataforma e aplica checagens mínimas e
conservadoras para liberar o pipeline de Inteligência Agronômica.

Regras:
- Só roda se existir produtividade com dados suficientes + variabilidade.
- Só roda se existir pelo menos 1 "processo" com pelo menos 1 variável EXPLICATIVA NUMÉRICA
  com variabilidade útil (espacial e/ou temporal).
- Não refaz validações profundas do gerador do grid_completo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class InputDataError(ValueError):
    """Erro de entrada: não atende pré-requisitos mínimos para rodar o pipeline."""


@dataclass(frozen=True)
class IOCheckReport:
    ok: bool
    source_file: str
    n_rows: int
    n_cols: int
    yield_col: str
    yield_non_null: int
    yield_unique: int
    yield_std: float
    yield_cv: float
    processes_found: Tuple[str, ...]
    informative_processes: Tuple[str, ...]
    informative_vars: Tuple[str, ...]
    non_numeric_excluded: int
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _cv(x: pd.Series) -> float:
    x = _safe_numeric(x).dropna()
    if len(x) < 2:
        return float("nan")
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if mu == 0:
        return float("inf") if sd > 0 else 0.0
    return abs(sd / mu)


def _infer_processes(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    processes: Dict[str, Dict[str, Any]] = {}

    for c in df.columns:
        if c.startswith("Data_") and len(c) > len("Data_"):
            proc = c.split("Data_", 1)[1].strip()
            if proc:
                processes.setdefault(proc, {"date_col": c, "vars": []})

    for proc in list(processes.keys()):
        suffix = f"_{proc}"
        vars_proc = [c for c in df.columns if c.endswith(suffix) and c != processes[proc]["date_col"]]
        processes[proc]["vars"] = vars_proc

    known_procs = set(processes.keys())
    for c in df.columns:
        if "_" in c and not c.startswith("Data_"):
            proc = c.rsplit("_", 1)[1]
            if proc and proc not in known_procs:
                suffix = f"_{proc}"
                vars_proc = [cc for cc in df.columns if cc.endswith(suffix) and not cc.startswith("Data_")]
                if len(vars_proc) >= 2:
                    processes.setdefault(proc, {"date_col": None, "vars": vars_proc})

    return processes


def _infer_yield_col(df: pd.DataFrame) -> str:
    candidates = ["Kg/Ha_prod", "yield", "yld", "produtividade"]
    for c in candidates:
        if c in df.columns:
            return c
    raise InputDataError(
        "Não foi possível localizar coluna de produtividade (yield). "
        "Esperado 'Kg/Ha_prod' (padrão da plataforma)."
    )


def _is_numeric_informative(
    df: pd.DataFrame,
    var_col: str,
    date_col: Optional[str],
    *,
    min_non_null: int,
    min_unique_spatial: int,
    min_unique_temporal: int,
) -> Tuple[bool, Dict[str, Any]]:
    s = df[var_col]
    nn = int(s.notna().sum())
    if nn < min_non_null:
        return False, {"non_null": nn, "reason": "too_few_values"}

    if not pd.api.types.is_numeric_dtype(s):
        return False, {"non_null": nn, "reason": "non_numeric_excluded"}

    x = _safe_numeric(s).dropna()
    uniq_all = int(x.nunique())

    if date_col is None or date_col not in df.columns:
        return (uniq_all >= min_unique_spatial), {
            "non_null": nn, "unique": uniq_all, "mode": "spatial_no_date"
        }

    tmp = df[[date_col, var_col]].copy()
    tmp[date_col] = tmp[date_col].astype("string")
    tmp[var_col] = _safe_numeric(tmp[var_col])

    by_date = tmp.dropna().groupby(date_col)[var_col]
    if by_date.ngroups == 0:
        return False, {"non_null": nn, "reason": "no_valid_date_groups"}

    spatial_uniqs = by_date.nunique(dropna=True)
    max_spatial_uniq = int(spatial_uniqs.max()) if not spatial_uniqs.empty else 0
    if max_spatial_uniq >= min_unique_spatial:
        return True, {
            "non_null": nn,
            "max_unique_within_date": max_spatial_uniq,
            "mode": "spatial_within_date",
        }

    means = by_date.mean()
    uniq_temporal = int(means.dropna().nunique())
    if uniq_temporal >= min_unique_temporal:
        return True, {
            "non_null": nn,
            "unique_between_dates_mean": uniq_temporal,
            "mode": "temporal_between_dates",
        }

    return False, {
        "non_null": nn,
        "max_unique_within_date": max_spatial_uniq,
        "unique_between_dates_mean": uniq_temporal,
        "mode": "not_informative",
    }


def load_and_check_grid(
    csv_path: str | Path = "data/grid_completo/pivo_grid_completo.csv",
    *,
    yield_min_non_null_frac: float = 0.70,
    yield_min_unique: int = 10,
    yield_min_cv: float = 0.02,
    min_informative_vars: int = 1,
    min_var_non_null: int = 30,
    min_unique_spatial: int = 5,
    min_unique_temporal: int = 2,
    observability: Optional[Any] = None,
) -> Tuple[pd.DataFrame, IOCheckReport]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise InputDataError(f"Arquivo não encontrado: {csv_path}")
    
    if observability is not None:
        observability.start_step("io_checks", file=str(csv_path))
    
    df = pd.read_csv(csv_path)
    
    if observability is not None and "Talhao" in df.columns:
        talhoes = df["Talhao"].dropna().astype(str).unique()
        if len(talhoes) == 1:
            observability.set_context(Talhao=talhoes[0])
        elif len(talhoes) > 1:
            observability.set_context(Talhao="MULTI_" + "_".join(sorted(talhoes)))
        else:
            observability.set_context(Talhao="unknown_talhao")
    
    warnings: List[str] = []

    required_fixed = ["id_ponto", "Lat", "Long"]
    missing_fixed = [c for c in required_fixed if c not in df.columns]
    if missing_fixed:
        raise InputDataError(f"Grid inválido: faltam colunas fixas mínimas: {missing_fixed}")

    yield_col = _infer_yield_col(df)
    y = _safe_numeric(df[yield_col])

    yield_non_null = int(y.notna().sum())
    if yield_non_null < int(len(df) * yield_min_non_null_frac):
        raise InputDataError(
            f"Yield insuficiente na coluna '{yield_col}': "
            f"{yield_non_null}/{len(df)} não-nulos (mínimo {yield_min_non_null_frac:.0%})."
        )

    yield_unique = int(y.dropna().nunique())
    yield_std = float(y.dropna().std(ddof=1))
    yield_cv = _cv(y)

    if yield_unique < yield_min_unique:
        raise InputDataError(
            f"Yield sem variabilidade suficiente: {yield_unique} valores únicos "
            f"em '{yield_col}' (mínimo {yield_min_unique})."
        )
    if not np.isfinite(yield_cv) or yield_cv < yield_min_cv:
        raise InputDataError(
            f"Yield com variabilidade muito baixa: CV={yield_cv:.4f} em '{yield_col}' "
            f"(mínimo {yield_min_cv})."
        )

    if observability is not None:
        observability.record_decision(
            "yield_check",
            decision="pass",
            reason="yield presente e variável",
            metrics={
                "yield_col": yield_col,
                "non_null": yield_non_null,
                "unique": yield_unique,
                "std": yield_std,
                "cv": yield_cv,
            },
        )

    processes = _infer_processes(df)
    processes_found = tuple(sorted(processes.keys()))
    if not processes_found:
        raise InputDataError("Nenhum processo (Data_<proc>) encontrado no grid. Não há variáveis explicativas.")

    informative_vars: List[str] = []
    informative_processes: set[str] = set()
    var_diagnostics: Dict[str, Any] = {}
    non_numeric_excluded = 0

    for proc, info in processes.items():
        date_col = info.get("date_col")
        vars_proc_all = [c for c in info.get("vars", []) if c != yield_col]

        vars_proc_numeric = []
        for vc in vars_proc_all:
            if pd.api.types.is_numeric_dtype(df[vc]):
                vars_proc_numeric.append(vc)
            else:
                non_numeric_excluded += 1

        if not vars_proc_numeric:
            continue

        for vc in vars_proc_numeric:
            ok, diag = _is_numeric_informative(
                df,
                vc,
                date_col,
                min_non_null=min_var_non_null,
                min_unique_spatial=min_unique_spatial,
                min_unique_temporal=min_unique_temporal,
            )
            var_diagnostics[vc] = {"proc": proc, "date_col": date_col, **diag}
            if ok:
                informative_vars.append(vc)
                informative_processes.add(proc)

    if len(informative_vars) < min_informative_vars:
        msg = (
            "Não há variáveis explicativas NUMÉRICAS com variabilidade útil suficiente. "
            "Pré-requisito: ao menos 1 processo registrado com variáveis numéricas informativas."
        )
        if observability is not None:
            observability.record_decision(
                "explanatory_vars_check",
                decision="fail",
                reason=msg,
                metrics={"processes_found": list(processes_found)},
            )
        raise InputDataError(msg)

    if non_numeric_excluded > 0:
        warnings.append(f"{non_numeric_excluded} variáveis foram ignoradas por não serem numéricas (conservador).")

    if observability is not None:
        observability.record_decision(
            "explanatory_vars_check",
            decision="pass",
            reason="há variáveis explicativas numéricas informativas",
            metrics={
                "informative_vars_count": len(informative_vars),
                "informative_processes": sorted(informative_processes),
                "non_numeric_excluded": non_numeric_excluded,
            },
        )

    const_like = [k for k, d in var_diagnostics.items() if d.get("mode") == "not_informative"]
    if len(const_like) > 0:
        warnings.append(f"{len(const_like)} variáveis numéricas foram consideradas não-informativas (sem variabilidade útil).")

    report = IOCheckReport(
        ok=True,
        source_file=str(csv_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        yield_col=yield_col,
        yield_non_null=yield_non_null,
        yield_unique=yield_unique,
        yield_std=yield_std,
        yield_cv=float(yield_cv),
        processes_found=processes_found,
        informative_processes=tuple(sorted(informative_processes)),
        informative_vars=tuple(sorted(set(informative_vars))),
        non_numeric_excluded=int(non_numeric_excluded),
        warnings=tuple(warnings),
        metadata={"var_diagnostics_sample": dict(list(var_diagnostics.items())[:15])},
    )

    return df, report


def summarize_report(report: IOCheckReport) -> str:
    lines = [
        f"IO checks OK | rows={report.n_rows} cols={report.n_cols}",
        f"Yield: col='{report.yield_col}' non_null={report.yield_non_null} "
        f"unique={report.yield_unique} std={report.yield_std:.3f} cv={report.yield_cv:.3f}",
        f"Processes found: {', '.join(report.processes_found) if report.processes_found else '(none)'}",
        f"Informative processes: {', '.join(report.informative_processes) if report.informative_processes else '(none)'}",
        f"Informative vars (n={len(report.informative_vars)}): "
        f"{', '.join(list(report.informative_vars)[:12])}{'...' if len(report.informative_vars) > 12 else ''}",
        f"Non-numeric excluded: {report.non_numeric_excluded}",
    ]
    if report.warnings:
        lines.append("Warnings:")
        lines.extend([f" - {w}" for w in report.warnings])
    return "\n".join(lines)
