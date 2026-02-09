# -*- coding: utf-8 -*-
"""
safety_quality.py

Objetivo:
- Seleção mínima e conservadora de variáveis numéricas para análise
- Estatísticas descritivas (especialmente: espacial vs temporal vs mista)
- Flag/registro de outliers (sem alterar dados, sem winsorizar, sem imputar)

Princípios:
- Yield não é alterado. Confia no processo de geração (filtro + interpolação)
- Outliers são apenas sinalizados (para warnings em recomendações).
- Sem redundâncias profundas: assume que io_checks já garantiu pré-requisitos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SafetyQualityReport:
    ok: bool
    n_rows: int
    yield_col: str
    selected_vars: Tuple[str, ...]
    dropped_vars: Dict[str, str]  
    variable_characterization: Dict[str, Dict[str, Any]]  
    outlier_flags: Dict[str, Dict[str, Any]]  
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]


# ------------
# Helpers
# ------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _iqr(x: pd.Series) -> float:
    x = _safe_numeric(x).dropna()
    if x.empty:
        return float("nan")
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    return q3 - q1


def _modified_z_outliers(x: pd.Series, threshold: float = 3.5) -> Tuple[int, float]:
    """
    Outliers via modified z-score.
    Retorna (n_outliers, frac_outliers). Apenas para flag, não para remoção.
    """
    x = _safe_numeric(x).dropna()
    n = int(len(x))
    if n < 10:
        return 0, 0.0

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    if mad == 0:
        return 0, 0.0

    mz = 0.6745 * (x - med) / mad
    out = np.abs(mz) > threshold
    n_out = int(out.sum())
    return n_out, float(n_out / n)


def _infer_process_and_datecol(df: pd.DataFrame, var: str) -> Tuple[Optional[str], Optional[str]]:
    if "_" not in var:
        return None, None
    proc = var.rsplit("_", 1)[1]
    date_col = f"Data_{proc}"
    if date_col in df.columns:
        return proc, date_col
    return proc, None


def _characterize_var(df: pd.DataFrame, var: str) -> Dict[str, Any]:
    """
    Caracteriza variável como:
      - spatial: varia dentro de datas (quando date_col existe)
      - temporal: quase constante dentro de datas, mas muda entre datas
      - mixed: ambos
      - spatial_no_date: sem data, mas tem variabilidade global
      - constant_like: baixa variabilidade
    """
    proc, date_col = _infer_process_and_datecol(df, var)

    s = df[var]
    x = _safe_numeric(s)

    non_null = int(x.notna().sum())
    uniq_all = int(x.dropna().nunique())
    std = float(x.dropna().std(ddof=1)) if non_null >= 2 else float("nan")
    iqr = _iqr(x)

    info: Dict[str, Any] = {
        "proc": proc,
        "date_col": date_col,
        "non_null": non_null,
        "unique_all": uniq_all,
        "std": std,
        "iqr": iqr,
    }

    if date_col is None or date_col not in df.columns:
        if uniq_all <= 3:
            info["type"] = "constant_like"
        else:
            info["type"] = "spatial_no_date"
        return info

    tmp = df[[date_col, var]].copy()
    tmp[date_col] = tmp[date_col].astype("string")
    tmp[var] = _safe_numeric(tmp[var])

    by_date = tmp.dropna().groupby(date_col)[var]
    n_dates = int(by_date.ngroups)
    info["n_dates"] = n_dates

    if n_dates == 0:
        info["type"] = "constant_like"
        return info

    uniq_within = by_date.nunique(dropna=True)
    max_uniq_within = int(uniq_within.max()) if not uniq_within.empty else 0
    info["max_unique_within_date"] = max_uniq_within

    means = by_date.mean()
    uniq_between_means = int(means.dropna().nunique())
    info["unique_between_dates_mean"] = uniq_between_means

    spatial = max_uniq_within > 3
    temporal = uniq_between_means >= 2

    if spatial and temporal:
        info["type"] = "mixed"
    elif spatial:
        info["type"] = "spatial"
    elif temporal:
        info["type"] = "temporal"
    else:
        info["type"] = "constant_like"

    return info


# --------------
# Main
# --------------

def apply_safety_quality(
    df: pd.DataFrame,
    io_report: Any,
    *,
    unique_min_keep: int = 4,          
    outlier_threshold_mz: float = 3.5, 
    observability: Optional[Any] = None,
) -> Tuple[pd.DataFrame, SafetyQualityReport]:
    """
    Aplica filtros e gera report de qualidade/segurança.

    - Seleciona vars a partir de io_report.informative_vars (numérico-only)
    - Remove variáveis com baixa variabilidade (unique <= 3)
    - Caracteriza cada variável (spatial/temporal/mixed)
    - Flag outliers (sem remover / sem alterar)

    Retorna:
      df_model: dataframe com yield + selected_vars (+ colunas de referência se existirem)
      report
    """
    if observability is not None:
        observability.start_step("safety_quality")

    yield_col = io_report.yield_col
    candidate_vars = list(io_report.informative_vars)

    dropped: Dict[str, str] = {}
    selected: List[str] = []
    characterization: Dict[str, Dict[str, Any]] = {}
    outliers: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    y = _safe_numeric(df[yield_col])
    y_non_null = int(y.notna().sum())
    y_unique = int(y.dropna().nunique())
    y_std = float(y.dropna().std(ddof=1)) if y_non_null >= 2 else float("nan")
    y_iqr = _iqr(y)

    y_out_n, y_out_frac = _modified_z_outliers(y, threshold=outlier_threshold_mz)
    outliers[yield_col] = {"n_outliers": y_out_n, "frac_outliers": y_out_frac, "method": "modified_z_mad"}

    if y_out_n > 0:
        warnings.append(
            f"Yield possui {y_out_n} potenciais outliers (flag apenas; sem correção)."
        )

    for var in candidate_vars:
        if var not in df.columns:
            dropped[var] = "missing_column"
            continue

        x = _safe_numeric(df[var])
        uniq = int(x.dropna().nunique())

        if uniq < unique_min_keep:
            dropped[var] = f"low_variability_unique<{unique_min_keep}"
            continue

        characterization[var] = _characterize_var(df, var)

        n_out, frac_out = _modified_z_outliers(x, threshold=outlier_threshold_mz)
        outliers[var] = {
            "n_outliers": n_out,
            "frac_outliers": frac_out,
            "method": "modified_z_mad",
        }

        if n_out > 0:
            warnings.append(
                f"Variável '{var}' possui {n_out} potenciais outliers (flag apenas)."
            )

        selected.append(var)

    # Audit
    if observability is not None:
        if dropped:
            observability.record_decision(
                "safety_quality_drops",
                decision="dropped_vars",
                reason="remoção por baixa variabilidade ou coluna ausente",
                metrics={
                    "dropped_count": len(dropped),
                    "examples": dict(list(dropped.items())[:10]),
                },
            )

        observability.record_decision(
            "safety_quality_summary",
            decision="pass",
            reason="seleção/flags concluídos",
            metrics={
                "candidate_vars": len(candidate_vars),
                "selected_vars": len(selected),
                "dropped_vars": len(dropped),
                "yield_non_null": y_non_null,
                "yield_unique": y_unique,
                "yield_std": y_std,
                "yield_iqr": y_iqr,
            },
        )

    keep_cols = [yield_col] + selected
    for c in ["Talhao", "id_ponto", "Lat", "Long"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.insert(0, c)

    df_model = df[keep_cols].copy()

    rep = SafetyQualityReport(
        ok=True,
        n_rows=int(df.shape[0]),
        yield_col=yield_col,
        selected_vars=tuple(selected),
        dropped_vars=dropped,
        variable_characterization=characterization,
        outlier_flags=outliers,
        warnings=tuple(warnings),
        metadata={
            "outlier_threshold_mz": outlier_threshold_mz,
            "unique_min_keep": unique_min_keep,
        },
    )

    if observability is not None and warnings:
        observability.record_decision(
            "safety_quality_warnings",
            decision="warning",
            reason="foram identificadas ressalvas nos dados (sem correção automática)",
            metrics={
                "n_warnings": len(warnings),
                "examples": warnings,
                "outlier_flags": outliers,
            },
        )

    return df_model, rep


def summarize_report(report: SafetyQualityReport, max_vars: int = 12) -> str:
    lines = [
        f"Safety quality OK | rows={report.n_rows}",
        f"Yield='{report.yield_col}' | selected_vars={len(report.selected_vars)} | dropped_vars={len(report.dropped_vars)}",
    ]
    if report.selected_vars:
        lines.append(
            "Selected (sample): " + ", ".join(list(report.selected_vars)[:max_vars]) +
            ("..." if len(report.selected_vars) > max_vars else "")
        )
    if report.variable_characterization:
        types = [v.get("type", "NA") for v in report.variable_characterization.values()]
        counts = {t: types.count(t) for t in sorted(set(types))}
        lines.append("Var types: " + ", ".join([f"{k}={v}" for k, v in counts.items()]))

    if report.warnings:
        lines.append("Warnings:")
        lines.extend([f" - {w}" for w in report.warnings[:10]])
        if len(report.warnings) > 10:
            lines.append(f" - (+{len(report.warnings)-10} mais warnings)")
    return "\n".join(lines)
