# -*- coding: utf-8 -*-
"""
insight_engine.py

Núcleo analítico observacional (intra-safra).

Princípios:
- Sem imputação, sem winsorization, sem "correção" dos dados.
- Produz evidências interpretáveis e auditáveis.
- Programado para dizer "nada robusto" quando não há sinal suficiente.

Defaults (hard-coded):
- yield classes: tercis (baixo/médio/alto)
- X (driver) binning: tercis
- efeito principal: Δ mediana de yield (Q3 - Q1) em kg/ha, e também Δ%

Saídas:
- lista de insights por variável (representantes de clusters)
- métricas + ressalvas
- registro completo no audit via observability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InsightItem:
    scope: str  
    var: str
    cluster_id: Optional[str]
    cluster_vars: Tuple[str, ...]
    var_type: str

    spearman_rho: Optional[float]
    delta_yield_q4_q1: Optional[float]        
    delta_yield_q4_q1_pct: Optional[float]   
    enrichment_top_yield_in_top_x: Optional[float]  

    delta_x_med_high_low: Optional[float]
    cliffs_delta: Optional[float]
    auc: Optional[float]
    mannwhitney_p: Optional[float]

    n_total: int
    n_valid: int
    outlier_frac: float
    warnings: Tuple[str, ...]

    driver_score: float

    details: Dict[str, Any]


@dataclass(frozen=True)
class InsightBundle:
    ok: bool
    scope: str
    yield_col: str
    analyzed_vars: Tuple[str, ...]
    insights: Tuple[InsightItem, ...]
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]


# -----------------------------
# Helpers 
# -----------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    """
    Spearman via ranks.
    """
    x = _safe_numeric(x)
    y = _safe_numeric(y)
    m = x.notna() & y.notna()
    if int(m.sum()) < 10:
        return None

    xr = x[m].rank(method="average")
    yr = y[m].rank(method="average")

    rho = float(xr.corr(yr))
    if not np.isfinite(rho):
        return None
    return rho


def _mannwhitney_pvalue(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Mann–Whitney U — p-value 2-sided.
    Usa SciPy se disponível; caso contrário, retorna None.
    """
    try:
        from scipy.stats import mannwhitneyu  
        if len(x) < 5 or len(y) < 5:
            return None
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return None


def _auc_from_ranks(x_pos: np.ndarray, x_neg: np.ndarray) -> Optional[float]:
    """
    AUC para separação pos vs neg usando ranks (equivalente ao U/(n_pos*n_neg)).
    Interpretação: probabilidade de um pos ter valor maior que um neg (com empates tratados por ranks).
    """
    n_pos = len(x_pos)
    n_neg = len(x_neg)
    if n_pos < 5 or n_neg < 5:
        return None

    all_vals = np.concatenate([x_pos, x_neg])
    ranks = pd.Series(all_vals).rank(method="average").to_numpy()

    r_pos = ranks[:n_pos].sum()

    U = r_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)

    return float(auc)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Cliff's delta (tamanho de efeito não-paramétrico):
      delta = P(x > y) - P(x < y)
    Intervalo [-1, 1]. 0 = nenhuma dominância.
    """
    n1 = len(x)
    n2 = len(y)
    if n1 < 5 or n2 < 5:
        return None

    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    denom = n1 * n2
    return float((gt - lt) / denom)


def _quantile_bins(s: pd.Series, q: int, labels: List[str]) -> pd.Series:
    """
    Quantile binning com fallback se houver poucos valores únicos.
    """
    s = _safe_numeric(s)
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        qs = s.quantile(np.linspace(0, 1, q + 1)).to_numpy()
        qs = np.unique(qs)
        if len(qs) <= 2:
            return pd.Series([np.nan] * len(s), index=s.index)
        return pd.cut(s, bins=qs, include_lowest=True)


def _driver_score(
    delta_yield_q4_q1: Optional[float],
    cliffs: Optional[float],
    rho: Optional[float],
    missing_frac: float,
    outlier_frac: float,
) -> float:
    """
    Score simples e auditável:
    - usa magnitude de efeito (cliffs) + monotonicidade (|rho|) + delta yield
    - penaliza missing e outliers (apenas para ranking, não exclui)
    """
    s = 0.0
    if cliffs is not None:
        s += abs(cliffs) * 1.2
    if rho is not None:
        s += abs(rho) * 0.8
    if delta_yield_q4_q1 is not None and np.isfinite(delta_yield_q4_q1):
        s += min(abs(delta_yield_q4_q1) / 500.0, 2.0) * 0.6

    quality_penalty = (1 - missing_frac) * (1 - min(outlier_frac, 0.5))
    return float(s * quality_penalty)


# -----------------------------
# Core
# -----------------------------

def run_intrasafra_insights(
    df_model: pd.DataFrame,
    io_report: Any,
    sq_report: Any,
    col_report: Any,
    *,
    yield_terciles: bool = True,
    x_terciles: int = 3,
    min_valid_n: int = 30,
    observability: Optional[Any] = None,
) -> InsightBundle:
    """
    Gera insights intra-safra (espacial) com duas vertentes:
      (1) efeito por tercis de X (Δ mediana yield)
      (2) contraste alto vs baixo yield (tercis) com Cliff's delta, AUC e Mann–Whitney

    Variáveis analisadas:
      - representantes de clusters (col_report.representatives)
      - variáveis temporais são excluídas (col_report.vars_temporal_heldout)
    """
    if observability is not None:
        observability.start_step("insight_engine_intrasafra", x_terciles=x_terciles, min_valid_n=min_valid_n)

    yield_col = io_report.yield_col
    y = _safe_numeric(df_model[yield_col])

    if yield_terciles:
        y_class = _quantile_bins(y, q=3, labels=["low", "mid", "high"])
    else:
        y_class = pd.Series(["all"] * len(df_model), index=df_model.index)

    reps: Dict[str, str] = dict(getattr(col_report, "representatives", {}) or {})
    clusters: List[List[str]] = list(getattr(col_report, "clusters", []) or [])
    temporal_heldout = set(getattr(col_report, "vars_temporal_heldout", []) or [])

    var_to_cluster: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    for i, cl in enumerate(clusters, start=1):
        cid = f"cluster_{i}"
        for v in cl:
            var_to_cluster[v] = (cid, tuple(cl))

    vars_to_analyze = [v for v in reps.values() if v not in temporal_heldout]
    vars_to_analyze = sorted(dict.fromkeys(vars_to_analyze))  

    char = getattr(sq_report, "variable_characterization", {}) or {}
    out_flags = getattr(sq_report, "outlier_flags", {}) or {}

    y_median = float(y.dropna().median()) if y.notna().sum() > 0 else float("nan")

    insights: List[InsightItem] = []
    bundle_warnings: List[str] = []

    mask_high = (y_class == "high")
    mask_low = (y_class == "low")

    for var in vars_to_analyze:
        if var not in df_model.columns:
            continue

        x = _safe_numeric(df_model[var])
        valid_mask = x.notna() & y.notna()
        n_valid = int(valid_mask.sum())
        n_total = int(len(df_model))
        x_valid = x[valid_mask]
        x_med = float(x_valid.median()) if len(x_valid) else float("nan")
        x_q10 = float(x_valid.quantile(0.10)) if len(x_valid) else float("nan")
        x_q90 = float(x_valid.quantile(0.90)) if len(x_valid) else float("nan")
        x_min = float(x_valid.min()) if len(x_valid) else float("nan")
        x_max = float(x_valid.max()) if len(x_valid) else float("nan")
        x_iqr = float(x_valid.quantile(0.75) - x_valid.quantile(0.25)) if len(x_valid) else float("nan")
        x_range = float(x_max - x_min) if np.isfinite(x_max) and np.isfinite(x_min) else float("nan")
        x_std = float(x_valid.std(ddof=1)) if len(x_valid) > 1 else float("nan")
        x_cv = float((x_std / x_med) * 100.0) if np.isfinite(x_std) and np.isfinite(x_med) and x_med != 0 else None
    
        warnings: List[str] = []
        if n_valid < min_valid_n:
            warnings.append(f"Poucos dados válidos para '{var}' (n_valid={n_valid} < {min_valid_n}). Resultados podem ser instáveis.")

        if x_cv is not None and x_cv < 5.0:
            warnings.append(
                f"Variável '{var}' tem baixa variabilidade (CV={x_cv:.2f}%). "
                "Diferenças podem ser pouco relevantes operacionalmente; interpretar com cautela."
            )

        of = out_flags.get(var, {}) or {}
        out_frac = float(of.get("frac_outliers", 0.0))
        out_n = int(of.get("n_outliers", 0))
        if out_n > 0:
            warnings.append(f"Outliers detectados em '{var}' (n={out_n}, frac={out_frac:.3f}). Sugere-se checar amostras.")

        vtype = (char.get(var, {}) or {}).get("type", "unknown")

        rho = _spearman(x, y)

        x_bins = _quantile_bins(x, q=x_terciles, labels=[f"T{i}" for i in range(1, x_terciles + 1)])
        by_bin = pd.DataFrame({"xbin": x_bins, "y": y}).dropna().groupby("xbin", observed=True)["y"].median()
        
        delta_y = None
        delta_y_pct = None
        if len(by_bin) >= 2 and "T1" in by_bin.index and f"T{x_terciles}" in by_bin.index:
            delta_y = float(by_bin.loc[f"T{x_terciles}"] - by_bin.loc["T1"])
            if np.isfinite(y_median) and y_median != 0:
                delta_y_pct = float((delta_y / y_median) * 100.0)
        
        enrichment = None
        if yield_terciles:
            m1 = mask_high & x_bins.notna()
            if int(m1.sum()) >= 10:
                frac_in_top_x = float((x_bins[m1] == f"T{x_terciles}").mean())
                baseline = 1.0 / x_terciles
                if baseline > 0:
                    enrichment = float(frac_in_top_x / baseline)

        delta_x_med = None
        cliffs = None
        auc = None
        pval = None

        if yield_terciles:
            x_high = x[mask_high & x.notna()].to_numpy()
            x_low = x[mask_low & x.notna()].to_numpy()

            if len(x_high) >= 5 and len(x_low) >= 5:
                delta_x_med = float(np.median(x_high) - np.median(x_low))
                cliffs = _cliffs_delta(x_high, x_low)
                auc = _auc_from_ranks(x_high, x_low)
                pval = _mannwhitney_pvalue(x_high, x_low)

        missing_frac = float(1 - (n_valid / n_total)) if n_total > 0 else 1.0
        score = _driver_score(delta_y, cliffs, rho, missing_frac, out_frac)

        cid, cvars = var_to_cluster.get(var, (None, (var,)))

        cluster_size = len(cvars)

        details = {
            "yield_median": y_median,
            "yield_terciles_used": yield_terciles,
            "x_terciles": x_terciles,
            "yield_by_x_tercile_median": {str(k): float(v) for k, v in by_bin.to_dict().items()} if len(by_bin) else {},
            "cluster_size": cluster_size,
            "outlier_method": (of.get("method") or "N/A"),
            "x_median": x_med,
            "x_iqr": x_iqr,
            "x_q10": x_q10,
            "x_q90": x_q90,
            "x_min": x_min,
            "x_max": x_max,
            "x_range": x_range,
            "x_cv_pct": x_cv,
        }

        insights.append(
            InsightItem(
                scope="intrasafra",
                var=var,
                cluster_id=cid,
                cluster_vars=tuple(cvars),
                var_type=vtype,
                spearman_rho=rho,
                delta_yield_q4_q1=delta_y,
                delta_yield_q4_q1_pct=delta_y_pct,
                enrichment_top_yield_in_top_x=enrichment,
                delta_x_med_high_low=delta_x_med,
                cliffs_delta=cliffs,
                auc=auc,
                mannwhitney_p=pval,
                n_total=n_total,
                n_valid=n_valid,
                outlier_frac=out_frac,
                warnings=tuple(warnings),
                driver_score=score,
                details=details,
            )
        )

    insights_sorted = sorted(insights, key=lambda it: it.driver_score, reverse=True)

    if len(vars_to_analyze) == 0:
        bundle_warnings.append("Nenhuma variável disponível para análise intra-safra (após filtros/representantes).")

    # Audit
    if observability is not None:
        observability.record_decision(
            "insight_engine_intrasafra_summary",
            decision="pass",
            reason="insights intra-safra gerados",
            metrics={
                "n_vars_analyzed": len(vars_to_analyze),
                "n_insights": len(insights_sorted),
                "x_terciles": x_terciles,
                "yield_groups": "terciles" if yield_terciles else "none",
                "min_valid_n": min_valid_n,
                "top_drivers": [
                    {"var": it.var, "score": it.driver_score, "rho": it.spearman_rho, "delta_y": it.delta_yield_q4_q1}
                    for it in insights_sorted[:10]
                ],
                "bundle_warnings": bundle_warnings,
            },
        )

        observability.record_decision(
            "insight_engine_intrasafra_full",
            decision="insights",
            reason="lista completa de insights e métricas",
            metrics={
                "insights": [
                    {
                        "var": it.var,
                        "cluster_id": it.cluster_id,
                        "cluster_vars": list(it.cluster_vars),
                        "var_type": it.var_type,
                        "spearman_rho": it.spearman_rho,
                        "delta_yield_q4_q1": it.delta_yield_q4_q1,
                        "delta_yield_q4_q1_pct": it.delta_yield_q4_q1_pct,
                        "enrichment_top_yield_in_top_x": it.enrichment_top_yield_in_top_x,
                        "delta_x_med_high_low": it.delta_x_med_high_low,
                        "cliffs_delta": it.cliffs_delta,
                        "auc": it.auc,
                        "mannwhitney_p": it.mannwhitney_p,
                        "n_valid": it.n_valid,
                        "outlier_frac": it.outlier_frac,
                        "warnings": list(it.warnings),
                        "driver_score": it.driver_score,
                        "details": it.details,
                    }
                    for it in insights_sorted
                ]
            },
        )

    return InsightBundle(
        ok=True,
        scope="intrasafra",
        yield_col=yield_col,
        analyzed_vars=tuple(vars_to_analyze),
        insights=tuple(insights_sorted),
        warnings=tuple(bundle_warnings),
        metadata={
            "x_terciles": x_terciles,
            "yield_groups": "terciles" if yield_terciles else "none",
            "min_valid_n": min_valid_n,
            "note": "temporal insights not implemented in v1",
        },
    )


def summarize_bundle(bundle: InsightBundle, top_k: int = 8) -> str:
    lines = [
        f"Insight engine OK | scope={bundle.scope} | yield='{bundle.yield_col}'",
        f"Vars analyzed: {len(bundle.analyzed_vars)} | insights: {len(bundle.insights)}",
    ]
    if bundle.warnings:
        lines.append("Bundle warnings:")
        lines.extend([f" - {w}" for w in bundle.warnings])

    lines.append(f"Top {min(top_k, len(bundle.insights))} drivers (score-based):")
    for it in bundle.insights[:top_k]:
        rho = "NA" if it.spearman_rho is None else f"{it.spearman_rho:+.2f}"
        dy = "NA" if it.delta_yield_q4_q1 is None else f"{it.delta_yield_q4_q1:+.0f} kg/ha"
        enr = "NA" if it.enrichment_top_yield_in_top_x is None else f"{it.enrichment_top_yield_in_top_x:.2f}x"
        cd = "NA" if it.cliffs_delta is None else f"{it.cliffs_delta:+.2f}"
        auc = "NA" if it.auc is None else f"{it.auc:.2f}"
        p = "NA" if it.mannwhitney_p is None else f"{it.mannwhitney_p:.3g}"
        lines.append(
            f" - {it.var} | score={it.driver_score:.2f} | rho={rho} | Δy(Q3-Q1)={dy} | enrich={enr} | Cliff={cd} | AUC={auc} | p={p}"
        )
    return "\n".join(lines)
