# -*- coding: utf-8 -*-
"""
collinearity.py

Objetivo:
- Identificar redundância forte entre variáveis (colinearidade) de forma conservadora.
- Agrupar variáveis altamente correlacionadas em clusters (grafo).
- Selecionar 1 representante por cluster para modelagem (sem olhar yield).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CollinearityReport:
    ok: bool
    method: str
    threshold: float
    min_pairwise_n: int
    vars_in: Tuple[str, ...]               
    vars_temporal_heldout: Tuple[str, ...] 
    pairs_strong: List[Dict[str, Any]]     
    clusters: List[List[str]]              
    representatives: Dict[str, str]        
    var_metrics: Dict[str, Dict[str, Any]] 
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _iqr(x: pd.Series) -> float:
    x = _safe_numeric(x).dropna()
    if x.empty:
        return float("nan")
    return float(x.quantile(0.75) - x.quantile(0.25))


def _pairwise_valid_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna matriz NxN com contagem de pares válidos (ambos não-nulos) por coluna.
    """
    m = df.notna().astype(np.int16)
    counts = m.T @ m
    counts = counts.astype(int)
    counts.index = df.columns
    counts.columns = df.columns
    return counts


def _build_edges(
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    threshold: float,
    min_pairwise_n: int,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """
    Cria lista de adjacência e lista de pares fortes.
    """
    cols = list(corr.columns)
    adj: Dict[str, List[str]] = {c: [] for c in cols}
    pairs: List[Dict[str, Any]] = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            n = int(counts.loc[a, b])
            if n < min_pairwise_n:
                continue
            rho = float(corr.loc[a, b])
            if not np.isfinite(rho):
                continue
            if abs(rho) >= threshold:
                adj[a].append(b)
                adj[b].append(a)
                pairs.append({"var1": a, "var2": b, "rho": rho, "n": n})

    return adj, pairs


def _connected_components(adj: Dict[str, List[str]]) -> List[List[str]]:
    """
    Componentes conexas em grafo não-direcionado via DFS.
    """
    visited = set()
    components: List[List[str]] = []

    for node in adj.keys():
        if node in visited:
            continue
        stack = [node]
        comp = []
        visited.add(node)

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        components.append(sorted(comp))

    components.sort(key=lambda x: (-len(x), x))
    return components


def _type_priority(var_type: str) -> int:
    """
    Preferência por "maior é melhor".
    """
    if var_type == "mixed":
        return 3
    if var_type == "spatial":
        return 2
    if var_type in ("spatial_no_date",):
        return 1
    return 0


def _compute_var_metrics(
    df: pd.DataFrame,
    vars_in: List[str],
    sq_report: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Gera métricas para seleção de representantes, sem olhar yield.
    Usa:
      - missing_frac
      - outlier frac (se disponível)
      - type (spatial/mixed) (se disponível)
      - iqr
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    char = getattr(sq_report, "variable_characterization", {}) or {}
    out_flags = getattr(sq_report, "outlier_flags", {}) or {}

    n_rows = int(df.shape[0])

    for v in vars_in:
        x = _safe_numeric(df[v])
        non_null = int(x.notna().sum())
        missing_frac = float(1 - (non_null / n_rows)) if n_rows > 0 else 1.0
        iqr = _iqr(x)

        vtype = (char.get(v, {}) or {}).get("type", "unknown")
        out_frac = float((out_flags.get(v, {}) or {}).get("frac_outliers", 0.0))
        out_n = int((out_flags.get(v, {}) or {}).get("n_outliers", 0))

        metrics[v] = {
            "non_null": non_null,
            "missing_frac": missing_frac,
            "iqr": iqr,
            "type": vtype,
            "type_priority": _type_priority(vtype),
            "outlier_frac": out_frac,
            "outlier_n": out_n,
        }

    return metrics


def _choose_representative(cluster: List[str], var_metrics: Dict[str, Dict[str, Any]]) -> str:
    """
    Heurística objetiva:
      1) menor missing_frac
      2) menor outlier_frac
      3) maior type_priority (mixed > spatial)
      4) maior iqr (desempate)
      5) nome (estável)
    """
    def key(v: str):
        m = var_metrics.get(v, {})
        return (
            m.get("missing_frac", 1.0),
            m.get("outlier_frac", 1.0),
            -m.get("type_priority", 0),
            -m.get("iqr", float("nan")) if np.isfinite(m.get("iqr", float("nan"))) else float("inf"),
            v,
        )

    return sorted(cluster, key=key)[0]


# -----------------------------
# API principal
# -----------------------------

def run_collinearity(
    df_model: pd.DataFrame,
    sq_report: Any,
    *,
    method: str = "spearman",
    threshold: float = 0.90,
    min_pairwise_n: int = 30,
    observability: Optional[Any] = None,
) -> CollinearityReport:
    """
    Executa colinearidade em vars spatial/mixed e devolve clusters + representantes.
    """
    if observability is not None:
        observability.start_step("collinearity", method=method, threshold=threshold, min_pairwise_n=min_pairwise_n)

    selected_vars = list(getattr(sq_report, "selected_vars", []) or [])
    char = getattr(sq_report, "variable_characterization", {}) or {}

    vars_in: List[str] = []
    vars_temporal: List[str] = []
    for v in selected_vars:
        vtype = (char.get(v, {}) or {}).get("type", "unknown")
        if vtype == "temporal":
            vars_temporal.append(v)
        else:
            vars_in.append(v)

    warnings: List[str] = []
    if len(vars_in) < 2:
        warnings.append("Poucas variáveis para colinearidade (vars_in < 2). Clustering não aplicável.")

        var_metrics = _compute_var_metrics(df_model, vars_in, sq_report)
        clusters = [sorted(vars_in)] if vars_in else []
        reps = {"cluster_1": vars_in[0]} if len(vars_in) == 1 else {}

        rep = CollinearityReport(
            ok=True,
            method=method,
            threshold=threshold,
            min_pairwise_n=min_pairwise_n,
            vars_in=tuple(vars_in),
            vars_temporal_heldout=tuple(sorted(vars_temporal)),
            pairs_strong=[],
            clusters=clusters,
            representatives=reps,
            var_metrics=var_metrics,
            warnings=tuple(warnings),
            metadata={"note": "collinearity skipped due to insufficient variables"},
        )

        if observability is not None:
            observability.record_decision(
                "collinearity_summary",
                decision="skip",
                reason="insufficient variables for clustering",
                metrics={
                    "vars_in": len(vars_in),
                    "vars_temporal_heldout": len(vars_temporal),
                    "warnings": warnings,
                },
            )

        return rep

    X = df_model[vars_in].apply(_safe_numeric)
    corr = X.corr(method=method)
    counts = _pairwise_valid_counts(X)
    adj, pairs = _build_edges(corr, counts, threshold, min_pairwise_n)
    clusters = _connected_components(adj)

    var_metrics = _compute_var_metrics(X, vars_in, sq_report)

    representatives: Dict[str, str] = {}
    for idx, cl in enumerate(clusters, start=1):
        cid = f"cluster_{idx}"
        representatives[cid] = _choose_representative(cl, var_metrics)

    # audit
    if observability is not None:
        observability.record_decision(
            "collinearity_summary",
            decision="pass",
            reason="clusters e representantes definidos",
            metrics={
                "method": method,
                "threshold": threshold,
                "min_pairwise_n": min_pairwise_n,
                "vars_in": len(vars_in),
                "vars_temporal_heldout": len(vars_temporal),
                "n_pairs_strong": len(pairs),
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
            },
        )

        observability.record_decision(
            "collinearity_clusters",
            decision="clusters",
            reason="componentes conexas do grafo de correlação forte",
            metrics={"clusters": clusters},
        )
        observability.record_decision(
            "collinearity_representatives",
            decision="representatives",
            reason="1 representante por cluster (missingness/outliers/type/iqr)",
            metrics={
                "representatives": representatives,
                "var_metrics": var_metrics,
            },
        )

        observability.record_decision(
            "collinearity_pairs_strong",
            decision="pairs_strong",
            reason="pares com |rho|>=threshold e n_pairwise>=min_pairwise_n",
            metrics={"pairs": pairs},
        )

    return CollinearityReport(
        ok=True,
        method=method,
        threshold=threshold,
        min_pairwise_n=min_pairwise_n,
        vars_in=tuple(vars_in),
        vars_temporal_heldout=tuple(sorted(vars_temporal)),
        pairs_strong=pairs,
        clusters=clusters,
        representatives=representatives,
        var_metrics=var_metrics,
        warnings=tuple(warnings),
        metadata={},
    )


def summarize_report(rep: CollinearityReport, max_clusters: int = 6) -> str:
    lines = [
        f"Collinearity OK | method={rep.method} | thr={rep.threshold} | min_pairwise_n={rep.min_pairwise_n}",
        f"Vars in clustering: {len(rep.vars_in)} | temporal heldout: {len(rep.vars_temporal_heldout)}",
        f"Strong pairs: {len(rep.pairs_strong)} | clusters: {len(rep.clusters)}",
    ]
    if rep.clusters:
        lines.append("Top clusters (sample):")
        for i, cl in enumerate(rep.clusters[:max_clusters], start=1):
            cid = f"cluster_{i}"
            r = rep.representatives.get(cid, "?")
            lines.append(f" - {cid} (n={len(cl)}), rep='{r}': {', '.join(cl[:10])}{'...' if len(cl)>10 else ''}")
    if rep.warnings:
        lines.append("Warnings:")
        lines.extend([f" - {w}" for w in rep.warnings])
    return "\n".join(lines)
