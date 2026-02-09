# -*- coding: utf-8 -*-
"""
temporal_engine.py (v2.2)

- Reusa o pipeline global (df_model + reports globais)
- Split por safra via Safra_prod (tetris por linhas)
- Filtro cruzado por processo: O identificador é Safra_{processo} == Safra_prod
- Roda insight_engine.run_intrasafra_insights por safra
- Agrega consistência temporal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonRunSummary:
    season: str
    n_rows: int
    n_vars_in: int
    n_insights: int
    top_vars: Tuple[str, ...]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalDriverConsistency:
    var: str
    direction: str 
    seasons_supported: Tuple[str, ...]
    seasons_tested: Tuple[str, ...]
    support_rate: float
    median_delta_y_kgha: float
    median_rho: float
    median_driver_score: float
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalBundle:
    ok: bool
    scope: str
    yield_col: str
    culture: str
    season_col: str
    seasons: Tuple[str, ...]
    per_season: Dict[str, Any]
    season_summaries: Tuple[SeasonRunSummary, ...]
    consistency: Tuple[TemporalDriverConsistency, ...]
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Helpers
# -----------------------------

def _pick_culture_from_df(df: pd.DataFrame) -> str:
    if "Cultura_prod" not in df.columns:
        return "default"
    s = df["Cultura_prod"].dropna().astype(str)
    if len(s) == 0:
        return "default"
    mode = s.mode()
    if len(mode) > 0:
        return str(mode.iloc[0]).strip().lower()
    return str(s.iloc[0]).strip().lower()


def _infer_process_from_var(col: str) -> Optional[str]:
    """
    Inferir processo a partir do sufixo: 'X_solo' -> 'solo'.
    Regra: pega o texto após o último '_' se existir.
    """
    if "_" not in col:
        return None
    proc = col.rsplit("_", 1)[-1].strip()
    return proc or None


def _season_col_for_process(proc: str) -> str:
    return f"Safra_{proc}"


def _vars_global_from_df_model(df_model: pd.DataFrame, yield_col: str) -> List[str]:
    ref_cols = {
        "Talhao", "id_ponto", "Lat", "Long", "Cultura_prod",
        "Data_prod", "Safra_prod",
        "_season_",
    }
    return [c for c in df_model.columns if c != yield_col and c not in ref_cols]


def _filter_vars_season(
    df_season_model: pd.DataFrame,
    vars_global: List[str],
    *,
    min_non_null: int,
    unique_min_keep: int
) -> List[str]:
    kept = []
    for v in vars_global:
        if v not in df_season_model.columns:
            continue
        x = pd.to_numeric(df_season_model[v], errors="coerce")
        nn = int(x.notna().sum())
        if nn < min_non_null:
            continue
        uniq = int(x.dropna().nunique())
        if uniq < unique_min_keep:
            continue
        kept.append(v)
    return kept


def _direction_from_item(item: Any) -> str:
    rho = getattr(item, "spearman_rho", None)
    if rho is None or not np.isfinite(rho):
        return "mixed"
    return "positive" if rho > 0 else "negative"


def _normalize_season_values(s: pd.Series) -> pd.Series:
    # string compare cego: não interpreta formato ("2024-1", "2024/25", "2025"...)
    return s.astype("string").fillna("").str.strip()


# -----------------------------
# Main
# -----------------------------

def run_temporal_consistency_v2(
    df_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    io_rep: Any,
    sq_rep: Any,
    col_rep: Any,
    *,
    season_col: str = "Safra_prod",
    enable_cross_season_filter: bool = True,
    unique_min_keep_season: int = 4,
    min_non_null_season: Optional[int] = None,
    top_k_per_season: int = 10,
    min_support_rate: float = 0.50,
    observability: Optional[Any] = None,
    insight_engine_module: Optional[Any] = None,
) -> TemporalBundle:
    """
    Temporal consistency:
    - Split por Safra_prod (ou outra coluna de safra informada em season_col)
    - Por safra, roda o insight_engine intra-safra usando df_model filtrado
    - Agrega consistência por variável ao longo das safras

    Filtro cruzado (enable_cross_season_filter=True):
    - Para cada variável X, tenta inferir processo pelo sufixo: *_solo -> processo 'solo'
    - Exige Safra_solo e Safra_prod iguais para considerar aquele valor.
    - Onde não bate, seta NaN (não inventa dados).
    - Se Safra_{processo} não existir, descarta a variável daquele processo no temporal (conservador).
    """

    if season_col not in df_raw.columns:
        raise ValueError(
            f"Coluna de safra '{season_col}' não encontrada no df_raw. "
            f"Este pipeline temporal requer Safra_{{processo}} e, em especial, Safra_prod."
        )

    warnings: List[str] = []

    if observability is not None:
        observability.start_step("temporal_engine", season_col=season_col)

    yield_col = io_rep.yield_col
    culture = _pick_culture_from_df(df_raw)

    dm = df_model.copy()
    dm["_season_"] = _normalize_season_values(df_raw[season_col]).values

    seasons_all = sorted(
        [s for s in pd.Series(dm["_season_"]).dropna().astype("string").unique().tolist()
         if str(s).strip().lower() not in {"", "nan", "none"}]
    )

    if len(seasons_all) < 2:
        warnings.append("Menos de 2 safras válidas encontradas. Consistência temporal não aplicável.")
        if observability is not None:
            observability.record_decision(
                "temporal_seasons_check",
                decision="skip",
                reason="insufficient seasons",
                metrics={"seasons": seasons_all, "season_col": season_col},
            )
        return TemporalBundle(
            ok=True,
            scope="temporal",
            yield_col=yield_col,
            culture=culture,
            season_col=season_col,
            seasons=tuple(seasons_all),
            per_season={},
            season_summaries=tuple(),
            consistency=tuple(),
            warnings=tuple(warnings),
            metadata={"note": "temporal skipped"},
        )

    vars_global = _vars_global_from_df_model(dm, yield_col)
    if len(vars_global) == 0:
        raise ValueError("Nenhuma variável global elegível encontrada (vars_global vazio).")

    if enable_cross_season_filter:
        needed_season_cols = set()
        cols_for_proc = list(vars_global) + [yield_col]  
        for c in cols_for_proc:
            proc = _infer_process_from_var(c)
            if proc is None:
                continue
            needed_season_cols.add(_season_col_for_process(proc))

        missing = []
        for sc in sorted(needed_season_cols):
            if sc in df_raw.columns:
                dm[sc] = _normalize_season_values(df_raw[sc]).values
            else:
                missing.append(sc)

        if missing:
            msg = (
                "Filtro cruzado de safra ativado, mas faltam colunas: "
                + ", ".join(missing)
                + ". Variáveis desses processos podem ser descartadas no temporal."
            )
            warnings.append(msg)
            if observability is not None:
                observability.record_decision(
                    "temporal_cross_season_missing_cols",
                    decision="warning",
                    reason="missing Safra_{processo} columns",
                    metrics={"missing": missing},
                )

    if observability is not None:
        observability.record_decision(
            "temporal_setup",
            decision="pass",
            reason="temporal setup OK",
            metrics={
                "n_seasons": len(seasons_all),
                "seasons": seasons_all,
                "vars_global": len(vars_global),
                "vars_global_sample": vars_global[:12],
                "enable_cross_season_filter": bool(enable_cross_season_filter),
            },
        )

    if insight_engine_module is None:
        from agente_agro.pipelines import insight_engine as ie  
    else:
        ie = insight_engine_module

    per_season: Dict[str, Any] = {}
    season_summaries: List[SeasonRunSummary] = []
    hits: Dict[str, Dict[str, Any]] = {}

    for s in seasons_all:
        dms = dm.loc[dm["_season_"].astype("string") == str(s)].copy()
        n_rows = int(dms.shape[0])

        min_nn = int(min_non_null_season) if min_non_null_season is not None else max(10, int(round(0.30 * n_rows)))

        vars_season = _filter_vars_season(
            dms,
            vars_global,
            min_non_null=min_nn,
            unique_min_keep=unique_min_keep_season
        )

        notes = []
        if len(vars_season) == 0:
            notes.append("Nenhuma variável com dados/variabilidade suficiente nesta safra; safra ignorada.")
            season_summaries.append(
                SeasonRunSummary(season=str(s), n_rows=n_rows, n_vars_in=0, n_insights=0, top_vars=tuple(), notes=tuple(notes))
            )
            continue

        if enable_cross_season_filter:
            vars_f = []
            dropped_by_cross = []

            for v in vars_season:
                proc = _infer_process_from_var(v)
                if proc is None:
                    vars_f.append(v)
                    continue

                sc = _season_col_for_process(proc)
                if sc not in dms.columns:
                    dropped_by_cross.append((v, sc, "missing_season_col"))
                    continue

                ok = (dms[sc].astype("string") == dms["_season_"].astype("string"))
                if int(ok.sum()) == 0:
                    dropped_by_cross.append((v, sc, "no_rows_match"))
                    continue

                dms.loc[~ok, v] = np.nan
                vars_f.append(v)

            vars_season = vars_f

            if dropped_by_cross and observability is not None:
                observability.record_decision(
                    "temporal_cross_season_filter",
                    decision="pass",
                    reason="aplicado filtro Safra_{processo} == Safra_prod",
                    metrics={
                        "safra": str(s),
                        "dropped_n": len(dropped_by_cross),
                        "dropped_examples": dropped_by_cross[:10],
                    },
                )

            if len(vars_season) == 0:
                notes.append("Variáveis descartadas pelo filtro cruzado de safra; safra ignorada.")
                season_summaries.append(
                    SeasonRunSummary(season=str(s), n_rows=n_rows, n_vars_in=0, n_insights=0, top_vars=tuple(), notes=tuple(notes))
                )
                continue

        df_model_season = dms[[yield_col] + vars_season].copy()

        if observability is not None:
            observability.start_step("temporal_season_run", safra=str(s), n_rows=n_rows, n_vars=len(vars_season))

        bundle_s = ie.run_intrasafra_insights(
            df_model_season, io_rep, sq_rep, col_rep,
            observability=observability,
        )
        per_season[str(s)] = bundle_s

        insights = list(getattr(bundle_s, "insights", []) or [])
        insights_sorted = sorted(insights, key=lambda it: float(getattr(it, "driver_score", 0.0)), reverse=True)
        top_vars = tuple([getattr(it, "var", "?") for it in insights_sorted[:top_k_per_season]])

        season_summaries.append(
            SeasonRunSummary(
                season=str(s),
                n_rows=n_rows,
                n_vars_in=len(vars_season),
                n_insights=len(insights_sorted),
                top_vars=top_vars,
                notes=tuple(notes),
            )
        )

        for it in insights_sorted:
            v = getattr(it, "var", None)
            if not v:
                continue
            hits.setdefault(v, {})
            hits[v][str(s)] = it

    seasons_tested = sorted(list(per_season.keys()))
    if len(seasons_tested) < 2:
        warnings.append("Menos de 2 safras com bundles válidos (após filtros). Temporal pouco informativo.")

    consistency: List[TemporalDriverConsistency] = []
    for var, by_season in hits.items():
        ss_supported = sorted(list(by_season.keys()))
        n_tested = max(1, len(seasons_tested))
        support_rate = len(ss_supported) / n_tested
        if support_rate < min_support_rate:
            continue

        dirs = [_direction_from_item(by_season[ss]) for ss in ss_supported]
        direction = dirs[0] if len(set(dirs)) == 1 else "mixed"

        deltas, rhos, scores = [], [], []
        for ss in ss_supported:
            it = by_season[ss]
            dy = getattr(it, "delta_yield_q4_q1", None)  
            if dy is not None and np.isfinite(float(dy)):
                deltas.append(float(dy))
            rho = getattr(it, "spearman_rho", None)
            if rho is not None and np.isfinite(float(rho)):
                rhos.append(float(rho))
            sc = getattr(it, "driver_score", None)
            if sc is not None and np.isfinite(float(sc)):
                scores.append(float(sc))

        med_dy = float(np.median(deltas)) if deltas else float("nan")
        med_rho = float(np.median(rhos)) if rhos else float("nan")
        med_sc = float(np.median(scores)) if scores else float("nan")

        notes2 = []
        if direction == "mixed":
            notes2.append("Sinal (rho) varia entre safras; interpretar com cautela.")

        consistency.append(
            TemporalDriverConsistency(
                var=var,
                direction=direction,
                seasons_supported=tuple(ss_supported),
                seasons_tested=tuple(seasons_tested),
                support_rate=float(support_rate),
                median_delta_y_kgha=med_dy,
                median_rho=med_rho,
                median_driver_score=med_sc,
                notes=tuple(notes2),
            )
        )

    consistency.sort(key=lambda x: (-x.support_rate, -np.nan_to_num(x.median_driver_score, nan=-1e9), x.var))

    if observability is not None:
        observability.record_decision(
            "temporal_consistency_summary",
            decision="pass",
            reason="consistência temporal agregada",
            metrics={
                "seasons_tested": seasons_tested,
                "n_vars_with_support": len(consistency),
                "min_support_rate": min_support_rate,
                "top_vars": [c.var for c in consistency[:10]],
            },
        )

    return TemporalBundle(
        ok=True,
        scope="temporal",
        yield_col=yield_col,
        culture=culture,
        season_col=season_col,
        seasons=tuple(seasons_tested),
        per_season=per_season,
        season_summaries=tuple(season_summaries),
        consistency=tuple(consistency),
        warnings=tuple(warnings),
        metadata={
            "vars_global": vars_global,
            "unique_min_keep_season": unique_min_keep_season,
            "min_non_null_season": min_non_null_season,
            "top_k_per_season": top_k_per_season,
            "min_support_rate": min_support_rate,
            "enable_cross_season_filter": enable_cross_season_filter,
        },
    )


def summarize_temporal_bundle(tb: TemporalBundle, top_n: int = 10) -> str:
    lines = [
        f"Temporal engine OK | culture={tb.culture} | safras={', '.join(tb.seasons) if tb.seasons else '(none)'}",
        f"Vars with temporal support: {len(tb.consistency)} (min_support_rate={tb.metadata.get('min_support_rate')})",
    ]
    if tb.warnings:
        lines.append("Warnings:")
        lines.extend([f" - {w}" for w in tb.warnings])

    if tb.consistency:
        lines.append(f"Top {min(top_n, len(tb.consistency))} consistent drivers:")
        for c in tb.consistency[:top_n]:
            dy = c.median_delta_y_kgha
            dy_s = "NA" if not np.isfinite(dy) else f"{dy:+.0f}"
            lines.append(
                f" - {c.var} | dir={c.direction} | support={len(c.seasons_supported)}/{len(c.seasons_tested)} "
                f"({c.support_rate:.0%}) | median Δyield≈{dy_s} kg/ha | median rho={c.median_rho:+.2f}"
            )

    return "\n".join(lines)
