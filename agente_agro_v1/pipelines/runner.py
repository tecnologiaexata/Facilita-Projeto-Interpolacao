# -*- coding: utf-8 -*-
"""
runner.py

Orquestra o pipeline do Agente Agronômico sem inputs do usuário.

- run_individual_seasons: roda o pipeline intra-safra para CADA Safra_prod disponível
- run_temporal: roda o pipeline temporal (consistência entre safras), quando possível
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class SeasonRunResult:
    season: str
    ok: bool
    culture: str
    n_rows: int
    report_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RunnerResult:
    ok: bool
    source_csv: str
    culture: str
    seasons_found: Tuple[str, ...]
    individual_runs: Tuple[SeasonRunResult, ...]
    temporal_ok: bool
    temporal_report_text: str
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------
# Helpers
# ---------

def _safe_str(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.strip()


def _pick_culture_from_df(df: pd.DataFrame) -> str:
    if "Cultura_prod" not in df.columns:
        return "default"
    s = df["Cultura_prod"].dropna().astype(str).str.strip()
    if len(s) == 0:
        return "default"
    mode = s.mode()
    if len(mode) > 0:
        return str(mode.iloc[0]).strip().lower()
    return str(s.iloc[0]).strip().lower()


def _list_seasons(df: pd.DataFrame, season_col: str = "Safra_prod") -> List[str]:
    if season_col not in df.columns:
        return []
    vals = (
        df[season_col]
        .dropna()
        .astype(str)
        .map(_safe_str)
        .tolist()
    )
    vals = [v for v in vals if v and v.lower() not in {"nan", "none"}]
    return sorted(list(dict.fromkeys(vals)))  


def _temp_csv_for_season(
    df: pd.DataFrame,
    *,
    base_dir: Path,
    season: str,
    prefix: str = "grid_tmp",
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in str(season)])
    out = base_dir / f"{prefix}__{safe}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    return out


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_tmp_dir() -> Path:
    return _get_project_root() / "outputs" / "_tmp_runner"


def _get_tb_seasons(tb: Any) -> Tuple[str, ...]:
    seas = getattr(tb, "seasons", None)
    if seas is None:
        seas = getattr(tb, "years", None)
    if seas is None:
        return tuple()
    try:
        return tuple([str(x) for x in seas])
    except Exception:
        return tuple()


# ------------
# Runners
# ------------

def run_individual_seasons(
    csv_path: str | Path,
    *,
    season_col: str = "Safra_prod",
    enable_cross_season_filter: bool = True,
    observability: Optional[Any] = None,
    top_k: int = 5,
) -> Tuple[List[SeasonRunResult], List[str]]:
    """
    Roda o pipeline intra-safra para todas as safras em Safra_prod.

    enable_cross_season_filter:
      - Se True, aplica filtro conservador: qualquer coluna Safra_* preenchida deve bater com a safra alvo,
        ou a linha é descartada (evita "misturar processo de outra safra" por acidente).
    """
    from agente_agro.pipelines.io_checks import load_and_check_grid, summarize_report
    from agente_agro.pipelines import safety_quality, collinearity, insight_engine, recommendation_engine, reporting

    csv_path = Path(csv_path)
    df_raw = pd.read_csv(csv_path)

    seasons = _list_seasons(df_raw, season_col=season_col)
    warnings: List[str] = []
    results: List[SeasonRunResult] = []

    if len(seasons) == 0:
        warnings.append(f"Nenhuma safra encontrada em '{season_col}'. Runner individual não executado.")
        return results, warnings

    tmp_dir = _get_tmp_dir()

    for season in seasons:
        if observability is not None:
            observability.start_step("runner_individual_season", season=season)

        d = df_raw.loc[df_raw[season_col].astype(str).map(_safe_str) == season].copy()

        # filtro cruzado por Safra_{processo} 
        if enable_cross_season_filter:
            safra_cols = [c for c in d.columns if c.startswith("Safra_")]
            if safra_cols:
                mask = pd.Series(True, index=d.index)
                for c in safra_cols:
                    s = d[c].astype("string")
                    mask = mask & (s.isna() | (s.astype(str).map(_safe_str) == season))
                before = int(d.shape[0])
                d = d.loc[mask].copy()
                after = int(d.shape[0])
                if observability is not None:
                    observability.record_decision(
                        "runner_cross_season_filter",
                        decision="applied" if before != after else "noop",
                        reason="filtro cruzado Safra_* por safra alvo",
                        metrics={"season": season, "rows_before": before, "rows_after": after, "n_safra_cols": len(safra_cols)},
                    )

        if int(d.shape[0]) == 0:
            results.append(
                SeasonRunResult(
                    season=season,
                    ok=False,
                    culture=_pick_culture_from_df(df_raw),
                    n_rows=0,
                    report_text="(sem dados após filtros)",
                    warnings=(f"Safra '{season}' sem linhas após filtros.",),
                )
            )
            continue

        tmp_csv = _temp_csv_for_season(d, base_dir=tmp_dir, season=season, prefix="grid_season")

        try:
            # 1) io checks
            df_season, io_rep = load_and_check_grid(tmp_csv, observability=observability)
            if observability is not None:
                observability.record_decision(
                    "runner_io_checks_summary",
                    decision="pass",
                    reason="io_checks OK para safra",
                    metrics={"season": season, "io_summary": summarize_report(io_rep)},
                )

            # 2) safety
            df_model, sq_rep = safety_quality.apply_safety_quality(df_season, io_rep, observability=observability)

            # 3) collinearity
            col_rep = collinearity.run_collinearity(df_model, sq_rep, observability=observability)

            # 4) insights
            bundle = insight_engine.run_intrasafra_insights(
                df_model, io_rep, sq_rep, col_rep, observability=observability
            )

            # 5) recomendações
            rec_bundle = recommendation_engine.build_recommendations(
                bundle,
                df_model=df_model,
                df_context=df_season,
                top_k=top_k,
                observability=observability,
            )

            # 6) report 
            render_full = getattr(reporting, "render_full_report", None)
            if callable(render_full):
                report_text = render_full(rec_bundle, include_evidence=True)
            else:
                report_text = (
                    "Relatório intra-safra (fallback)\n"
                    f"Cultura: {getattr(rec_bundle, 'culture', _pick_culture_from_df(df_season))}\n"
                    f"Recomendações: {len(getattr(rec_bundle, 'recommendations', ()) or [])}\n"
                    f"Warnings: {len(getattr(rec_bundle, 'warnings', ()) or [])}"
                )
                warnings.append("reporting.render_full_report não encontrado; usando fallback.")

            results.append(
                SeasonRunResult(
                    season=season,
                    ok=True,
                    culture=getattr(rec_bundle, "culture", None) or _pick_culture_from_df(df_season),
                    n_rows=int(df_season.shape[0]),
                    report_text=report_text,
                    warnings=tuple(getattr(rec_bundle, "warnings", ()) or ()),
                    metadata={
                        "tmp_csv": str(tmp_csv),
                        "enable_cross_season_filter": enable_cross_season_filter,
                    },
                )
            )

            if observability is not None:
                observability.export_audit(step_name=f"runner_individual__{season}")

        except Exception as e:
            results.append(
                SeasonRunResult(
                    season=season,
                    ok=False,
                    culture=_pick_culture_from_df(df_raw),
                    n_rows=int(d.shape[0]),
                    report_text=f"(erro) {type(e).__name__}: {e}",
                    warnings=(f"Falha ao rodar safra '{season}': {type(e).__name__}",),
                    metadata={"tmp_csv": str(tmp_csv)},
                )
            )
            if observability is not None:
                observability.record_decision(
                    "runner_individual_error",
                    decision="fail",
                    reason=str(e),
                    metrics={"season": season, "exc_type": type(e).__name__},
                )
                observability.export_audit(step_name=f"runner_individual__{season}__error")

    return results, warnings


def run_temporal(
    csv_path: str | Path,
    *,
    season_col: str = "Safra_prod",
    enable_cross_season_filter: bool = True,
    top_k: int = 5,
    min_support_rate: float = 0.50,
    observability: Optional[Any] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Roda pipeline temporal:
    io_checks -> safety_quality -> collinearity -> temporal_engine -> rec temporal -> report temporal

    Retorna:
      (temporal_ok, temporal_report_text, warnings)
    """
    from agente_agro.pipelines.io_checks import load_and_check_grid
    from agente_agro.pipelines import safety_quality, collinearity, temporal_engine, recommendation_engine, reporting

    csv_path = Path(csv_path)
    warnings: List[str] = []

    if observability is not None:
        observability.start_step("runner_temporal", file=str(csv_path))

    # 1) io checks
    df_raw, io_rep = load_and_check_grid(csv_path, observability=observability)

    # 2) safety e 3) collinearity (global, sem split)
    df_model, sq_rep = safety_quality.apply_safety_quality(df_raw, io_rep, observability=observability)
    col_rep = collinearity.run_collinearity(df_model, sq_rep, observability=observability)

    # 4) temporal_engine (split interno por Safra_prod e filtro cruzado)
    tb = temporal_engine.run_temporal_consistency_v2(
        df_raw=df_raw,
        df_model=df_model,
        io_rep=io_rep,
        sq_rep=sq_rep,
        col_rep=col_rep,
        season_col=season_col,
        enable_cross_season_filter=enable_cross_season_filter,
        min_support_rate=min_support_rate,
        observability=observability,
    )

    tb_seasons = _get_tb_seasons(tb)

    if len(tb_seasons) < 2:
        msg = getattr(tb, "warnings", None) or ("Consistência temporal não aplicável.",)
        warnings.extend(list(msg))
        report_text = (
            f"Temporal (skip) | cultura={getattr(tb, 'culture', 'default')} | "
            f"safras={', '.join(tb_seasons)}\n"
            f"Motivo: {' | '.join(msg)}"
        )
        if observability is not None:
            observability.export_audit(step_name="runner_temporal__skip")
        return True, report_text, warnings

    # 5) recomendações temporais
    rec_t = recommendation_engine.build_temporal_recommendations(
        tb,
        df_context=df_raw,
        top_k=top_k,
        min_support_rate=min_support_rate,
        observability=observability,
    )

    # 6) report temporal 
    render_temporal = getattr(reporting, "render_temporal_full_report", None)
    if callable(render_temporal):
        report_text = render_temporal(rec_t, include_evidence=True)
    else:
        report_text = (
            "Relatório temporal (fallback)\n"
            f"Cultura: {getattr(tb, 'culture', 'default')}\n"
            f"Safras: {', '.join(tb_seasons)}\n"
            f"Recomendações: {len(getattr(rec_t, 'recommendations', ()) or [])}"
        )
        warnings.append("reporting.render_temporal_full_report não encontrado; usando fallback.")

    if observability is not None:
        observability.export_audit(step_name="runner_temporal")

    return True, report_text, warnings


def run_pipeline(
    csv_path: str | Path,
    *,
    season_col: str = "Safra_prod",
    enable_temporal: bool = True,
    enable_individual: bool = True,
    enable_cross_season_filter: bool = True,
    top_k: int = 5,
    min_support_rate: float = 0.50,
    observability: Optional[Any] = None,
) -> RunnerResult:
    """
    Runner principal: executa (opcionalmente) individual + temporal e devolve um objeto consolidado.
    """
    csv_path = Path(csv_path)
    df_raw = pd.read_csv(csv_path)
    culture = _pick_culture_from_df(df_raw)
    seasons_found = tuple(_list_seasons(df_raw, season_col=season_col))

    warnings: List[str] = []

    individual_runs: List[SeasonRunResult] = []
    if enable_individual:
        runs, w = run_individual_seasons(
            csv_path,
            season_col=season_col,
            enable_cross_season_filter=enable_cross_season_filter,
            observability=observability,
            top_k=top_k,
        )
        individual_runs = runs
        warnings.extend(w)

    temporal_ok = False
    temporal_report_text = ""
    if enable_temporal:
        temporal_ok, temporal_report_text, w2 = run_temporal(
            csv_path,
            season_col=season_col,
            enable_cross_season_filter=enable_cross_season_filter,
            top_k=top_k,
            min_support_rate=min_support_rate,
            observability=observability,
        )
        warnings.extend(w2)

    ok = True
    if enable_individual and all(not r.ok for r in individual_runs) and len(individual_runs) > 0:
        ok = False

    return RunnerResult(
        ok=ok,
        source_csv=str(csv_path),
        culture=culture,
        seasons_found=seasons_found,
        individual_runs=tuple(individual_runs),
        temporal_ok=bool(temporal_ok),
        temporal_report_text=str(temporal_report_text),
        warnings=tuple(warnings),
        metadata={
            "season_col": season_col,
            "enable_cross_season_filter": enable_cross_season_filter,
            "top_k": top_k,
            "min_support_rate": min_support_rate,
        },
    )
