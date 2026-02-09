# -*- coding: utf-8 -*-
"""
recommendation_engine.py (v1)

Responsabilidade:
- Converter InsightBundle em recomendações (top-K), usando regras conservadoras de gating.
- Não prescrever manejo. Apenas "atenção / investigar / possível limitante / possível alavanca".
- Rastreável: cada recomendação carrega evidência + regras aplicadas + ressalvas.

Regras (hard-coded, podem ser editadas):
- TOP_K recomendações por rodada
- min_n_valid
- magnitude mínima em kg/ha por cultura (via coluna 'Cultura_prod' do grid_completo)
- concordância de sinais (pelo menos 2 de 3): Δyield, Spearman, AUC
- Cliff mínimo canônico
- CV baixo vira caveat (não rejeita)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------
# Hard-coded knobs 
# ------------------

TOP_K = 5
MIN_VALID_N = 30

MIN_ABS_CLIFF = 0.25

AUC_POS_MIN = 0.55   # positivo: espera AUC acima disso
AUC_NEG_MAX = 0.45   # negativo: espera AUC abaixo disso

CV_WARN_THRESHOLD_PCT = 5.0

# Tenta ler df['Cultura_prod'] para selecionar o threshold.
MIN_YIELD_DELTA_BY_CULTURE_KGHA: Dict[str, float] = {
    "default": 300.0,
    "soja": 300.0,          # 5 sacas ~ 300 kg/ha
    # exemplos
    # "milho": 600.0,
    # "cana": 5000.0,
}


@dataclass(frozen=True)
class RecommendationItem:
    var: str
    direction: str             
    priority: str             
    title: str
    claim: str
    action_hint: str
    evidence: Dict[str, Any]
    caveats: Tuple[str, ...]
    rules_fired: Tuple[str, ...]
    driver_score: float


@dataclass(frozen=True)
class RecommendationBundle:
    ok: bool
    scope: str
    culture: str
    top_k: int
    recommendations: Tuple[RecommendationItem, ...]
    rejected_count: int
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]


# ----------
# Utilities
# ----------

def _normalize_culture(c: str) -> str:
    if c is None:
        return "default"
    s = str(c).strip().lower()
    s = s.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("à", "a").replace("â", "a")
    s = s.replace("é", "e").replace("ê", "e").replace("í", "i").replace("ó", "o").replace("ô", "o").replace("õ", "o")
    s = s.replace("ú", "u")
    return s or "default"


def _pick_culture_from_df(df: pd.DataFrame) -> str:
    if "Cultura_prod" not in df.columns:
        return "default"
    s = df["Cultura_prod"].dropna().astype(str)
    if len(s) == 0:
        return "default"
    mode = s.mode()
    if len(mode) > 0:
        return _normalize_culture(mode.iloc[0])
    return _normalize_culture(s.iloc[0])


def _sign(x: Optional[float]) -> Optional[int]:
    if x is None:
        return None
    if not np.isfinite(x):
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _auc_direction_ok(auc: Optional[float], direction: int) -> Optional[bool]:
    """
    direction: +1 (positivo), -1 (negativo)
    Retorna None se AUC não disponível.
    """
    if auc is None or not np.isfinite(auc):
        return None
    if direction > 0:
        return auc >= AUC_POS_MIN
    if direction < 0:
        return auc <= AUC_NEG_MAX
    return None


def _direction_label(direction: int) -> str:
    return "positive" if direction > 0 else "negative"


def _priority_from_strength(abs_cliff: Optional[float], abs_delta_y: Optional[float], signal_votes: int) -> str:
    """
    Heurística conservadora:
    - high: votos >=2 e (|cliff| alto ou |Δy| alto)
    - medium: votos >=2 e força moderada
    - low: votos ==2 mas força mínima
    """
    c = abs_cliff or 0.0
    d = abs_delta_y or 0.0
    if signal_votes >= 2 and (c >= 0.35 or d >= 500):
        return "high"
    if signal_votes >= 2 and (c >= MIN_ABS_CLIFF or d >= 0):
        return "medium"
    return "low"

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x is None:
            continue
        s = str(x)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# -----------------------------
# Core gating + generation
# -----------------------------

def build_recommendations(
    insight_bundle: Any,
    df_model: pd.DataFrame,
    *,
    df_context=None,
    top_k: int = TOP_K,
    observability: Optional[Any] = None,
) -> RecommendationBundle:
    """
    Consome InsightBundle (do insight_engine) e devolve top-K recomendações com regras conservadoras.

    Regras principais:
      - n_valid >= MIN_VALID_N
      - direção definida por Δyield(T3-T1) ou fallback de correlação
      - pelo menos 2 de 3 sinais concordam: Δyield, Spearman, AUC (quando disponíveis)
      - magnitude mínima: |Δyield| >= threshold_cultura OU |Cliff| >= MIN_ABS_CLIFF
      - CV < 5% -> caveat (não rejeita)
    """

    if observability is not None:
        observability.start_step("recommendation_engine", top_k=top_k)

    culture = _pick_culture_from_df(df_context if df_context is not None else df_model)
    min_delta_y = float(MIN_YIELD_DELTA_BY_CULTURE_KGHA.get(culture, MIN_YIELD_DELTA_BY_CULTURE_KGHA["default"]))

    recs: List[RecommendationItem] = []
    rejected = 0
    bundle_warnings: List[str] = []

    insights = list(getattr(insight_bundle, "insights", []) or [])
    if len(insights) == 0:
        bundle_warnings.append("Nenhum insight disponível para recomendação.")
        if observability is not None:
            observability.record_decision(
                "recommendation_engine_summary",
                decision="fail",
                reason="sem insights",
                metrics={"top_k": top_k, "culture": culture},
            )
        return RecommendationBundle(
            ok=False,
            scope=getattr(insight_bundle, "scope", "unknown"),
            culture=culture,
            top_k=top_k,
            recommendations=tuple(),
            rejected_count=0,
            warnings=tuple(bundle_warnings),
            metadata={"min_delta_y_kgha": min_delta_y, "min_abs_cliff": MIN_ABS_CLIFF},
        )

    for it in insights:
        rules_fired: List[str] = []
        caveats: List[str] = list(getattr(it, "warnings", []) or [])

        # regra 1: n válido
        n_valid = int(getattr(it, "n_valid", 0) or 0)
        if n_valid < MIN_VALID_N:
            rejected += 1
            continue

        dy = getattr(it, "delta_yield_q4_q1", None) 
        rho = getattr(it, "spearman_rho", None)
        auc = getattr(it, "auc", None)
        cliffs = getattr(it, "cliffs_delta", None)
        details = getattr(it, "details", {}) or {}

        dy_sign = _sign(dy)
        rho_sign = _sign(rho)

        if dy_sign is None and rho_sign is None:
            rejected += 1
            continue

        direction = dy_sign if dy_sign is not None and dy_sign != 0 else rho_sign
        if direction is None or direction == 0:
            rejected += 1
            continue

        votes = 0
        available = 0

        if dy_sign is not None and dy_sign != 0:
            available += 1
            if dy_sign == direction:
                votes += 1

        if rho_sign is not None and rho_sign != 0:
            available += 1
            if rho_sign == direction:
                votes += 1

        auc_ok = _auc_direction_ok(auc, direction)
        if auc_ok is not None:
            available += 1
            if auc_ok:
                votes += 1

        # se só tiver 1 sinal disponível, rejeita (conservador)
        if available < 2 or votes < 2:
            rejected += 1
            continue
        rules_fired.append(f"signal_consensus_{votes}_of_{available}")

        # regra 3: magnitude mínima (cultura OU cliff)
        abs_dy = abs(float(dy)) if dy is not None and np.isfinite(dy) else None
        abs_cliff = abs(float(cliffs)) if cliffs is not None and np.isfinite(cliffs) else None

        passes_magnitude = False
        if abs_dy is not None and abs_dy >= min_delta_y:
            passes_magnitude = True
            rules_fired.append(f"min_delta_y_pass({abs_dy:.0f}>={min_delta_y:.0f})")
        if abs_cliff is not None and abs_cliff >= MIN_ABS_CLIFF:
            passes_magnitude = True
            rules_fired.append(f"min_cliff_pass({abs_cliff:.2f}>={MIN_ABS_CLIFF:.2f})")

        if not passes_magnitude:
            rejected += 1
            continue

        # regra 4: CV baixo -> caveat
        x_cv = details.get("x_cv_pct", None)
        if x_cv is not None and np.isfinite(x_cv) and float(x_cv) < CV_WARN_THRESHOLD_PCT:
            caveats.append(
                f"Baixa variabilidade de '{it.var}' na lavoura (CV={float(x_cv):.2f}%). "
                "O impacto operacional pode ser limitado; interpretar com cautela."
            )
            rules_fired.append("cv_low_caveat")

        seen = set()
        caveats = [c for c in caveats if not (c in seen or seen.add(c))]
        

        dir_label = _direction_label(direction)
        var = it.var

        if dir_label == "positive":
            title = f"Potencial alavanca: {var}"
            claim = f"Valores mais altos de '{var}' tendem a ocorrer em áreas de maior produtividade."
            action_hint = "Investigar se a variável indica condição favorável replicável ou manejável."
        else:
            title = f"Possível limitante: {var}"
            claim = f"Valores mais altos de '{var}' tendem a ocorrer em áreas de menor produtividade."
            action_hint = "Investigar se a variável pode estar associada a restrição/ambiente desfavorável."


        evidence = {
            "yield_col": getattr(insight_bundle, "yield_col", "yield"),
            "direction": dir_label,
            "delta_yield_T3_T1_kgha": dy,
            "delta_yield_pct": getattr(it, "delta_yield_q4_q1_pct", None),
            "spearman_rho": rho,
            "cliffs_delta": cliffs,
            "auc": auc,
            "mannwhitney_p": getattr(it, "mannwhitney_p", None),
            "enrich_top_yield_in_top_x": getattr(it, "enrichment_top_yield_in_top_x", None),
            "n_valid": n_valid,
            "outlier_frac": getattr(it, "outlier_frac", None),
            "x_variability": {
                "x_median": details.get("x_median"),
                "x_iqr": details.get("x_iqr"),
                "x_q10": details.get("x_q10"),
                "x_q90": details.get("x_q90"),
                "x_cv_pct": details.get("x_cv_pct"),
            },
            "cluster_context": {
                "cluster_id": getattr(it, "cluster_id", None),
                "cluster_vars": list(getattr(it, "cluster_vars", (var,))),
                "var_type": getattr(it, "var_type", "unknown"),
            },
            "thresholds_used": {
                "min_delta_y_kgha": min_delta_y,
                "min_abs_cliff": MIN_ABS_CLIFF,
                "auc_pos_min": AUC_POS_MIN,
                "auc_neg_max": AUC_NEG_MAX,
            },
        }

        priority = _priority_from_strength(abs_cliff, abs_dy, votes)

        recs.append(
            RecommendationItem(
                var=var,
                direction=dir_label,
                priority=priority,
                title=title,
                claim=claim,
                action_hint=action_hint,
                evidence=evidence,
                caveats=tuple(caveats),
                rules_fired=tuple(rules_fired),
                driver_score=float(getattr(it, "driver_score", 0.0) or 0.0),
            )
        )


    recs_sorted = sorted(recs, key=lambda r: ({"high": 2, "medium": 1, "low": 0}[r.priority], r.driver_score), reverse=True)
    recs_top = recs_sorted[:top_k]

    if len(recs_top) == 0:
        bundle_warnings.append("Nenhuma recomendação passou no gating (critérios conservadores).")

    if observability is not None:
        observability.record_decision(
            "recommendation_engine_summary",
            decision="pass" if len(recs_top) else "warning",
            reason="recomendações geradas a partir de insights intra-safra",
            metrics={
                "culture": culture,
                "top_k": top_k,
                "min_delta_y_kgha": min_delta_y,
                "min_abs_cliff": MIN_ABS_CLIFF,
                "n_insights_in": len(insights),
                "n_recommendations_out": len(recs_top),
                "rejected_count": rejected,
                "top_recommendations": [
                    {"var": r.var, "direction": r.direction, "priority": r.priority, "driver_score": r.driver_score}
                    for r in recs_top
                ],
                "bundle_warnings": bundle_warnings,
            },
        )

        observability.record_decision(
            "recommendation_engine_full",
            decision="recommendations",
            reason="lista completa (top_k) com evidências e caveats",
            metrics={
                "recommendations": [
                    {
                        "var": r.var,
                        "direction": r.direction,
                        "priority": r.priority,
                        "title": r.title,
                        "claim": r.claim,
                        "action_hint": r.action_hint,
                        "caveats": list(r.caveats),
                        "rules_fired": list(r.rules_fired),
                        "driver_score": r.driver_score,
                        "evidence": r.evidence,
                    }
                    for r in recs_top
                ]
            },
        )

    return RecommendationBundle(
        ok=True,
        scope=getattr(insight_bundle, "scope", "unknown"),
        culture=culture,
        top_k=top_k,
        recommendations=tuple(recs_top),
        rejected_count=rejected,
        warnings=tuple(bundle_warnings),
        metadata={
            "min_delta_y_kgha": min_delta_y,
            "min_abs_cliff": MIN_ABS_CLIFF,
            "note": "recommendations are observational; no causal claims",
        },
    )


def summarize_recommendations(bundle: RecommendationBundle) -> str:
    lines = [
        f"Recommendation engine OK | scope={bundle.scope} | culture={bundle.culture} | top_k={bundle.top_k}",
        f"Recommendations: {len(bundle.recommendations)} | rejected={bundle.rejected_count}",
    ]
    if bundle.warnings:
        lines.append("Bundle warnings:")
        lines.extend([f" - {w}" for w in bundle.warnings])

    for r in bundle.recommendations:
        dy = r.evidence.get("delta_yield_T3_T1_kgha", None)
        rho = r.evidence.get("spearman_rho", None)
        enr = r.evidence.get("enrich_top_yield_in_top_x", None)
        dy_s = "NA" if dy is None else f"{dy:+.0f} kg/ha"
        rho_s = "NA" if rho is None else f"{rho:+.2f}"
        enr_s = "NA" if enr is None else f"{enr:.2f}x"
        lines.append(f"- [{r.priority.upper()}] {r.title} | dir={r.direction} | Δy={dy_s} | rho={rho_s} | enrich={enr_s}")
        lines.append(f"  {r.claim}")
        if r.caveats:
            lines.append(f"  Caveats: " + " | ".join(r.caveats))
    return "\n".join(lines)


# -----------------------------
# Temporal recommendations (v1)
# -----------------------------

def _collect_insight_items_by_var_from_temporal(tb: Any, var: str) -> List[Tuple[str, Any]]:
    """
    Busca, dentro do TemporalBundle, os InsightItems por safra para uma variável.
    Retorna lista de tuplas (safra, InsightItem).
    """
    out: List[Tuple[str, Any]] = []
    per = getattr(tb, "per_season", {}) or {}
    for season, bundle in per.items():
        ins = list(getattr(bundle, "insights", []) or [])
        for it in ins:
            if getattr(it, "var", None) == var:
                out.append((str(season), it))
                break
    return out


def _median_finite(xs: List[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    if not vals:
        return None
    return float(np.median(vals))


def build_temporal_recommendations(
    temporal_bundle: Any,
    df_context: pd.DataFrame,
    *,
    top_k: int = TOP_K,
    min_support_rate: float = 0.50,
    min_supported_seasons: int = 2,
    observability: Optional[Any] = None,
) -> RecommendationBundle:
    """
    Converte TemporalBundle -> RecommendationBundle (scope='temporal').
    """

    if observability is not None:
        observability.start_step("recommendation_engine_temporal", top_k=top_k, min_support_rate=min_support_rate)

    culture = _pick_culture_from_df(df_context)
    min_delta_y = float(MIN_YIELD_DELTA_BY_CULTURE_KGHA.get(culture, MIN_YIELD_DELTA_BY_CULTURE_KGHA["default"]))

    consistency = list(getattr(temporal_bundle, "consistency", []) or [])
    seasons_tested = list(getattr(temporal_bundle, "seasons", []) or [])
    warnings: List[str] = list(getattr(temporal_bundle, "warnings", []) or [])
    rejected = 0

    if len(seasons_tested) < min_supported_seasons:
        warnings.append("Menos de 2 safras válidas para recomendações temporais.")
        if observability is not None:
            observability.record_decision(
                "recommendation_engine_temporal_summary",
                decision="skip",
                reason="insufficient seasons",
                metrics={"seasons_tested": seasons_tested, "min_supported_seasons": min_supported_seasons},
            )
        return RecommendationBundle(
            ok=True,
            scope="temporal",
            culture=culture,
            top_k=top_k,
            recommendations=tuple(),
            rejected_count=0,
            warnings=tuple(warnings),
            metadata={"note": "temporal recommendations skipped", "min_delta_y_kgha": min_delta_y},
        )

    recs: List[RecommendationItem] = []

    for c in consistency:
        var = getattr(c, "var", None)
        if not var:
            rejected += 1
            continue

        support_rate = float(getattr(c, "support_rate", 0.0) or 0.0)
        supported = list(getattr(c, "seasons_supported", []) or [])
        tested = list(getattr(c, "seasons_tested", []) or seasons_tested)

        if support_rate < float(min_support_rate):
            rejected += 1
            continue
        if len(supported) < int(min_supported_seasons):
            rejected += 1
            continue

        direction = str(getattr(c, "direction", "mixed") or "mixed").strip().lower()
        med_dy = getattr(c, "median_delta_y_kgha", None)
        med_rho = getattr(c, "median_rho", None)
        med_score = getattr(c, "median_driver_score", None)

        items = _collect_insight_items_by_var_from_temporal(temporal_bundle, var) 

        cliffs = _median_finite([getattr(it, "cliffs_delta", None) for (season, it) in items])
        aucs = _median_finite([getattr(it, "auc", None) for (season, it) in items])

        abs_dy = abs(float(med_dy)) if med_dy is not None and np.isfinite(float(med_dy)) else None
        abs_cliff = abs(float(cliffs)) if cliffs is not None and np.isfinite(float(cliffs)) else None

        passes_magnitude = False
        rules_fired: List[str] = []

        if abs_dy is not None and abs_dy >= min_delta_y:
            passes_magnitude = True
            rules_fired.append(f"min_median_delta_y_pass({abs_dy:.0f}>={min_delta_y:.0f})")
        if abs_cliff is not None and abs_cliff >= MIN_ABS_CLIFF:
            passes_magnitude = True
            rules_fired.append(f"min_median_cliff_pass({abs_cliff:.2f}>={MIN_ABS_CLIFF:.2f})")

        if not passes_magnitude:
            rejected += 1
            continue

        caveats: List[str] = []
        c_notes = list(getattr(c, "notes", []) or [])
        caveats.extend(c_notes)

        for season, it in items:
            for w in list(getattr(it, "warnings", []) or []):
                caveats.append(f"[{season}] {w}")

        seen = set()
        caveats = [x for x in caveats if not (x in seen or seen.add(x))]

        if direction not in {"positive", "negative"}:
            direction_label = "positive" if (med_rho is not None and np.isfinite(float(med_rho)) and float(med_rho) > 0) else "negative"
            caveats.append("Sinal variou entre safras (mixed). Recomendação apenas como pista; validar antes de agir.")
            priority = "low"
        else:
            direction_label = direction
            votes = 2
            priority = _priority_from_strength(abs_cliff, abs_dy, votes)
            # se suporte for 100%, dá um “empurrão” para HIGH quando impacto é grande
            if support_rate >= 0.99 and priority == "medium" and (abs_dy or 0) >= 500:
                priority = "high"
                rules_fired.append("support_100pct_boost")

        if direction_label == "positive":
            title = f"Alavanca consistente: {var}"
            claim = f"Em múltiplas safras, valores mais altos de '{var}' estiveram associados a maior produtividade."
            action_hint = "Investigar se há condição favorável replicável ou manejável (sem inferir causalidade)."
        else:
            title = f"Limitante consistente: {var}"
            claim = f"Em múltiplas safras, valores mais altos de '{var}' estiveram associados a menor produtividade."
            action_hint = "Investigar se a variável pode indicar restrição/ambiente desfavorável (sem inferir causalidade)."

        per_season = []
        for season, it in items:
            per_season.append(
                {
                    "safra": season,
                    "delta_yield_kgha": getattr(it, "delta_yield_q4_q1", None),
                    "rho": getattr(it, "spearman_rho", None),
                    "cliff": getattr(it, "cliffs_delta", None),
                    "auc": getattr(it, "auc", None),
                    "p": getattr(it, "mannwhitney_p", None),
                }
            )


        evidence = {
            "scope": "temporal",
            "yield_col": getattr(temporal_bundle, "yield_col", "yield"),
            "direction": direction_label,
            "temporal_support": {
                "seasons_supported": supported,
                "seasons_tested": tested,
                "support_rate": support_rate,
            },
            "median_effects": {
                "median_delta_y_kgha": med_dy,
                "median_rho": med_rho,
                "median_driver_score": med_score,
                "median_cliff": cliffs,
                "median_auc": aucs,
            },
            "per_season_evidence": per_season,
            "thresholds_used": {
                "min_support_rate": min_support_rate,
                "min_supported_seasons": min_supported_seasons,
                "min_delta_y_kgha": min_delta_y,
                "min_abs_cliff": MIN_ABS_CLIFF,
            },
        }

        recs.append(
            RecommendationItem(
                var=var,
                direction=direction_label,
                priority=priority,
                title=title,
                claim=claim,
                action_hint=action_hint,
                evidence=evidence,
                caveats=tuple(caveats),
                rules_fired=tuple(rules_fired),
                driver_score=float(med_score) if med_score is not None and np.isfinite(float(med_score)) else 0.0,
            )
        )

    recs_sorted = sorted(
        recs,
        key=lambda r: ({"high": 2, "medium": 1, "low": 0}[r.priority], r.driver_score),
        reverse=True,
    )
    recs_top = recs_sorted[:top_k]

    if observability is not None:
        observability.record_decision(
            "recommendation_engine_temporal_summary",
            decision="pass" if len(recs_top) else "warning",
            reason="recomendações temporais geradas a partir de consistência entre safras",
            metrics={
                "culture": culture,
                "top_k": top_k,
                "n_consistency_in": len(consistency),
                "n_recommendations_out": len(recs_top),
                "rejected_count": rejected,
                "min_support_rate": min_support_rate,
                "min_delta_y_kgha": min_delta_y,
                "top_recommendations": [
                    {"var": r.var, "priority": r.priority, "direction": r.direction, "driver_score": r.driver_score}
                    for r in recs_top
                ],
            },
        )
        observability.record_decision(
            "recommendation_engine_temporal_full",
            decision="recommendations",
            reason="lista completa (top_k) com evidências temporais",
            metrics={
                "recommendations": [
                    {
                        "var": r.var,
                        "priority": r.priority,
                        "direction": r.direction,
                        "title": r.title,
                        "claim": r.claim,
                        "action_hint": r.action_hint,
                        "caveats": list(r.caveats),
                        "rules_fired": list(r.rules_fired),
                        "driver_score": r.driver_score,
                        "evidence": r.evidence,
                    }
                    for r in recs_top
                ]
            },
        )

    bundle_warnings: List[str] = []
    if len(recs_top) == 0:
        bundle_warnings.append("Nenhuma recomendação temporal passou nos critérios conservadores.")

    return RecommendationBundle(
        ok=True,
        scope="temporal",
        culture=culture,
        top_k=top_k,
        recommendations=tuple(recs_top),
        rejected_count=rejected,
        warnings=tuple(_dedupe_preserve_order(list(warnings) + bundle_warnings)),
        metadata={
            "min_delta_y_kgha": min_delta_y,
            "min_abs_cliff": MIN_ABS_CLIFF,
            "min_support_rate": min_support_rate,
            "min_supported_seasons": min_supported_seasons,
            "note": "temporal recommendations are observational; no causal claims",
        },
    )
