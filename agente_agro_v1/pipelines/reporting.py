# -*- coding: utf-8 -*-
"""
reporting.py (v2)

Responsabilidade:
- Renderizar outputs consumíveis a partir de RecommendationBundle.
- Não faz gating, não altera dados, não "decide" nada.
- Produz:
  - resumo executivo (texto) -> para display no app
  - relatório técnico (texto/markdown) -> para backup/audit
  - payload para API (dict JSON-serializable)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import numpy as np


# -----------
# Helpers
# -----------
def _fmt_num(x: Any, nd: int = 2) -> str:
    if x is None:
        return "NA"
    try:
        if isinstance(x, (int,)):
            return f"{x:d}"
        xf = float(x)
        if not np.isfinite(xf):
            return "NA"
        return f"{xf:.{nd}f}"
    except Exception:
        return "NA"


def _fmt_signed(x: Any, nd: int = 0, unit: str = "") -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "NA"
        s = f"{xf:+.{nd}f}"
        return f"{s}{unit}"
    except Exception:
        return "NA"


def _fmt_abs(x: Any, nd: int = 0, unit: str = "") -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "NA"
        s = f"{abs(xf):.{nd}f}"
        return f"{s}{unit}"
    except Exception:
        return "NA"


def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _priority_label(priority: str) -> str:
    p = (priority or "").strip().lower()
    if p == "high":
        return "ALTA prioridade"
    if p == "medium":
        return "MÉDIA prioridade"
    if p == "low":
        return "BAIXA prioridade"
    return (priority or "prioridade").upper()


def _direction_phrase(direction: str) -> str:
    d = (direction or "").strip().lower()
    if d == "positive":
        return "associada a MAIOR produtividade"
    if d == "negative":
        return "associada a MENOR produtividade"
    return "associação indefinida"


def _safe_get(dct: Any, key: str, default=None):
    try:
        if dct is None:
            return default
        return dct.get(key, default)
    except Exception:
        return default


# -------------------------
# Camada 1 — Resumo executivo
# -------------------------
def render_executive_summary(rec_bundle: Any) -> str:
    """
    Resumo curto (para app / visão do gestor).
    Deve ser autoexplicativo e não depender de jargões estatísticos.

    Filosofia:
    - Prioridade = "por onde começar a investigar" nesta safra.
    - Sem inferência causal.
    """
    lines = []

    culture = getattr(rec_bundle, "culture", "NA")
    scope = getattr(rec_bundle, "scope", "intra-safra")
    top_k = getattr(rec_bundle, "top_k", "NA")

    lines.append(f"Resumo ({scope}) | Cultura: {culture}")
    lines.append(f"Principais pontos de atenção (Top {top_k}):")

    recs = list(getattr(rec_bundle, "recommendations", []) or [])
    if not recs:
        lines.append("- Nenhuma recomendação passou nos critérios conservadores.")
        lines.append("")
        lines.append("Nota: associações observacionais; interpretar com cautela.")
        return "\n".join(lines)

    for r in recs:
        ev = getattr(r, "evidence", {}) or {}

        dy = _safe_get(ev, "delta_yield_T3_T1_kgha", None)
        dy_s = _fmt_signed(dy, nd=0, unit=" kg/ha")

        d_phrase = _direction_phrase(getattr(r, "direction", ""))

        lines.append(
            f"- [{_priority_label(getattr(r, 'priority', 'NA'))}] {r.var}: {d_phrase} "
            f"(diferença estimada: {dy_s} entre X alto vs X baixo)"
        )

    lines.append("")
    lines.append("Legenda de prioridade:")
    lines.append("- ALTA prioridade: sinal mais consistente e com maior impacto potencial nesta safra (vale investigar primeiro).")
    lines.append("- MÉDIA prioridade: sinal moderado ou com mais incertezas (investigar depois).")
    lines.append("")
    lines.append("Nota: associações observacionais; não implicam causalidade. Recomenda-se checar ressalvas antes de agir.")
    return "\n".join(lines)


# -------------------------
# Camada 2 — Relatório técnico
# -------------------------
def render_technical_report(rec_bundle: Any, *, include_evidence: bool = True) -> str:
    """
    Relatório técnico em texto (markdown-friendly).
    Serve como backup/audit e para time técnico entender "por que isso apareceu".
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    culture = getattr(rec_bundle, "culture", "NA")
    scope = getattr(rec_bundle, "scope", "NA")
    recs = list(getattr(rec_bundle, "recommendations", []) or [])

    lines = []
    lines.append(f"# Relatório técnico — Agente Agronômico ({scope})")
    lines.append(f"- Gerado em: {ts}")
    lines.append(f"- Cultura: **{culture}**")
    lines.append(f"- Nº recomendações (após critérios): **{len(recs)}**")
    lines.append("")

    warnings = list(getattr(rec_bundle, "warnings", []) or [])
    if warnings:
        lines.append("## Avisos gerais")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if not recs:
        lines.append("## Recomendações")
        lines.append("Nenhuma recomendação passou nos critérios conservadores.")
        lines.append("")
        lines.append("## Observação final")
        lines.append("Este relatório apresenta associações observacionais extraídas dos dados da lavoura; interpretar com cautela.")
        return "\n".join(lines)

    # Glossário
    lines.append("## Como ler as evidências (glossário rápido)")
    lines.append("- **Diferença estimada (X alto vs X baixo)**: compara produtividade entre os pontos com valores mais altos de X e os pontos com valores mais baixos de X (por tercis).")
    lines.append("- **ρ (Spearman)**: correlação monotônica (sinal +/−); não mede causalidade.")
    lines.append("- **Cliff’s delta**: tamanho de efeito (|0.1| pequeno, |0.25| moderado, |0.4| grande; regra de bolso).")
    lines.append("- **AUC**: capacidade de separar grupos (0.5 ~ nenhum poder; >0.6 começa a separar).")
    lines.append("- **p (Mann–Whitney)**: evidência estatística de diferença entre grupos.")
    lines.append("")

    lines.append("## Recomendações (Top)")
    for i, r in enumerate(recs, start=1):
        pr = _priority_label(getattr(r, "priority", "NA"))
        direction = getattr(r, "direction", "NA")
        d_phrase = _direction_phrase(direction)

        lines.append(f"### {i}. {r.var} — {pr}")

        lines.append(f"**Leitura:** {d_phrase}.")
        lines.append(f"**Resumo:** {getattr(r, 'claim', '').strip()}")
        lines.append("")
        if getattr(r, "action_hint", None):
            lines.append(f"**Sugestão de ação:** {r.action_hint}")

        caveats = _dedupe_preserve_order(list(getattr(r, "caveats", []) or []))
        if caveats:
            lines.append("")
            lines.append("**Ressalvas / checagens recomendadas:**")
            for c in caveats:
                lines.append(f"- {c}")

        if include_evidence:
            ev = getattr(r, "evidence", {}) or {}
            lines.append("")
            lines.append("**Evidências (principais):**")

            dy = _safe_get(ev, "delta_yield_T3_T1_kgha", None)
            dyp = _safe_get(ev, "delta_yield_pct", None)
            lines.append(f"- Diferença estimada de produtividade (X alto − X baixo): {_fmt_signed(dy, nd=0, unit=' kg/ha')}")
            lines.append(f"- Diferença estimada (%): {_fmt_signed(dyp, nd=1, unit='%')}")

            lines.append(f"- ρ (Spearman): {_fmt_signed(_safe_get(ev, 'spearman_rho', None), nd=2)}")
            lines.append(f"- Cliff’s delta: {_fmt_signed(_safe_get(ev, 'cliffs_delta', None), nd=2)}")
            lines.append(f"- AUC: {_fmt_num(_safe_get(ev, 'auc', None), nd=2)}")
            lines.append(f"- Mann–Whitney p: {_fmt_num(_safe_get(ev, 'mannwhitney_p', None), nd=4)}")
            lines.append(f"- n válido: {_fmt_num(_safe_get(ev, 'n_valid', None), nd=0)}")
            lines.append(f"- outliers (fração): {_fmt_num(_safe_get(ev, 'outlier_frac', None), nd=3)}")

            xv = _safe_get(ev, "x_variability", {}) or {}
            lines.append("")
            lines.append("**Variabilidade do driver (X):**")
            lines.append(f"- mediana: {_fmt_num(_safe_get(xv, 'x_median', None), nd=3)}")
            lines.append(f"- IQR: {_fmt_num(_safe_get(xv, 'x_iqr', None), nd=3)}")
            lines.append(
                f"- q10–q90: {_fmt_num(_safe_get(xv, 'x_q10', None), nd=3)} → {_fmt_num(_safe_get(xv, 'x_q90', None), nd=3)}"
            )
            lines.append(f"- CV%: {_fmt_num(_safe_get(xv, 'x_cv_pct', None), nd=2)}")

            cc = _safe_get(ev, "cluster_context", {}) or {}
            if cc:
                lines.append("")
                lines.append("**Contexto (colinearidade):**")
                lines.append(f"- cluster_id: {cc.get('cluster_id', 'NA')}")
                cvs = cc.get("cluster_vars", []) or []
                if cvs:
                    lines.append(f"- variáveis no mesmo cluster: {', '.join(cvs)}")

            rf = list(getattr(r, "rules_fired", []) or [])
            if rf:
                lines.append("")
                lines.append("**Regras acionadas (audit):**")
                for rule in rf:
                    lines.append(f"- {rule}")

        lines.append("\n---\n")

    lines.append("## Observação final")
    lines.append("Este relatório apresenta associações observacionais extraídas dos dados da lavoura; interpretar com cautela.")
    return "\n".join(lines)


def render_full_report(rec_bundle: Any, *, include_evidence: bool = True) -> str:
    """
    Conveniência: retorna um único texto contendo:
    - Camada 1: Resumo executivo
    - Camada 2: Relatório técnico (backup/audit)
    """
    parts = [
        render_executive_summary(rec_bundle),
        "",
        "============================================================",
        "",
        render_technical_report(rec_bundle, include_evidence=include_evidence),
    ]
    return "\n".join(parts)


# ------------
# API payload
# ------------
def to_api_payload(rec_bundle: Any) -> Dict[str, Any]:
    """
    Payload estável para API (JSON-serializable).

    Obs: não inclui o texto do relatório por padrão (pode ser pesado),
    mas vocês podem gerar via render_* e armazenar como quiserem.
    """
    payload = {
        "ok": bool(getattr(rec_bundle, "ok", True)),
        "scope": getattr(rec_bundle, "scope", "NA"),
        "culture": getattr(rec_bundle, "culture", "NA"),
        "top_k": int(getattr(rec_bundle, "top_k", 0) or 0),
        "rejected_count": int(getattr(rec_bundle, "rejected_count", 0) or 0),
        "warnings": list(getattr(rec_bundle, "warnings", []) or []),
        "metadata": dict(getattr(rec_bundle, "metadata", {}) or {}),
        "recommendations": [],
    }

    recs = list(getattr(rec_bundle, "recommendations", []) or [])
    for r in recs:
        payload["recommendations"].append(
            {
                "var": r.var,
                "direction": r.direction,
                "priority": r.priority,
                "title": r.title,
                "claim": r.claim,
                "action_hint": r.action_hint,
                "driver_score": float(getattr(r, "driver_score", 0.0) or 0.0),
                "caveats": list(getattr(r, "caveats", []) or []),
                "rules_fired": list(getattr(r, "rules_fired", []) or []),
                "evidence": getattr(r, "evidence", {}) or {},
            }
        )
    return payload


# -----------------------
# Temporal reporting (v1)
# -----------------------

def render_temporal_full_report(rec_bundle: Any, *, include_evidence: bool = True) -> str:
    """
    Report temporal no MESMO estilo do report anual:
    - Resumo executivo (para app)
    - Relatório técnico (para audit)

    Diferença:
    - inclui explicitamente "consistência entre safras" (support X/Y)
    - Δyield é reportado como "mediana do efeito entre safras"
    """
    parts = [
        render_temporal_executive_summary(rec_bundle),
        "",
        "============================================================",
        "",
        render_temporal_technical_report(rec_bundle, include_evidence=include_evidence),
    ]
    return "\n".join(parts)


def render_temporal_executive_summary(rec_bundle: Any) -> str:
    lines = []
    culture = getattr(rec_bundle, "culture", "NA")
    top_k = getattr(rec_bundle, "top_k", "NA")
    lines.append(f"Resumo (consistência temporal) | Cultura: {culture}")
    lines.append(f"Principais pontos de atenção consistentes (Top {top_k}):")

    recs = list(getattr(rec_bundle, "recommendations", []) or [])
    if not recs:
        lines.append("- Nenhuma recomendação temporal passou nos critérios conservadores.")
        lines.append("")
        lines.append("Nota: consistência temporal exige pelo menos 2 safras e evidências observacionais.")
        return "\n".join(lines)

    for r in recs:
        ev = getattr(r, "evidence", {}) or {}
        tsup = ev.get("temporal_support", {}) or {}
        supported = tsup.get("seasons_supported", []) or []
        tested = tsup.get("seasons_tested", []) or []
        sr = tsup.get("support_rate", None)

        med = ev.get("median_effects", {}) or {}
        dy = med.get("median_delta_y_kgha", None)

        dy_s = _fmt_signed(dy, nd=0, unit=" kg/ha")
        sr_s = "NA" if sr is None else f"{float(sr):.0%}"

        d_phrase = _direction_phrase(getattr(r, "direction", ""))

        lines.append(
            f"- [{_priority_label(getattr(r, 'priority', 'NA'))}] {r.var}: {d_phrase} "
            f"(mediana Δyield: {dy_s}; suporte: {len(supported)}/{len(tested)} = {sr_s})"
        )

    lines.append("")
    lines.append("Nota: associações observacionais consistentes entre safras; não implicam causalidade.")
    return "\n".join(lines)


def render_temporal_technical_report(rec_bundle: Any, *, include_evidence: bool = True) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    culture = getattr(rec_bundle, "culture", "NA")
    recs = list(getattr(rec_bundle, "recommendations", []) or [])
    warnings = list(getattr(rec_bundle, "warnings", []) or [])

    lines = []
    lines.append("# Relatório técnico — Agente Agronômico (Consistência temporal)")
    lines.append(f"- Gerado em: {ts}")
    lines.append(f"- Cultura: **{culture}**")
    lines.append(f"- Nº recomendações (após critérios): **{len(recs)}**")
    lines.append("")

    if warnings:
        lines.append("## Avisos gerais")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Como ler (temporal)")
    lines.append("- **Suporte (X/Y)**: em quantas safras o driver apareceu com sinal consistente, dentre as safras avaliadas.")
    lines.append("- **Mediana Δyield**: efeito típico (mediano) observado entre safras, comparando X alto vs X baixo (por tercis).")
    lines.append("- As evidências por safra ficam detalhadas abaixo (quando disponíveis).")
    lines.append("")

    if not recs:
        lines.append("## Recomendações")
        lines.append("Nenhuma recomendação temporal passou nos critérios conservadores.")
        lines.append("")
        lines.append("## Observação final")
        lines.append("Este relatório apresenta associações observacionais consistentes; interpretar com cautela.")
        return "\n".join(lines)

    lines.append("## Recomendações (Top)")
    for i, r in enumerate(recs, start=1):
        pr = _priority_label(getattr(r, "priority", "NA"))
        d_phrase = _direction_phrase(getattr(r, "direction", ""))

        lines.append(f"### {i}. {r.var} — {pr}")
        lines.append(f"**Leitura:** {d_phrase}.")
        lines.append(f"**Resumo:** {getattr(r, 'claim', '').strip()}")
        if getattr(r, "action_hint", None):
            lines.append(f"**Sugestão de ação:** {r.action_hint}")

        caveats = _dedupe_preserve_order(list(getattr(r, "caveats", []) or []))
        if caveats:
            lines.append("")
            lines.append("**Ressalvas / checagens recomendadas:**")
            for c in caveats:
                lines.append(f"- {c}")

        if include_evidence:
            ev = getattr(r, "evidence", {}) or {}
            tsup = ev.get("temporal_support", {}) or {}
            med = ev.get("median_effects", {}) or {}

            supported = tsup.get("seasons_supported", []) or []
            tested = tsup.get("seasons_tested", []) or []
            sr = tsup.get("support_rate", None)
            sr_s = "NA" if sr is None else f"{float(sr):.0%}"

            lines.append("")
            lines.append("**Evidência temporal (agregada):**")
            lines.append(f"- Suporte: {len(supported)}/{len(tested)} ({sr_s})")
            lines.append(f"- Mediana Δyield (X alto − X baixo): {_fmt_signed(med.get('median_delta_y_kgha', None), nd=0, unit=' kg/ha')}")
            lines.append(f"- Mediana ρ (Spearman): {_fmt_signed(med.get('median_rho', None), nd=2)}")
            lines.append(f"- Mediana Cliff’s delta: {_fmt_signed(med.get('median_cliff', None), nd=2)}")
            lines.append(f"- Mediana AUC: {_fmt_num(med.get('median_auc', None), nd=2)}")

            pse = ev.get("per_season_evidence", []) or []
            if pse:
                lines.append("")
                lines.append("**Evidência por safra (detalhe):**")
                for row in pse:
                    lines.append(
                        f"- [{row.get('safra','NA')}] Δyield={_fmt_signed(row.get('delta_yield_kgha', None), nd=0, unit=' kg/ha')}, "
                        f"ρ={_fmt_signed(row.get('rho', None), nd=2)}, "
                        f"Cliff={_fmt_signed(row.get('cliff', None), nd=2)}, "
                        f"AUC={_fmt_num(row.get('auc', None), nd=2)}, "
                        f"p={_fmt_num(row.get('p', None), nd=4)}"
                    )

            rf = list(getattr(r, "rules_fired", []) or [])
            if rf:
                lines.append("")
                lines.append("**Regras acionadas (audit):**")
                for rule in rf:
                    lines.append(f"- {rule}")

        lines.append("\n---\n")

    lines.append("## Observação final")
    lines.append("Este relatório apresenta associações observacionais consistentes entre safras; interpretar com cautela.")
    return "\n".join(lines)
