"""
motor_calculos.py

Repositório de recomendações e fórmulas

- Aqui ficam as fórmulas (ou funções) de recomendação.
- Também fica o catálogo de metadados:
  - required_soil_vars: variáveis necessárias da análise de solo
  - required_params: parâmetros obrigatórios do usuário (config)
  - units, output_field

Exemplo implementado: calcário (IAC/SP)
Dose (unidade depende do produto) — no exemplo de calcário, t/ha.

Nomes conforme plataforma do Facilita Agro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Regras
# ----------------------------
def calcular_calcario(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Calcula dose de calcário (ex.: t/ha) para cada linha (ponto ou pixel).

    Necessário no df:
      - "T"   : CTC efetiva
      - "V %" : saturação por bases atual

    Necessário em params:
      - "V_desejado" (ex.: 80)
      - "PRNT"       (ex.: 80)
    """
    # parâmetros obrigatórios
    if params.get("V_desejado", None) is None:
        raise ValueError("Não executado pois o valor obrigatório de V_desejado não foi informado")
    if params.get("PRNT", None) is None:
        raise ValueError("Não executado pois o valor obrigatório de PRNT não foi informado")

    Vd = float(params["V_desejado"])
    PRNT = float(params["PRNT"])

    # variáveis de solo
    if "T" not in df.columns:
        raise ValueError('Variável de solo obrigatória ausente: "T"')
    if "V %" not in df.columns:
        raise ValueError('Variável de solo obrigatória ausente: "V %"')

    T = pd.to_numeric(df["T"], errors="coerce")
    Va = pd.to_numeric(df["V %"], errors="coerce")

    # fórmula base
    dose = ( T * 10 * (Vd - Va) ) / (10.0 * PRNT)

    dose = dose.where((Vd - Va) > 0, 0.0)

    return dose


# ----------------------------
# Registry 
# ----------------------------
@dataclass(frozen=True)
class RecommendationSpec:
    key: str
    required_soil_vars: List[str]
    required_params: List[str]
    output_field: str

    fn: Callable[[pd.DataFrame, Dict[str, Any]], pd.Series] = field(repr=False)

    units: str = "kg/ha"
    decimals: int = 2
    suggested_round_step: Optional[float] = None
    suggested_min: Optional[float] = None
    suggested_max: Optional[float] = None




_RECOMMENDATIONS: Dict[str, RecommendationSpec] = {
    "calcario": RecommendationSpec(
        key="calcario",
        required_soil_vars=["T", "V %"],
        required_params=["V_desejado", "PRNT"],
        output_field="taxa_aplicacao",
        units="t/ha",
        decimals=2,
        suggested_round_step=0.5,
        suggested_min=None,
        suggested_max=None,
        fn=calcular_calcario,
    )
}

_EXPORT_FORMATS: List[str] = ["shp", "all", "stara"]


# ----------------------------
# API do módulo
# ----------------------------
def list_recommendations() -> List[str]:
    return sorted(_RECOMMENDATIONS.keys())


def get_recommendations_catalog() -> List[Dict[str, Any]]:
    """Catálogo para UI/integração (pode ser exposto na API)."""
    out: List[Dict[str, Any]] = []
    for k in list_recommendations():
        spec = _RECOMMENDATIONS[k]
        out.append(
            {
                "key": spec.key,
                "required_soil_vars": list(spec.required_soil_vars),
                "required_params": list(spec.required_params),
                "output_field": spec.output_field,
                "units": spec.units,
                "decimals": int(spec.decimals),
                "suggested_round_step": spec.suggested_round_step,
                "suggested_min": spec.suggested_min,
                "suggested_max": spec.suggested_max,
            }
        )
    return out


def get_export_formats_catalog() -> List[str]:
    return list(_EXPORT_FORMATS)


def get_required_soil_vars(rec_key: str) -> List[str]:
    spec = _RECOMMENDATIONS.get(rec_key)
    return list(spec.required_soil_vars) if spec else []


def get_required_params(rec_key: str) -> List[str]:
    spec = _RECOMMENDATIONS.get(rec_key)
    return list(spec.required_params) if spec else []



def get_recommendation_spec(rec_key: str) -> RecommendationSpec:
    """Retorna o spec completo da recomendação (metadados + função)."""
    spec = _RECOMMENDATIONS.get(rec_key)
    if not spec:
        raise KeyError(f"Recomendação não encontrada: {rec_key}")
    return spec


def get_recommendation_fn(rec_key: str) -> Callable[[pd.DataFrame, Dict[str, Any]], pd.Series]:
    spec = _RECOMMENDATIONS.get(rec_key)
    if not spec:
        raise KeyError(f"Recomendação não encontrada: {rec_key}")
    return spec.fn
