from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import geopandas as gpd

from .logger import LoggerAgricola, NivelLog
from .processador_amostragem import ProcessadorAmostragem


class ProcessadorAmostragemV2(ProcessadorAmostragem):
    """
    Processa amostragem a partir de payload (sem CSV).

    Mantém o fluxo original do ProcessadorAmostragem:
      1. carregar_csv (sobrescrito para payload)
      2. _detectar_campos_comuns_plataforma
      3. detectar_lat_lon
      4. criar_gdf_4326
      5. reprojetar_para_utm
      6. checar_pontos_dentro_contorno
      7. identificar_atributos (genérico)
    """

    def __init__(
        self,
        nome_lavoura: str,
        dados: List[Dict[str, Any]],
        metadados: Dict[str, Any],
        contorno_utm: gpd.GeoDataFrame,
        crs_utm: str,
        logger: Optional[LoggerAgricola] = None,
    ) -> None:
        super().__init__(
            nome_lavoura=nome_lavoura,
            caminho_csv=Path("payload_v2"),
            contorno_utm=contorno_utm,
            crs_utm=crs_utm,
            logger=logger,
        )
        self.dados = dados
        self.metadados = metadados

    def carregar_csv(self) -> Tuple[bool, str]:
        """
        Constrói o DataFrame bruto a partir do payload.
        """
        if not self.dados:
            msg = "Payload sem dados de amostragem."
            self._log(NivelLog.ERROR, "carregar_payload", msg, mostrar_usuario=True)
            return False, msg

        self._log(
            NivelLog.INFO,
            "carregar_payload",
            f"Processando {len(self.dados)} registros do payload V2.",
            mostrar_usuario=True,
        )
        rows: list[dict[str, Any]] = []
        atributos_disponiveis: set[str] = set()

        for dado in self.dados:
            atributos = dado.get("atributos") or []
            attr_map = {
                item.get("nome"): item.get("valor")
                for item in atributos
                if isinstance(item, dict) and item.get("nome") is not None
            }
            atributos_disponiveis.update(attr_map.keys())

            row = {
                "latitude": dado.get("latitude"),
                "longitude": dado.get("longitude"),
                "Ponto": dado.get("ponto_coleta"),
                "Profundidade": dado.get("profundidade", self.metadados.get("profundidade")),
                "Data": self.metadados.get("data"),
                "Cliente": self.metadados.get("cliente"),
                "Fazenda": self.metadados.get("fazenda"),
                "Talhão": self.metadados.get("talhao"),
                "Gleba": self.metadados.get("gleba"),
                "Cultura": self.metadados.get("cultura"),
            }
            row.update(attr_map)
            rows.append(row)

        df = pd.DataFrame(rows)
        for attr in atributos_disponiveis:
            if attr not in df.columns:
                df[attr] = None

        self.df_raw = df
        msg = f"Payload V2 carregado: {len(df)} linhas, {len(df.columns)} colunas."
        self._log(NivelLog.INFO, "carregar_payload", msg, mostrar_usuario=True)
        return True, msg
