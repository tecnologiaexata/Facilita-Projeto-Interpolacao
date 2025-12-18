"""
pipeline_completa.py - Orquestração do fluxo completo de processamento agrícola

Fluxo:
  1. Contorno + grade: ProcessadorLavoura
  2. CSV de amostragem: ProcessadorAmostragem
  3. Definição de atributos interpoláveis conforme o processo (solo, prod, compac, nemat, foliar)
  4. Análise espacial: AnalisadorEspacial
  5. Interpolação: InterpoladorAtributo (GeoTIFF por atributo)
  6. Amostragem dos rasters na grade: tabela wide por campanha
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from .logger import LoggerAgricola, NivelLog
from .processador_lavoura import ProcessadorLavoura
from .processador_amostragem import ProcessadorAmostragem
from .analise_espacial import AnalisadorEspacial
from .interpolador import InterpoladorAtributo


# -------------------- Configuração de processos --------------------

PROCESSOS_VALIDOS = {"solo", "prod", "compac", "nemat", "foliar"}

# Colunas "comuns" da plataforma (presentes em todos os processos)
CAMPOS_COMUNS = ["Data", "Cliente", "Fazenda", "Talhão", "Gleba", "Lat", "Long"]

# Colunas oficiais de cada processo (do pdf fornecdio)
COLS_PROCESSO = {
    "solo": [
        "Profundidade", "Ponto",
        "pH H2O", "pH CaCl", "pH SMP",
        "P meh", "P res", "P total",
        "Na", "K", "S", "Ca", "Mg", "Al", "H + Al", "MO", "CO",
        "B", "Cu", "Fe", "Mn", "Zn",
        "SB", "t", "T", "V %", "m %",
        "Ca/Mg", "Ca/K", "Mg/K", "(Ca+Mg)/K",
        "Ca/t", "Mg/t", "Ca/T", "Mg/T", "K/T",
        "(H+Al)/T", "(Ca+Mg)/T", "(Ca+Mg+K)/T",
        "Argila", "Silte", "Areia Total", "Areia Grossa", "Areia Fina", "Cascalho",
    ],
    "prod": [
        "Cultura", "Variedade", "Data de plantio", "Data de colheita",
        "Acúmulo de chuva (mm)", "Média de produção Kg/Ha (Balança)",
        "Kg/Ha",
    ],
    "compac": ["Ponto", "Profundidade", "Kpa"],
    "nemat": ["Espécie", "Quantidade", "U/100cc"],
    "foliar": [
        "N", "P", "K", "Ca", "Mg", "S",
        "B", "Cu", "Fe", "Mn", "Zn",
        "Massa Fresca", "Massa Seca",
    ],
}


COLS_INTERPOLAVEIS: Dict[str, List[str]] = {
    "solo": [
        "pH H2O", "pH CaCl", "pH SMP",
        "P meh", "P res", "P total",
        "Na", "K", "S", "Ca", "Mg", "Al", "H + Al", "MO", "CO",
        "B", "Cu", "Fe", "Mn", "Zn",
        "SB", "t", "T", "V %", "m %",
        "Ca/Mg", "Ca/K", "Mg/K", "(Ca+Mg)/K",
        "Ca/t", "Mg/t", "Ca/T", "Mg/T", "K/T",
        "(H+Al)/T", "(Ca+Mg)/T", "(Ca+Mg+K)/T",
        "Argila", "Silte", "Areia Total", "Areia Grossa", "Areia Fina", "Cascalho",
    ],

    "prod": ["Kg/Ha"],

    "compac": ["Kpa"],

    "nemat": ["Quantidade", "U/100cc"],

    "foliar": [
        "N", "P", "K", "Ca", "Mg", "S",
        "B", "Cu", "Fe", "Mn", "Zn",
        "Massa Fresca", "Massa Seca",
    ],
}


class PipelineAgricola:
    """
    Orquestra o pipeline completo para uma lavoura e um arquivo de amostragem,
    dado um tipo de processo (solo, prod, compac, nemat, foliar).
    """

    def __init__(
        self,
        logger: LoggerAgricola,
        dir_contornos: str | Path,
        dir_amostragem: str | Path,
        dir_rasters: str | Path,
        dir_csv_base: str | Path,
        pixel_size: float = 10.0,
    ) -> None:
        self.logger = logger
        self.dir_contornos = Path(dir_contornos)
        self.dir_amostragem = Path(dir_amostragem)
        self.dir_rasters = Path(dir_rasters)
        self.dir_csv_base = Path(dir_csv_base)
        self.pixel_size = float(pixel_size)

    # ---------------------------------------------------------
    # Helpers de log
    # ---------------------------------------------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)

    # ---------------------------------------------------------
    # Helpers: valores únicos e rótulo de campanha
    # ---------------------------------------------------------

    def _valor_unico(
        self,
        df: pd.DataFrame,
        nome_coluna: Optional[str],
        etapa: str,
        nome_logico: str,
    ) -> Optional[str]:
        """
        Tenta extrair um valor único de uma coluna (ex.: Data, Cliente, etc.).
        - Se não existir ou só tiver NaN -> retorna None
        - Se tiver mais de um valor distinto -> loga warning e usa o primeiro
        """
        if nome_coluna is None or nome_coluna not in df.columns:
            return None

        serie = df[nome_coluna].dropna().astype(str)
        if serie.empty:
            return None

        valores = serie.unique()
        if len(valores) > 1:
            self._log(
                NivelLog.WARNING,
                etapa,
                f"Coluna '{nome_logico}' possui múltiplos valores. "
                f"Usando o primeiro: {valores[0]}",
                dados={"coluna": nome_coluna, "valores": valores.tolist()},
                mostrar_usuario=True,
            )
        return str(valores[0])

    def _obter_rotulo_campanha(
        self,
        proc_am: ProcessadorAmostragem,
    ) -> str:
        """
        Define o rótulo da campanha para nome de arquivos:

        - Usa a coluna 'Data' do CSV (proc_am.col_data)
        - Sanitiza para uso em nome de arquivo
        - Se não conseguir, usa 'campanha'
        """
        rotulo = None

        if proc_am.df_raw is not None and getattr(proc_am, "col_data", None):
            valor_data = self._valor_unico(
                proc_am.df_raw,
                proc_am.col_data,
                etapa="rotulo_campanha",
                nome_logico="Data",
            )
            if valor_data:
                rotulo = (
                    valor_data.strip()
                    .replace("/", "-")
                    .replace("\\", "-")
                    .replace(" ", "_")
                    .replace(":", "-")
                )

        if not rotulo:
            rotulo = "campanha"

        self._log(
            NivelLog.INFO,
            "rotulo_campanha",
            f"Rótulo de campanha definido como '{rotulo}'.",
            mostrar_usuario=True,
        )

        return rotulo


    # ---------------------------------------------------------
    # Helpers: diretórios de saída padronizados (lavoura/processo/campanha)
    # ---------------------------------------------------------

    def _dirs_saida(
        self,
        nome_lavoura: str,
        processo: str,
        rotulo_campanha: str,
    ) -> Tuple[Path, Path]:
        """
        Retorna (dir_rasters_campanha, dir_csv_processo) no padrão:

          - Rasters: dir_rasters / {nome_lavoura} / "{processo}_{rotulo_campanha}"
          - CSVs:    dir_csv_base / {nome_lavoura} / "csv_{processo}"

        Ex.: outputs/rasters/pivo/solo_05-05-2025/
             outputs/csv/pivo/csv_solo/
        """
        lavoura_s = str(nome_lavoura).strip()
        processo_s = str(processo).strip().lower()
        rotulo_s = str(rotulo_campanha).strip()

        dir_rasters_campanha = self.dir_rasters / lavoura_s / f"{processo_s}_{rotulo_s}"
        dir_csv_processo = self.dir_csv_base / lavoura_s / f"csv_{processo_s}"

        dir_rasters_campanha.mkdir(parents=True, exist_ok=True)
        dir_csv_processo.mkdir(parents=True, exist_ok=True)

        return dir_rasters_campanha, dir_csv_processo
    # ---------------------------------------------------------
    # Etapa 1: preparar lavoura (contorno + grade)
    # ---------------------------------------------------------

    def _preparar_lavoura(
        self,
        nome_lavoura: str,
    ) -> Tuple[bool, str, Optional[ProcessadorLavoura]]:
        proc_lav = ProcessadorLavoura(
            nome_lavoura=nome_lavoura,
            diretorio_base=self.dir_contornos,
            logger=self.logger,
        )

        ok, msg = proc_lav.preparar_base_amostragem()
        if not ok:
            return False, msg, None

        return True, msg, proc_lav

    # ---------------------------------------------------------
    # Etapa 2: processar amostragem (CSV → gdf_utm + atributos brutos)
    # ---------------------------------------------------------

    def _processar_amostragem(
        self,
        nome_lavoura: str,
        nome_arquivo_csv: str,
        proc_lav: ProcessadorLavoura,
    ) -> Tuple[bool, str, Optional[ProcessadorAmostragem]]:
        caminho_csv = self.dir_amostragem / nome_arquivo_csv

        proc_am = ProcessadorAmostragem(
            nome_lavoura=nome_lavoura,
            caminho_csv=caminho_csv,
            contorno_utm=proc_lav.contorno,
            crs_utm=proc_lav.crs_utm,
            logger=self.logger,
        )

        ok, msg, gdf_utm, atributos = proc_am.processar_amostragem()
        if not ok:
            return False, msg, None

        proc_am.gdf_utm = gdf_utm
        proc_am.col_atributos = atributos

        return True, msg, proc_am

    # ---------------------------------------------------------
    # Helper: seleção de atributos por processo
    # ---------------------------------------------------------

    def _selecionar_atributos_processo(
        self,
        atributos_disponiveis: List[str],
        processo: str,
    ) -> List[str]:
        """
        Intersecta os atributos numéricos disponíveis (vindos do ProcessadorAmostragem)
        com a lista oficial de colunas interpoláveis para o processo escolhido,
        preservando a ordem definida em COLS_INTERPOLAVEIS[processo].
        """
        processo_key = processo.lower()

        if processo_key not in COLS_INTERPOLAVEIS:
            msg = (
                f"Processo '{processo}' não reconhecido em COLS_INTERPOLAVEIS. "
                f"Opções: {list(COLS_INTERPOLAVEIS.keys())}"
            )
            self._log(
                NivelLog.ERROR,
                "selecionar_atributos_processo",
                msg,
                mostrar_usuario=True,
            )
            raise ValueError(msg)

        alvo = COLS_INTERPOLAVEIS[processo_key]
        disp_set = set(atributos_disponiveis)

        selecionados = [c for c in alvo if c in disp_set]

        self._log(
            NivelLog.INFO,
            "selecionar_atributos_processo",
            f"Atributos disponíveis: {sorted(atributos_disponiveis)} | "
            f"Alvo ({processo_key}): {alvo} | Selecionados: {selecionados}",
            mostrar_usuario=True,
        )

        return selecionados

    # ---------------------------------------------------------
    # Etapa 3: analisar atributos para os interpoláveis
    # ---------------------------------------------------------

    def _analisar_atributos(
        self,
        proc_am: ProcessadorAmostragem,
        processo: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Filtra os atributos interpoláveis conforme o processo
        e roda a análise espacial apenas nesses atributos.
        """
        attrs_interp = self._selecionar_atributos_processo(
            atributos_disponiveis=proc_am.col_atributos,
            processo=processo,
        )

        if not attrs_interp:
            msg = (
                f"Nenhum atributo interpolável encontrado para o processo '{processo}'. "
                f"Atributos disponíveis (numéricos): {sorted(proc_am.col_atributos)}"
            )
            self._log(
                NivelLog.ERROR,
                "analise_atributos",
                msg,
                mostrar_usuario=True,
            )
            return {}, []

        analisador = AnalisadorEspacial(logger=self.logger)

        resultados = analisador.analisar_atributos(
            proc_am.gdf_utm,
            attrs_interp,
        )

        resumo = {attr: resultados[attr]["metodo"] for attr in attrs_interp}
        self._log(
            NivelLog.INFO,
            "analise_atributos",
            f"Métodos escolhidos por atributo (processo={processo}): {resumo}",
            mostrar_usuario=True,
        )

        return resultados, attrs_interp

    # ---------------------------------------------------------
    # Etapa 4: interpolar todos os atributos selecionados
    # ---------------------------------------------------------

    def _interpolar_todos_atributos(
        self,
        nome_lavoura: str,
        rotulo_campanha: str,
        processo: str,
        proc_lav: ProcessadorLavoura,
        proc_am: ProcessadorAmostragem,
        analise_atributos: Dict[str, Dict[str, Any]],
        attrs_interp: List[str],
    ) -> Dict[str, Path]:
        """
        Para cada atributo da lista attrs_interp, interpolar e salvar raster.
        Retorna dict { atributo: caminho_tif }.
        """
        interp = InterpoladorAtributo(
            contorno_utm=proc_lav.contorno,
            crs_utm=proc_lav.crs_utm,
            logger=self.logger,
            pixel_size=self.pixel_size,
        )

        # Diretório de saída padronizado: {dir_rasters}/{lavoura}/{processo}_{campanha}
        dir_rasters_campanha, _ = self._dirs_saida(nome_lavoura, processo, rotulo_campanha)

        caminhos: Dict[str, Path] = {}

        for attr in attrs_interp:
            metodo = analise_atributos[attr]["metodo"]
            ok, msg, caminho_tif = interp.interpolar_atributo(
                gdf_utm=proc_am.gdf_utm,
                value_col=attr,
                metodo=metodo,
                nome_lavoura=nome_lavoura,
                ano_amostragem=rotulo_campanha, 
                diretorio_saida=dir_rasters_campanha,
            )
            if ok and caminho_tif is not None:
                caminhos[attr] = caminho_tif
            else:
                self._log(
                    NivelLog.ERROR,
                    "interpolar_todos_atributos",
                    f"Falha ao interpolar atributo '{attr}'. "
                    "Ele será ignorado na etapa de amostragem.",
                    mostrar_usuario=True,
                )

        return caminhos

    # ---------------------------------------------------------
    # Etapa 5: amostrar rasters na grade e montar DataFrame
    # ---------------------------------------------------------

    def _amostrar_rasters_na_grade(
        self,
        proc_lav: ProcessadorLavoura,
        proc_am: ProcessadorAmostragem,
        processo: str,
        caminhos_rasters: Dict[str, Path],
    ) -> pd.DataFrame:
        """
        Para cada raster, amostra o valor nos pontos da grade.
        Constrói um DataFrame com:

          - id_ponto
          - Lat, Long (EPSG:4326)
          - Data, Cliente, Fazenda, Talhão, Gleba
          - todas as colunas oficiais do processo (mesmo se vazias)
          - colunas dos atributos interpolados (nome simples, sem sufixo/ano)
        """
        grade = proc_lav.grade.copy()

        if "id_ponto" not in grade.columns:
            grade["id_ponto"] = range(len(grade))

        xs = grade.geometry.x.values
        ys = grade.geometry.y.values
        coords = list(zip(xs, ys))

        grade_4326 = grade.to_crs(4326)
        lons = grade_4326.geometry.x.values
        lats = grade_4326.geometry.y.values

        df = pd.DataFrame(
            {
                "id_ponto": grade["id_ponto"].values,
                "Long": lons,
                "Lat": lats,
            }
        )

        if proc_am.df_raw is not None:
            df["Data"] = self._valor_unico(
                proc_am.df_raw,
                getattr(proc_am, "col_data", None),
                etapa="amostrar_rasters",
                nome_logico="Data",
            )
            df["Cliente"] = self._valor_unico(
                proc_am.df_raw,
                getattr(proc_am, "col_cliente", None),
                etapa="amostrar_rasters",
                nome_logico="Cliente",
            )
            df["Fazenda"] = self._valor_unico(
                proc_am.df_raw,
                getattr(proc_am, "col_fazenda", None),
                etapa="amostrar_rasters",
                nome_logico="Fazenda",
            )
            df["Talhão"] = self._valor_unico(
                proc_am.df_raw,
                getattr(proc_am, "col_talhao", None),
                etapa="amostrar_rasters",
                nome_logico="Talhão",
            )
            df["Gleba"] = self._valor_unico(
                proc_am.df_raw,
                getattr(proc_am, "col_gleba", None),
                etapa="amostrar_rasters",
                nome_logico="Gleba",
            )
        else:
            df["Data"] = None
            df["Cliente"] = None
            df["Fazenda"] = None
            df["Talhão"] = None
            df["Gleba"] = None

        for attr, caminho_tif in caminhos_rasters.items():
            with rasterio.open(caminho_tif) as src:
                sampled = list(src.sample(coords))
                vals = np.array([v[0] for v in sampled])

                nodata = src.nodata
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)

                df[attr] = vals

            self._log(
                NivelLog.INFO,
                "amostrar_rasters",
                f"Atributo '{attr}' amostrado na grade ({len(df)} pontos).",
            )

        cols_oficiais = COLS_PROCESSO.get(processo, [])
        for col in cols_oficiais:
            if col not in df.columns:
                df[col] = np.nan

        cols_base = ["id_ponto", "Data", "Cliente", "Fazenda", "Talhão", "Gleba", "Lat", "Long"]
        cols_processo = cols_oficiais  # já garantidas
        cols_restantes = [c for c in df.columns if c not in cols_base + cols_processo]

        df = df[cols_base + cols_processo + cols_restantes]

        return df

    # ---------------------------------------------------------
    # Etapa 6: exportar CSV final na pasta do processo
    # ---------------------------------------------------------

    def _exportar_csv_final(
        self,
        df: pd.DataFrame,
        nome_lavoura: str,
        rotulo_campanha: str,
        processo: str,
    ) -> Path:
        """
        Exporta o CSV da campanha em:

            dir_csv_base / {nome_lavoura} / f"csv_{processo}" / "{lavoura}_{processo}_{rotulo}_grid.csv"
        """
        _, dir_processo = self._dirs_saida(nome_lavoura, processo, rotulo_campanha)

        nome_arquivo = f"{nome_lavoura}_{processo}_{rotulo_campanha}_grid.csv"
        caminho_csv = dir_processo / nome_arquivo

        df.to_csv(caminho_csv, index=False)

        self._log(
            NivelLog.INFO,
            "exportar_csv",
            f"Arquivo CSV final exportado: {caminho_csv}",
            mostrar_usuario=True,
        )

        return caminho_csv

    # ---------------------------------------------------------
    # Método público: pipeline completa
    # ---------------------------------------------------------

    def executar_pipeline_completa(
        self,
        nome_lavoura: str,
        nome_arquivo_csv: str,
        processo: str,
    ) -> Tuple[bool, str, Optional[pd.DataFrame], Optional[Path]]:
        """
        Executa o pipeline completo para uma lavoura e um arquivo de amostragem.

        Args:
            nome_lavoura: nome da lavoura (deve casar com *_contorno.kml, *_grade.gpkg)
            nome_arquivo_csv: arquivo CSV de amostragem (dentro de dir_amostragem)
            processo: tipo do processo ('solo', 'prod', 'compac', 'nemat', 'foliar')

        Retorna:
            (sucesso, mensagem_final, df_resultado, caminho_csv_final_ou_None)
        """

        processo = processo.lower()
        if processo not in PROCESSOS_VALIDOS:
            msg = f"Processo inválido: '{processo}'. Esperado: {sorted(PROCESSOS_VALIDOS)}."
            self._log(
                NivelLog.ERROR,
                "executar_pipeline",
                msg,
                mostrar_usuario=True,
            )
            return False, msg, None, None

        self.logger.adicionar_contexto(
            lavoura=nome_lavoura,
            processo=processo,
            pixel_size=self.pixel_size,
        )

        ok, msg_lav, proc_lav = self._preparar_lavoura(nome_lavoura)
        if not ok or proc_lav is None:
            return False, msg_lav, None, None

        ok, msg_am, proc_am = self._processar_amostragem(
            nome_lavoura, nome_arquivo_csv, proc_lav
        )
        if not ok or proc_am is None:
            return False, msg_am, None, None

        rotulo_campanha = self._obter_rotulo_campanha(proc_am)

        analise_atributos, attrs_interp = self._analisar_atributos(proc_am, processo)
        if not analise_atributos or not attrs_interp:
            msg = (
                f"Nenhum atributo interpolável pôde ser analisado para processo '{processo}'. "
                "Pipeline interrompida."
            )
            return False, msg, None, None

        caminhos_rasters = self._interpolar_todos_atributos(
            nome_lavoura,
            rotulo_campanha,
            processo,
            proc_lav,
            proc_am,
            analise_atributos,
            attrs_interp,
        )

        if not caminhos_rasters:
            msg = "Nenhum raster foi gerado. Pipeline interrompida."
            self._log(
                NivelLog.ERROR,
                "executar_pipeline",
                msg,
                mostrar_usuario=True,
            )
            return False, msg, None, None

        df_final = self._amostrar_rasters_na_grade(
            proc_lav,
            proc_am,
            processo,
            caminhos_rasters,
        )

        caminho_csv = self._exportar_csv_final(
            df_final,
            nome_lavoura,
            rotulo_campanha,
            processo,
        )

        msg_final = (
            f"Pipeline concluída para lavoura '{nome_lavoura}', "
            f"processo '{processo}', campanha '{rotulo_campanha}'. "
            f"Pontos: {len(df_final)}, colunas: {len(df_final.columns)}."
        )

        self._log(
            NivelLog.INFO,
            "executar_pipeline",
            msg_final,
            mostrar_usuario=True,
        )

        return True, msg_final, df_final, caminho_csv
