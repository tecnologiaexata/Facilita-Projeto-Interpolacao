"""
pipeline_yield.py - Pipeline específica para interpolação de mapas de produtividade (yield)

Fluxo "staging" (múltiplas colhedoras / múltiplos arquivos):
   1. Contorno + grade: ProcessadorLavoura
   2. Ler arquivos pré-processados (já filtrados + corrigidos) em:
        dir_yield_temp / {nome_lavoura} / {projeto_colheita} / *.gpkg|*.parquet
   3. Juntar tudo (concat) e interpolar uma única vez
   4. Amostragem do raster na grade
   5. Exportar raster e CSV final em subpastas:
        dir_rasters  / {nome_lavoura} / {projeto_colheita}/
        dir_csv_base / {nome_lavoura} / {projeto_colheita}/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from .logger import LoggerAgricola, NivelLog
from .processador_lavoura import ProcessadorLavoura
from .processador_amostragem import ProcessadorAmostragem 
from .interpolador import InterpoladorAtributo


class PipelineYield:
    """
    Orquestra o processamento de mapas de produtividade (módulo produção).
    """
    def __init__(
        self,
        logger: LoggerAgricola,
        dir_contornos: str | Path,
        dir_yield: str | Path,
        dir_rasters: str | Path,
        dir_csv_base: str | Path,
        pixel_size: float = 10.0,
        dir_yield_temp: Optional[str | Path] = None,
    ) -> None:
        self.logger = logger
        self.dir_contornos = Path(dir_contornos)
        self.dir_yield = Path(dir_yield)
        self.dir_rasters = Path(dir_rasters)
        self.dir_csv_base = Path(dir_csv_base)
        self.pixel_size = float(pixel_size)

        self.dir_yield_temp = Path(dir_yield_temp) if dir_yield_temp else Path("outputs") / "yield_temp"

    # ---------------------------------------------------------
    # Helpers: log e sanitização
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

    def _sanitizar_nome(self, texto: str) -> str:
        """
        Gera um nome 'seguro' para arquivo/pasta:

        - remove acentos
        - converte não alfanumérico em '_'
        - comprime múltiplos '_' em um só
        - remove '_' do começo/fim
        """
        s = unicodedata.normalize("NFKD", str(texto))
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    # ---------------------------------------------------------
    # Helpers: valores únicos
    # ---------------------------------------------------------

    def _valor_unico(
        self,
        df: pd.DataFrame,
        nome_coluna: Optional[str],
        etapa: str,
        nome_logico: str,
    ) -> Optional[str]:
        """
        Extrai um valor único de uma coluna (ex.: Data, Cliente, etc.).
        Se houver múltiplos, loga e usa o primeiro.
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
                f"Coluna '{nome_logico}' possui múltiplos valores. Usando o primeiro: {valores[0]}",
                dados={"coluna": nome_coluna, "valores": valores.tolist()},
                mostrar_usuario=True,
            )
        return str(valores[0])

    # ---------------------------------------------------------
    # 1) Lavoura (contorno + grade)
    # ---------------------------------------------------------

    def _preparar_lavoura(self, nome_lavoura: str) -> Tuple[bool, str, Optional[ProcessadorLavoura]]:
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
    # 2B) Carregar yield PRÉ-PROCESSADO do staging
    # ---------------------------------------------------------

    def _listar_arquivos_preproc(
        self,
        nome_lavoura: str,
        projeto_colheita: str,
    ) -> List[Path]:
        lav_s = self._sanitizar_nome(nome_lavoura)
        proj_s = self._sanitizar_nome(projeto_colheita)

        base = self.dir_yield_temp / lav_s / proj_s
        if not base.exists():
            return []

        arquivos = []
        arquivos += sorted(base.glob("*.gpkg"))
        arquivos += sorted(base.glob("*.parquet"))
        return arquivos

    def _ler_preproc(self, path: Path) -> gpd.GeoDataFrame:
        suf = path.suffix.lower()
        if suf == ".gpkg":
            return gpd.read_file(path, layer="yield_preproc")
        if suf == ".parquet":
            return gpd.read_parquet(path)
        raise ValueError(f"Formato de arquivo pré-processado não suportado: {path.name}")

    def _carregar_yield_temp(
        self,
        nome_lavoura: str,
        projeto_colheita: str,
        proc_lav: ProcessadorLavoura,
    ) -> Tuple[bool, str, Optional[gpd.GeoDataFrame]]:
        arquivos = self._listar_arquivos_preproc(nome_lavoura, projeto_colheita)
        lav_s = self._sanitizar_nome(nome_lavoura)
        proj_s = self._sanitizar_nome(projeto_colheita)
        pasta = self.dir_yield_temp / lav_s / proj_s

        if not arquivos:
            msg = (
                f"Nenhum arquivo pré-processado encontrado em: {pasta}. "
                "Rode primeiro o módulo de pré-processamento (filtro + balança)."
            )
            self._log(NivelLog.ERROR, "carregar_yield_temp", msg, mostrar_usuario=True)
            return False, msg, None

        self._log(
            NivelLog.INFO,
            "carregar_yield_temp",
            f"{len(arquivos)} arquivo(s) pré-processado(s) encontrados em: {pasta}",
            mostrar_usuario=True,
        )

        gdfs = []
        for i, arq in enumerate(arquivos, start=1):
            self._log(
                NivelLog.INFO,
                "carregar_yield_temp",
                f"({i}/{len(arquivos)}) Lendo: {arq.name}",
                mostrar_usuario=True,
            )
            g = self._ler_preproc(arq)

            if g.crs is None:
                raise ValueError(f"Arquivo pré-processado sem CRS definido: {arq.name}")

            if g.crs != proc_lav.crs_utm:
                try:
                    g = g.to_crs(proc_lav.crs_utm)
                except Exception as e:
                    raise ValueError(
                        f"Falha ao reprojetar '{arq.name}' para CRS da lavoura ({proc_lav.crs_utm}): {e}"
                    )

            g["__origem_arquivo__"] = arq.name
            gdfs.append(g)

        gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=proc_lav.crs_utm)

        geom_union = proc_lav.contorno.geometry.unary_union
        dentro_mask = gdf_all.geometry.within(geom_union)
        n_total = len(gdf_all)
        n_dentro = int(dentro_mask.sum())
        n_fora = int(n_total - n_dentro)

        if n_fora > 0:
            self._log(
                NivelLog.WARNING,
                "carregar_yield_temp",
                (
                    f"Arquivos pré-processados: {n_total} pontos carregados. "
                    f"{n_fora} pontos fora do contorno foram descartados no CLIP defensivo."
                ),
                mostrar_usuario=True,
            )

        gdf_in = gdf_all[dentro_mask].copy()

        if len(gdf_in) == 0:
            msg = "Após CLIP defensivo, nenhum ponto pré-processado ficou dentro do contorno."
            self._log(NivelLog.ERROR, "carregar_yield_temp", msg, mostrar_usuario=True)
            return False, msg, None

        return True, "Yield pré-processado carregado e preparado.", gdf_in

    # ---------------------------------------------------------
    # 4) Interpolar
    # ---------------------------------------------------------

    def _interpolar_yield(
        self,
        nome_lavoura: str,
        rotulo: str,
        gdf_yield: gpd.GeoDataFrame,
        col_yield: str,
        proc_lav: ProcessadorLavoura,
        diretorio_saida: Path,
    ) -> Tuple[bool, str, Optional[Path]]:
        interp = InterpoladorAtributo(
            contorno_utm=proc_lav.contorno,
            crs_utm=proc_lav.crs_utm,
            logger=self.logger,
            pixel_size=self.pixel_size,
        )

        ok, msg, caminho_tif = interp.interpolar_atributo(
            gdf_utm=gdf_yield,
            value_col=col_yield,
            metodo="kriging",
            nome_lavoura=nome_lavoura,
            ano_amostragem=rotulo,
            diretorio_saida=diretorio_saida,
        )
        return ok, msg, caminho_tif

    # ---------------------------------------------------------
    # 5) Amostrar raster de yield na grade
    # ---------------------------------------------------------

    def _amostrar_yield_na_grade(
        self,
        proc_lav: ProcessadorLavoura,
        df_meta: Optional[pd.DataFrame],
        caminho_tif: Path,
        col_yield: str,
    ) -> pd.DataFrame:
        """
        Amostra o raster de yield na grade e monta um DataFrame com:

        - id_ponto
        - Lat, Long (EPSG:4326)
        - Data, Cliente, Fazenda, Talhão, Gleba (1 valor por campanha, se existir no df_meta)
        - Kg/Ha (coluna fixa de produtividade)
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

        if df_meta is not None:
            df["Data"] = self._valor_unico(df_meta, "Data", etapa="amostrar_yield", nome_logico="Data")
            df["Cliente"] = self._valor_unico(df_meta, "Cliente", etapa="amostrar_yield", nome_logico="Cliente")
            df["Fazenda"] = self._valor_unico(df_meta, "Fazenda", etapa="amostrar_yield", nome_logico="Fazenda")
            df["Talhão"] = self._valor_unico(df_meta, "Talhão", etapa="amostrar_yield", nome_logico="Talhão")
            df["Gleba"] = self._valor_unico(df_meta, "Gleba", etapa="amostrar_yield", nome_logico="Gleba")
        else:
            df["Data"] = None
            df["Cliente"] = None
            df["Fazenda"] = None
            df["Talhão"] = None
            df["Gleba"] = None

        with rasterio.open(caminho_tif) as src:
            sampled = list(src.sample(coords))
            vals = np.array([v[0] for v in sampled])

            nodata = src.nodata
            if nodata is not None:
                vals = np.where(vals == nodata, np.nan, vals)

            df[col_yield] = vals

        self._log(
            NivelLog.INFO,
            "amostrar_yield",
            f"Yield '{col_yield}' amostrado na grade ({len(df)} pontos).",
        )

        cols_base = ["id_ponto", "Data", "Cliente", "Fazenda", "Talhão", "Gleba", "Lat", "Long", col_yield]
        cols_restantes = [c for c in df.columns if c not in cols_base]
        return df[cols_base + cols_restantes]

    # ---------------------------------------------------------
    # 6) Exportar CSV final
    # ---------------------------------------------------------

    def _exportar_csv_yield(
        self,
        df: pd.DataFrame,
        nome_lavoura: str,
        rotulo: str,
        diretorio_saida: Path,
    ) -> Path:
        diretorio_saida.mkdir(parents=True, exist_ok=True)

        lav_sanit = self._sanitizar_nome(nome_lavoura)
        rot_sanit = self._sanitizar_nome(rotulo)

        nome_arquivo = f"{lav_sanit}_prod_{rot_sanit}_grid.csv"
        caminho_csv = diretorio_saida / nome_arquivo

        df.to_csv(caminho_csv, index=False)

        self._log(
            NivelLog.INFO,
            "exportar_csv_yield",
            f"Arquivo CSV de yield exportado: {caminho_csv}",
            mostrar_usuario=True,
        )
        return caminho_csv

    # ---------------------------------------------------------
    # Método público: interpolar a partir do staging
    # ---------------------------------------------------------

    def executar_interpolacao_yield(
        self,
        nome_lavoura: str,
        projeto_colheita: str,
    ) -> Tuple[bool, str, Optional[pd.DataFrame], Optional[Path]]:
        """
        Interpola yield a partir dos arquivos pré-processados (módulo A),
        lendo de:
            dir_yield_temp / {nome_lavoura} / {projeto_colheita} / *.gpkg|*.parquet

        E exporta em subpastas:
            dir_rasters  / {nome_lavoura} / {projeto_colheita}/
            dir_csv_base / {nome_lavoura} / {projeto_colheita}/
        """
        col_yield = "Kg/Ha"

        self.logger.adicionar_contexto(
            lavoura=nome_lavoura,
            processo="prod",
            pixel_size=self.pixel_size,
            projeto_colheita=projeto_colheita,
        )

        lav_s = self._sanitizar_nome(nome_lavoura)
        proj_s = self._sanitizar_nome(projeto_colheita)

        dir_rasters_out = self.dir_rasters / lav_s / proj_s
        dir_csv_out = self.dir_csv_base / lav_s / proj_s

        ok, msg_lav, proc_lav = self._preparar_lavoura(nome_lavoura)
        if not ok or proc_lav is None:
            return False, msg_lav, None, None

        ok, msg_yld, gdf_all = self._carregar_yield_temp(nome_lavoura, projeto_colheita, proc_lav)
        if not ok or gdf_all is None:
            return False, msg_yld, None, None

        if col_yield not in gdf_all.columns:
            msg = f"Coluna de produtividade '{col_yield}' não encontrada nos arquivos pré-processados."
            self._log(NivelLog.ERROR, "executar_interpolacao_yield", msg, mostrar_usuario=True)
            return False, msg, None, None

        serie = gdf_all[col_yield]
        if serie.dtype == "O":
            s = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
            gdf_all[col_yield] = pd.to_numeric(s, errors="coerce")

        rotulo = projeto_colheita

        ok, msg_interp, caminho_tif = self._interpolar_yield(
            nome_lavoura=nome_lavoura,
            rotulo=rotulo,
            gdf_yield=gdf_all,
            col_yield=col_yield,
            proc_lav=proc_lav,
            diretorio_saida=dir_rasters_out,
        )
        if not ok or caminho_tif is None:
            return False, msg_interp, None, None

        df_final = self._amostrar_yield_na_grade(
            proc_lav=proc_lav,
            df_meta=gdf_all,
            caminho_tif=caminho_tif,
            col_yield=col_yield,
        )

        caminho_csv = self._exportar_csv_yield(
            df=df_final,
            nome_lavoura=nome_lavoura,
            rotulo=rotulo,
            diretorio_saida=dir_csv_out,
        )

        msg_final = (
            f"Interpolação de yield concluída para lavoura '{nome_lavoura}', "
            f"projeto '{projeto_colheita}'. Pontos na grade: {len(df_final)}."
        )
        self._log(NivelLog.INFO, "executar_interpolacao_yield", msg_final, mostrar_usuario=True)
        return True, msg_final, df_final, caminho_csv

    # ---------------------------------------------------------
    # Método público: pipeline yield
    # ---------------------------------------------------------

    def executar_pipeline_yield(
        self,
        nome_lavoura: str,
        nome_arquivo_csv: str,
    ) -> Tuple[bool, str, Optional[pd.DataFrame], Optional[Path]]:
        """
        Fluxo legado: recebe um CSV bruto em dir_yield.
        Mantido para compatibilidade.
        """
        col_yield = "Kg/Ha"

        self.logger.adicionar_contexto(
            lavoura=nome_lavoura,
            processo="prod",
            pixel_size=self.pixel_size,
        )

        ok, msg_lav, proc_lav = self._preparar_lavoura(nome_lavoura)
        if not ok or proc_lav is None:
            return False, msg_lav, None, None

        ok, msg_yld, gdf_utm, proc_am = self._carregar_yield_legado(
            nome_lavoura,
            nome_arquivo_csv,
            proc_lav,
        )
        if not ok or gdf_utm is None or proc_am is None:
            return False, msg_yld, None, None

        if col_yield not in gdf_utm.columns:
            msg = f"Coluna de produtividade '{col_yield}' não encontrada no arquivo."
            self._log(NivelLog.ERROR, "executar_pipeline_yield", msg, mostrar_usuario=True)
            return False, msg, None, None

        serie = gdf_utm[col_yield]
        if serie.dtype == "O":
            s = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
            gdf_utm[col_yield] = pd.to_numeric(s, errors="coerce")

        ok, msg_filt, gdf_filtrado = self._filtrar_yield_legado(gdf_utm, col_yield)
        if not ok or gdf_filtrado is None:
            return False, msg_filt, None, None

        rotulo = self._valor_unico(proc_am.df_raw, getattr(proc_am, "col_data", None), "rotulo_yield", "Data") or "campanha"

        ok, msg_interp, caminho_tif = self._interpolar_yield(
            nome_lavoura=nome_lavoura,
            rotulo=rotulo,
            gdf_yield=gdf_filtrado,
            col_yield=col_yield,
            proc_lav=proc_lav,
            diretorio_saida=self.dir_rasters,
        )
        if not ok or caminho_tif is None:
            return False, msg_interp, None, None

        df_final = self._amostrar_yield_na_grade(proc_lav, proc_am.df_raw, caminho_tif, col_yield)

        caminho_csv = self._exportar_csv_yield(df_final, nome_lavoura, rotulo, self.dir_csv_base / "csv_prod")

        msg_final = (
            f"Pipeline de produção concluído para lavoura '{nome_lavoura}', "
            f"campanha '{rotulo}'. Pontos: {len(df_final)}."
        )
        self._log(NivelLog.INFO, "executar_pipeline_yield", msg_final, mostrar_usuario=True)
        return True, msg_final, df_final, caminho_csv

    # ------------------------
    # Helpers legado
    # ------------------------

    def _carregar_yield_legado(
        self,
        nome_lavoura: str,
        nome_arquivo_csv: str,
        proc_lav: ProcessadorLavoura,
    ) -> Tuple[bool, str, Optional[gpd.GeoDataFrame], Optional[ProcessadorAmostragem]]:
        caminho_csv = self.dir_yield / nome_arquivo_csv

        proc_am = ProcessadorAmostragem(
            nome_lavoura=nome_lavoura,
            caminho_csv=caminho_csv,
            contorno_utm=proc_lav.contorno,
            crs_utm=proc_lav.crs_utm,
            logger=self.logger,
        )

        ok, msg = proc_am.carregar_csv()
        if not ok:
            return False, msg, None, None

        if proc_am.df_raw is not None:
            colmap = {
                "col_data": "Data",
                "col_cliente": "Cliente",
                "col_fazenda": "Fazenda",
                "col_talhao": "Talhão",
                "col_gleba": "Gleba",
            }
            for attr, colname in colmap.items():
                setattr(proc_am, attr, colname if colname in proc_am.df_raw.columns else None)

        ok, msg = proc_am.detectar_lat_lon()
        if not ok:
            return False, msg, None, None

        ok, msg = proc_am.criar_gdf_4326()
        if not ok:
            return False, msg, None, None

        ok, msg = proc_am.reprojetar_para_utm()
        if not ok:
            return False, msg, None, None

        gdf_utm = proc_am.gdf_utm.copy()

        geom_union = proc_lav.contorno.geometry.unary_union
        dentro_mask = gdf_utm.geometry.within(geom_union)
        n_total = len(gdf_utm)
        n_dentro = int(dentro_mask.sum())
        n_fora = int(n_total - n_dentro)

        self._log(
            NivelLog.INFO,
            "clip_yield",
            (
                f"Yield CSV: {n_total} pontos carregados. "
                f"{n_fora} pontos fora do contorno "
                f"({(n_fora/n_total*100 if n_total > 0 else 0):.1f}%). "
                f"{n_dentro} pontos mantidos dentro da lavoura."
            ),
            mostrar_usuario=True,
        )

        gdf_in = gdf_utm[dentro_mask].copy()

        if n_dentro == 0:
            msg = (
                "Nenhum ponto de yield está dentro do contorno da lavoura após o CLIP. "
                "Verifique se o contorno correto foi associado ou se o arquivo de produtividade "
                "está na mesma área."
            )
            self._log(NivelLog.ERROR, "clip_yield", msg, mostrar_usuario=True)
            return False, msg, None, None

        return True, "Yield carregado, reprojetado e clipado com sucesso.", gdf_in, proc_am

    def _filtrar_yield_legado(
        self,
        gdf_utm: gpd.GeoDataFrame,
        col_yield: str,
    ) -> Tuple[bool, str, Optional[gpd.GeoDataFrame]]:
        filtro = FiltroYield(logger=self.logger, config=self.config_filtro)
        gdf_filtrado, stats = filtro.filtrar(gdf_utm, col_yield)

        n_final = stats.get("n_final", 0)
        if n_final <= 0:
            msg = "Nenhum ponto restante após filtragem de yield."
            self._log(NivelLog.ERROR, "filtrar_yield", msg, mostrar_usuario=True)
            return False, msg, None

        msg = (
            f"Filtragem de yield concluída. Pontos finais: {n_final}. "
            f"RAW→GLOBAL→LOCAL: "
            f"{stats['raw']['count']:.0f} → "
            f"{(stats['global']['count'] if stats['global'] else stats['raw']['count']):.0f} → "
            f"{(stats['local']['count'] if stats['local'] else stats['global']['count'] if stats['global'] else stats['raw']['count']):.0f}"
        )
        self._log(NivelLog.INFO, "filtrar_yield", msg, mostrar_usuario=True)
        return True, msg, gdf_filtrado
