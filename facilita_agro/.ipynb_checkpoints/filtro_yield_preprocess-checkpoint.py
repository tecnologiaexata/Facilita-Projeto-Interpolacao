"""
filtro_yield.py - Filtragem automática de mapas de produtividade (yield)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from .logger import LoggerAgricola, NivelLog


@dataclass
class ConfigFiltroYield:
    """
    Configuração dos filtros de yield.

    Todos os percentuais são frações (0.5 = 50%).
    """
    aplicar_global: bool = True
    aplicar_local: bool = True

    global_pct: float = 0.5      # variação global ±50% da mediana
    local_pct: float = 0.3       # variação local ±30% da mediana local
    raio_local_m: float = 30.0   # raio em metros para filtragem local

    # Mínimo de pontos após filtragem para considerar válido
    min_pontos_pos_filtro: int = 30


class FiltroYield:
    """
    Aplica filtragem global e local em dados de produtividade (yield) em UTM.

    Fluxo:
      - filtragem global (opcional): remove valores fora de [med*(1±global_pct)]
      - filtragem local (opcional): para cada ponto, avalia vizinhança em raio
        'raio_local_m' e remove outliers locais fora de [med_loc*(1±local_pct)]
    """

    def __init__(
        self,
        logger: Optional[LoggerAgricola] = None,
        config: Optional[ConfigFiltroYield] = None,
    ) -> None:
        self.logger = logger
        self.config = config or ConfigFiltroYield()

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        if self.logger is not None:
            self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)
        else:
            print(f"[{nivel.value}] {etapa}: {mensagem}")

    # ------------------------------------------------------------------
    # Filtros
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_global(gdf: gpd.GeoDataFrame, value_col: str, pct: float) -> gpd.GeoDataFrame:
        """
        Mantém valores dentro de mediana * (1 ± pct).
        """
        med = gdf[value_col].median()
        lower, upper = med * (1 - pct), med * (1 + pct)
        return gdf[(gdf[value_col] >= lower) & (gdf[value_col] <= upper)].copy()

    @staticmethod
    def _filter_local_by_radius(
        gdf: gpd.GeoDataFrame,
        value_col: str,
        local_pct: float,
        radius: float,
    ) -> gpd.GeoDataFrame:
        """
        Para cada ponto, avalia vizinhança dentro de 'radius' e mantém apenas
        aqueles cujo valor está dentro de med_loc * (1 ± local_pct).
        """
        coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
        tree = BallTree(coords, leaf_size=20)
        idxs = tree.query_radius(coords, r=radius)

        mask = []
        for i, neighbors in enumerate(idxs):
            vals = gdf.iloc[neighbors][value_col].values
            med_loc = np.median(vals)
            lower, upper = med_loc * (1 - local_pct), med_loc * (1 + local_pct)
            mask.append(lower <= gdf.iloc[i][value_col] <= upper)

        return gdf[mask].copy()

    @staticmethod
    def _compute_stats(gdf: gpd.GeoDataFrame, value_col: str) -> Dict[str, float]:
        """
        Estatísticas de apoio para log/debug.
        """
        vals = gdf[value_col].dropna().values
        if vals.size == 0:
            return {
                "count": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "cv": np.nan,
            }

        stats = {
            "count": float(len(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        }
        stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else np.nan
        return stats

    def _corrigir_por_balanca(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
    ) -> Tuple[gpd.GeoDataFrame, Optional[Dict[str, float]]]:
        """
        Ajusta a coluna de yield (value_col) para que sua média bata com a média
        da balança, usando a coluna padrão 'Média de produção Kg/Ha (Balança)'.

        Se a coluna de balança não existir ou as médias forem inválidas,
        não aplica ajuste e apenas retorna o gdf original.
        """
        col_bal = "Média de produção Kg/Ha (Balança)"

        if col_bal not in gdf.columns:
            self._log(
                NivelLog.INFO,
                "filtro_yield",
                f"Coluna '{col_bal}' não encontrada. Ajuste por balança não será aplicado.",
            )
            return gdf, None

        serie_y = gdf[value_col]
        serie_bal = gdf[col_bal]

        def _to_numeric(s: pd.Series) -> np.ndarray:
            if s.dtype == "O":
                s = (
                    s.astype(str)
                     .str.strip()
                     .str.replace(",", ".", regex=False)
                )
            return pd.to_numeric(s, errors="coerce").values

        y_vals = _to_numeric(serie_y)
        bal_vals = _to_numeric(serie_bal)

        mask_y = np.isfinite(y_vals)
        if mask_y.sum() == 0:
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                "Não há valores numéricos válidos em yield após filtragem. "
                "Ajuste por balança não aplicado.",
                mostrar_usuario=True,
            )
            return gdf, None

        mean_monitor = float(y_vals[mask_y].mean())

        bal_valid = bal_vals[np.isfinite(bal_vals)]
        if bal_valid.size == 0:
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                "Coluna de balança não possui valores numéricos válidos. "
                "Ajuste por balança não aplicado.",
                mostrar_usuario=True,
            )
            return gdf, None

        mean_balanca = float(bal_valid.mean())

        if mean_monitor <= 0 or mean_balanca <= 0:
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                f"Médias inválidas para ajuste por balança "
                f"(monitor={mean_monitor:.3f}, balança={mean_balanca:.3f}). "
                "Ajuste não aplicado.",
                mostrar_usuario=True,
            )
            return gdf, None

        fator = mean_balanca / mean_monitor

        gdf_corrigido = gdf.copy()
        y_corr = np.where(np.isfinite(y_vals), y_vals * fator, np.nan)
        gdf_corrigido[value_col] = y_corr

        self._log(
            NivelLog.INFO,
            "filtro_yield",
            f"Ajuste por balança aplicado. "
            f"Média monitor (pós-filtro) = {mean_monitor:.2f}, "
            f"Média balança = {mean_balanca:.2f}, "
            f"fator = {fator:.4f}.",
            mostrar_usuario=True,
        )

        info = {
            "mean_monitor": mean_monitor,
            "mean_balanca": mean_balanca,
            "fator": fator,
        }

        return gdf_corrigido, info


    # ------------------------------------------------------------------
    # Interface 
    # ------------------------------------------------------------------

    def filtrar(
        self,
        gdf_utm: gpd.GeoDataFrame,
        value_col: str,
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Aplica filtragem global e local sobre gdf_utm[value_col], e em seguida
        ajusta a média do yield para bater com a média da balança (se disponível).

        Pré-condição:
          - gdf_utm deve estar em CRS projetado em metros (ex: UTM).
        """
        if gdf_utm.crs is None or not gdf_utm.crs.is_projected:
            raise ValueError("FiltroYield espera um GeoDataFrame em CRS projetado (UTM, metros).")

        if value_col not in gdf_utm.columns:
            raise ValueError(f"Coluna de valor '{value_col}' não encontrada no GeoDataFrame.")

        cfg = self.config

        raw_stats = self._compute_stats(gdf_utm, value_col)

        gdf_glob = gdf_utm.copy()
        glob_stats = raw_stats
        if cfg.aplicar_global:
            gdf_glob = self._filter_global(
                gdf_utm,
                value_col,
                cfg.global_pct,
            )
            glob_stats = self._compute_stats(gdf_glob, value_col)
            self._log(
                NivelLog.INFO,
                "filtro_yield",
                f"Filtragem GLOBAL aplicada (±{cfg.global_pct*100:.0f}%). "
                f"Pontos restantes: {glob_stats['count']:.0f}",
            )

        gdf_loc = gdf_glob.copy()
        loc_stats = glob_stats
        if cfg.aplicar_local:
            gdf_loc = self._filter_local_by_radius(
                gdf_glob,
                value_col,
                cfg.local_pct,
                cfg.raio_local_m,
            )
            loc_stats = self._compute_stats(gdf_loc, value_col)
            self._log(
                NivelLog.INFO,
                "filtro_yield",
                f"Filtragem LOCAL aplicada (raio={cfg.raio_local_m}m, "
                f"±{cfg.local_pct*100:.0f}%). Pontos restantes: {loc_stats['count']:.0f}",
            )

        if loc_stats is not None:
            n_final = int(loc_stats["count"])
        elif glob_stats is not None:
            n_final = int(glob_stats["count"])
        else:
            n_final = int(raw_stats["count"])

        if n_final < cfg.min_pontos_pos_filtro:
            msg = (
                f"Apenas {n_final} pontos restantes após filtragem de yield. "
                f"Mínimo configurado: {cfg.min_pontos_pos_filtro}. "
                "Considere ajustar parâmetros de filtro."
            )
            self._log(
                NivelLog.WARNING,
                "filtro_yield",
                msg,
                mostrar_usuario=True,
            )

        resumo_stats = {
            "raw": raw_stats,
            "global": glob_stats,
            "local": loc_stats,
            "n_final": n_final,
        }

        gdf_corrigido, info_bal = self._corrigir_por_balanca(gdf_loc, value_col)
        if info_bal is not None:
            resumo_stats["ajuste_balanca"] = info_bal

        return gdf_corrigido, resumo_stats


# ======================================================================
# Pré-processamento (CSV -> UTM -> filtros -> correção balança -> export)
# ======================================================================

@dataclass
class ConfigPreprocessamentoYield:
    """Configuração do módulo A (pré-processamento) para staging de yield."""

    dir_yield_temp: Path = Path("outputs") / "yield_temp"

    col_yield: str = "Kg/Ha"

    col_balanca: str = "Média de produção Kg/Ha (Balança)"

    formato: str = "gpkg"  # "gpkg" | "parquet"


class PreprocessadorYield:
    """
    Módulo A: lê um CSV de yield, reprojeta para UTM, aplica filtros + correção por balança
    e exporta o resultado para um diretório temporário (staging).

    Saída padrão:
        {dir_yield_temp}/{nome_lavoura}/{projeto_colheita}/preproc_{arquivo}.gpkg
        {dir_yield_temp}/{nome_lavoura}/{projeto_colheita}/preproc_{arquivo}.json   (stats)
    """

    def __init__(
        self,
        logger: Optional[LoggerAgricola] = None,
        config_filtro: Optional[ConfigFiltroYield] = None,
        config_preprocess: Optional[ConfigPreprocessamentoYield] = None,
    ) -> None:
        self.logger = logger
        self.cfg_pre = config_preprocess or ConfigPreprocessamentoYield()
        self.filtro = FiltroYield(logger=logger, config=config_filtro)

    # ---------------------------
    # Utilitários 
    # ---------------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        if self.logger is not None:
            self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)
        else:
            print(f"[{nivel.value}] {etapa}: {mensagem}")

    @staticmethod
    def _sanitizar_nome(texto: str) -> str:
        s = unicodedata.normalize("NFKD", str(texto))
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "sem_nome"

    @staticmethod
    def _inferir_epsg_utm(lon: float, lat: float) -> int:
        """Inferência simples de EPSG UTM a partir de lon/lat (WGS84)."""
        zona = int((lon + 180.0) // 6.0) + 1
        if lat >= 0:
            return 32600 + zona
        return 32700 + zona

    @staticmethod
    def _to_numeric_series(s: pd.Series) -> pd.Series:
        if s.dtype == "O":
            s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    # ---------------------------
    # CSV -> GeoDataFrame 4326
    # ---------------------------

    def _ler_csv_robusto(self, caminho_csv: Path) -> pd.DataFrame:
        """Tenta ler CSV com fallbacks de encoding comuns em exportações de máquina."""
        tentativas = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err: Optional[Exception] = None
        for enc in tentativas:
            try:
                return pd.read_csv(caminho_csv, encoding=enc)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Falha ao ler CSV com encodings comuns: {last_err}")

    def _detectar_lat_lon(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Detecta nomes de colunas de lat/lon (ou x/y) por heurística."""
        cols = list(df.columns)

        cand_lat = ["Lat", "Latitude", "LAT", "lat", "latitude", "Y", "y"]
        cand_lon = ["Long", "Lon", "Longitude", "LONG", "lon", "longitude", "X", "x"]

        col_lat = next((c for c in cand_lat if c in cols), None)
        col_lon = next((c for c in cand_lon if c in cols), None)

        if col_lat is None or col_lon is None:
            raise ValueError(
                "Não consegui detectar colunas de coordenadas. "
                "Esperava algo como Lat/Long (graus) ou X/Y."
            )

        return col_lat, col_lon

    def _df_para_gdf_4326(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        col_lat, col_lon = self._detectar_lat_lon(df)

        lat = self._to_numeric_series(df[col_lat])
        lon = self._to_numeric_series(df[col_lon])

        lat_med = float(np.nanmedian(lat.values))
        lon_med = float(np.nanmedian(lon.values))

        if (abs(lat_med) > 90 and abs(lon_med) <= 90) or (lat_med < -90 or lat_med > 90):
            lat, lon = lon, lat
            self._log(
                NivelLog.WARNING,
                "coords",
                "Colunas parecem invertidas; trocando (lat<->lon).",
                dados={"lat_med": lat_med, "lon_med": lon_med},
                mostrar_usuario=True,
            )

        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(lon, lat), crs=4326)

        gdf = gdf[gdf.geometry.notna()].copy()

        if len(gdf) == 0:
            raise ValueError("Nenhum ponto válido foi gerado a partir das coordenadas.")

        return gdf

    # ---------------------------
    # Fazenda: coluna ou arquivo
    # ---------------------------

    def _inferir_fazenda(self, df: pd.DataFrame, caminho_csv: Path) -> str:
        """
        Regra:
        - Se existir coluna 'Fazenda' com algum valor, usa o 1º valor não-nulo.
        - Senão, tenta inferir do nome do arquivo (stem), pegando o 1º token.
        """
        if "Fazenda" in df.columns:
            serie = df["Fazenda"].dropna().astype(str)
            if not serie.empty:
                return str(serie.iloc[0]).strip()

        stem = caminho_csv.stem
        tokens = re.split(r"[\s\-\_]+", stem)
        tokens = [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]
        return tokens[0] if tokens else "fazenda"

    # ---------------------------
    # Export (gpkg/parquet) + stats
    # ---------------------------

    def _exportar(
        self,
        gdf_utm: gpd.GeoDataFrame,
        stats: Dict[str, Any],
        nome_lavoura: str,
        projeto_colheita: str,
        caminho_csv: Path,
    ) -> Path:
        lavoura_s = self._sanitizar_nome(nome_lavoura)
        projeto_s = self._sanitizar_nome(projeto_colheita)
    
        out_dir = Path(self.cfg_pre.dir_yield_temp) / lavoura_s / projeto_s
        out_dir.mkdir(parents=True, exist_ok=True)
    
        base = f"preproc_{self._sanitizar_nome(caminho_csv.stem)}"
        formato = (self.cfg_pre.formato or "gpkg").lower().strip()
    
        if gdf_utm.crs is None or not gdf_utm.crs.is_projected:
            raise ValueError("Saída do preprocessamento deveria estar em UTM (CRS projetado).")
    
        out_path: Path
    
        if formato == "parquet":
            out_path = out_dir / f"{base}.parquet"
            try:
                gdf_utm.to_parquet(out_path, index=False)
            except Exception as e:
                self._log(
                    NivelLog.WARNING,
                    "export_preproc",
                    f"Falha ao exportar Parquet ({e}). Fazendo fallback para GPKG.",
                    mostrar_usuario=True,
                )
                formato = "gpkg"
    
        if formato == "gpkg":
            out_path = out_dir / f"{base}.gpkg"
            gdf_utm.to_file(out_path, layer="yield_preproc", driver="GPKG")
    
        if formato not in ("gpkg", "parquet"):
            raise ValueError(f"Formato inválido para exportação: '{formato}'. Use 'gpkg' ou 'parquet'.")
    
        stats_out = out_dir / f"{base}.json"
        payload = {
            "lavoura": nome_lavoura,
            "projeto_colheita": projeto_colheita,
            "arquivo_origem": str(caminho_csv.name),
            "crs_epsg": int(gdf_utm.crs.to_epsg()) if gdf_utm.crs and gdf_utm.crs.to_epsg() else None,
            "criado_em": datetime.utcnow().isoformat() + "Z",
            "stats": stats,
        }
        stats_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
        self._log(
            NivelLog.INFO,
            "export_preproc",
            f"Pré-processamento exportado: {out_path}",
            mostrar_usuario=True,
        )
    
        return out_path


    # ---------------------------
    # API: CSV -> staging
    # ---------------------------

    def preprocessar_csv_para_staging(
        self,
        caminho_csv: str | Path,
        nome_lavoura: str,
        projeto_colheita: str,
        *,
        col_yield: Optional[str] = None,
        epsg_utm: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Lê um CSV de yield, converte para UTM, aplica filtro+balança, e exporta para:
            outputs/yield_temp/{nome_lavoura}/{projeto_colheita}/...

        Args:
            caminho_csv: caminho do CSV
            projeto_colheita: rótulo dado pelo usuário (ex.: soja_2025)
            col_yield: opcional; default "Kg/Ha"
            epsg_utm: opcional; se None, inferimos pelo lon/lat mediano (WGS84)

        Returns:
            (ok, msg, caminho_arquivo_exportado_ou_None)
        """
        try:
            caminho_csv = Path(caminho_csv)
            if not caminho_csv.exists():
                return False, f"CSV não encontrado: {caminho_csv}", None

            df = self._ler_csv_robusto(caminho_csv)

            col_val = col_yield or self.cfg_pre.col_yield
            if col_val not in df.columns:
                return False, f"Coluna de produtividade '{col_val}' não encontrada no CSV.", None

            gdf_4326 = self._df_para_gdf_4326(df)

            if epsg_utm is None:
                lon_med = float(np.nanmedian(gdf_4326.geometry.x.values))
                lat_med = float(np.nanmedian(gdf_4326.geometry.y.values))
                epsg_utm = self._inferir_epsg_utm(lon_med, lat_med)

            gdf_utm = gdf_4326.to_crs(epsg_utm)

            gdf_utm[col_val] = self._to_numeric_series(gdf_utm[col_val])

            gdf_f, stats = self.filtro.filtrar(gdf_utm, col_val)

            out_path = self._exportar(gdf_f, stats, nome_lavoura, projeto_colheita, caminho_csv)
            return True, "Pré-processamento concluído com sucesso.", out_path

        except Exception as e:
            self._log(
                NivelLog.ERROR,
                "preprocessar_yield",
                f"Erro no pré-processamento: {e}",
                mostrar_usuario=True,
            )
            return False, f"Erro no pré-processamento: {e}", None


    def preprocessar_pasta_yield(
        self,
        *,
        dir_yield: str | Path,
        nome_lavoura: str,
        projeto_colheita: str,
        col_yield: Optional[str] = None,
        epsg_utm: Optional[int] = None,
        extensoes: Tuple[str, ...] = (".csv",),
    ) -> Tuple[bool, str, Optional[list[Path]]]:
        """
        Processa TODOS os arquivos de yield dentro de:

            {dir_yield}/{nome_lavoura}/{projeto_colheita}/

        Regras (plug-and-play):
          - se está na pasta e tem extensão válida, processa.
          - para cada arquivo: filtra + corrige pela balança + exporta para o staging:
                outputs/yield_temp/{nome_lavoura}/{projeto_colheita}/

        Retorna:
          (ok, msg, lista_de_paths_exportados)
        """
        base_in = Path(dir_yield) / nome_lavoura / projeto_colheita
        if not base_in.exists():
            msg = f"Diretório de entrada não encontrado: {base_in}"
            self._log(NivelLog.ERROR, "preprocessar_pasta_yield", msg, mostrar_usuario=True)
            return False, msg, None

        exts = {e.lower() for e in extensoes}
        arquivos = sorted([p for p in base_in.iterdir() if p.is_file() and p.suffix.lower() in exts])

        if not arquivos:
            msg = f"Nenhum arquivo com extensão {sorted(exts)} encontrado em: {base_in}"
            self._log(NivelLog.WARNING, "preprocessar_pasta_yield", msg, mostrar_usuario=True)
            return False, msg, []

        self._log(
            NivelLog.INFO,
            "preprocessar_pasta_yield",
            f"{len(arquivos)} arquivo(s) encontrados para pré-processar em: {base_in}",
            dados={"arquivos": [a.name for a in arquivos]},
            mostrar_usuario=True,
        )

        saidas: list[Path] = []
        for i, arq in enumerate(arquivos, start=1):
            self._log(
                NivelLog.INFO,
                "preprocessar_pasta_yield",
                f"({i}/{len(arquivos)}) Pré-processando: {arq.name}",
                mostrar_usuario=True,
            )
            ok, msg, out_path = self.preprocessar_csv_para_staging(
                caminho_csv=arq,
                nome_lavoura=nome_lavoura,
                projeto_colheita=projeto_colheita,
                col_yield=col_yield,
                epsg_utm=epsg_utm,
            )
            if not ok or out_path is None:
                # aborta no primeiro erro para não produzir staging parcial "silencioso"
                msg_err = f"Falha ao pré-processar '{arq.name}': {msg}"
                self._log(NivelLog.ERROR, "preprocessar_pasta_yield", msg_err, mostrar_usuario=True)
                return False, msg_err, saidas

            saidas.append(out_path)

        msg_ok = f"Pré-processamento concluído: {len(saidas)} arquivo(s) exportado(s) para staging."
        self._log(NivelLog.INFO, "preprocessar_pasta_yield", msg_ok, mostrar_usuario=True)
        return True, msg_ok, saidas
