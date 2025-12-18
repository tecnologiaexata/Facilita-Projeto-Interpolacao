"""
processador_amostragem.py - Ingestão e validação de arquivos de amostragem (CSV)

Responsável por:
  - importar CSV de amostragem
  - validar campos comuns da plataforma (Data, Cliente, Fazenda, Talhão, Gleba)
  - detectar colunas de lat/long (em graus decimais, EPSG:4326)
  - criar GeoDataFrame em 4326 e reprojetar para UTM
  - checar se os pontos estão dentro do contorno da lavoura
  - identificar colunas de atributos disponíveis (genérico)
    * A filtragem por processo (quais interpolar) é feita no pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import unicodedata

import pandas as pd
import geopandas as gpd

from .logger import LoggerAgricola, NivelLog


# Nomes possíveis para latitude e longitude no CSV
LAT_CANDIDATOS = [
    "latitude", "Latitude", "LATITUDE",
    "lat", "Lat", "LAT",
    "y", "Y",
    "Lat",  
]

LON_CANDIDATOS = [
    "longitude", "Longitude", "LONGITUDE",
    "long", "Long", "LONG",
    "x", "X",
    "Long",  
]

# Possíveis nomes de coluna de ID
ID_CANDIDATOS = [
    "id", "Id", "ID",
    "id_ponto", "ID_PONTO", "idPonto", "Ponto"
]


CAMPOS_COMUNS_PLATAFORMA = [
    "Data",
    "Cliente",
    "Fazenda",
    "Talhão",
    "Gleba",
]

# Campos que podem aparecer como numéricos ou textos, mas que
# NÃO devem ser interpolados (IDs, profundidade, datas etc.)
CAMPOS_NAO_INTERPOLAVEIS = [
    # Solo / Compactação
    "Profundidade",
    "Ponto",

    # Produção
    "Cultura",
    "Variedade",
    "Data de plantio",
    "Data de colheita"

    # Nematoide
    "Espécie",
]


class ProcessadorAmostragem:
    """
    Processa um arquivo CSV de amostragem para uma lavoura específica.

    Etapas:
      1. carregar_csv
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
        caminho_csv: str | Path,
        contorno_utm: gpd.GeoDataFrame,
        crs_utm: str,
        logger: Optional[LoggerAgricola] = None,
    ) -> None:
        self.nome_lavoura = nome_lavoura
        self.caminho_csv = Path(caminho_csv)
        self.contorno_utm = contorno_utm
        self.crs_utm = crs_utm
        self.logger = logger

        self.df_raw: Optional[pd.DataFrame] = None
        self.gdf_4326: Optional[gpd.GeoDataFrame] = None
        self.gdf_utm: Optional[gpd.GeoDataFrame] = None

        self.col_lat: Optional[str] = None
        self.col_lon: Optional[str] = None
        self.col_id: Optional[str] = None

        self.col_data: Optional[str] = None
        self.col_cliente: Optional[str] = None
        self.col_fazenda: Optional[str] = None
        self.col_talhao: Optional[str] = None
        self.col_gleba: Optional[str] = None

        self.col_atributos: List[str] = []

    # ---------------------------------------------------------
    # Helper de log
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Helpers de coding
    # ---------------------------------------------------------
    
    def _normalize_nome_coluna(self, s: str) -> str:
        """
        Normaliza nome de coluna para comparação:
          - string -> str
          - strip
          - lower
          - remove acentos (NFKD)
        """
        if s is None:
            return ""
        s = str(s).strip()
        s_norm = unicodedata.normalize("NFKD", s)
        s_sem_acentos = "".join(ch for ch in s_norm if not unicodedata.combining(ch))
        return s_sem_acentos.lower()


    def _coagir_para_numerico(self, serie: pd.Series) -> pd.Series:
        """
        Tenta converter uma Series para numérico:

          - Se for object, converte para string e troca vírgula por ponto
          - Usa pd.to_numeric(errors='coerce') para derrubar valores inválidos para NaN

        NÃO deve ser usado em colunas de ID ou textos categóricos,
        apenas nas candidatas a atributos numéricos.
        """
        s = serie.copy()

        if s.dtype == "O":
            s = (
                s.astype(str)
                 .str.strip()
                 .str.replace(",", ".", regex=False)
            )

        s_num = pd.to_numeric(s, errors="coerce")
        return s_num

    # ---------------------------------------------------------
    # Helpers de ponto central
    # ---------------------------------------------------------
    def _criar_ponto_central_sem_coordenadas(self) -> Tuple[bool, str]:
        """
        Fluxo alternativo para arquivos SEM colunas de coordenadas.

        - Calcula o centróide do contorno UTM como ponto de amostragem
        - Atribui a esse ponto:
            * campos comuns (Data, Cliente, Fazenda, Talhão, Gleba)
              como um único valor representativo (modo / primeiro não-nulo)
            * colunas numéricas como média (ignorando NaN)
            * demais colunas como primeiro valor não-nulo ou None
        """

        if self.df_raw is None:
            msg = "DataFrame bruto não carregado. Execute carregar_csv() primeiro."
            self._log(
                NivelLog.ERROR,
                "ponto_central_sem_coords",
                msg,
                mostrar_usuario=True,
            )
            return False, msg

        if self.contorno_utm is None or self.contorno_utm.empty:
            msg = "Contorno UTM indisponível para gerar ponto central."
            self._log(
                NivelLog.ERROR,
                "ponto_central_sem_coords",
                msg,
                mostrar_usuario=True,
            )
            return False, msg

        try:
            geom_union = self.contorno_utm.geometry.unary_union
            centroid = geom_union.centroid
        except Exception as e:
            msg = f"Erro ao calcular centróide do contorno: {e}"
            self._log(
                NivelLog.ERROR,
                "ponto_central_sem_coords",
                msg,
                mostrar_usuario=True,
            )
            return False, msg

        df = self.df_raw

        campos_comuns = {
            c
            for c in [
                self.col_data,
                self.col_cliente,
                self.col_fazenda,
                self.col_talhao,
                self.col_gleba,
            ]
            if c is not None and c in df.columns
        }

        dados_linha: Dict[str, Any] = {}

        for col in df.columns:
            serie = df[col]

            if col in campos_comuns:
                serie_nonnull = serie.dropna()
                if serie_nonnull.empty:
                    valor = None
                else:
                    try:
                        modo = serie_nonnull.mode(dropna=True)
                        valor = modo.iloc[0] if not modo.empty else serie_nonnull.iloc[0]
                    except Exception:
                        valor = serie_nonnull.iloc[0]

                    if serie_nonnull.nunique(dropna=True) > 1:
                        self._log(
                            NivelLog.WARNING,
                            "ponto_central_sem_coords",
                            (
                                f"Campo '{col}' possui múltiplos valores; "
                                f"usando '{valor}' como representativo."
                            ),
                            dados={"valores_unicos": serie_nonnull.unique().tolist()},
                            mostrar_usuario=False,
                        )
            else:
                s_num = self._coagir_para_numerico(serie)
                if s_num.notna().any():
                    valor = float(s_num.mean(skipna=True))
                else:
                    serie_nonnull = serie.dropna()
                    valor = serie_nonnull.iloc[0] if not serie_nonnull.empty else None

            dados_linha[col] = valor

        df_um = pd.DataFrame([dados_linha])
        gdf = gpd.GeoDataFrame(df_um, geometry=[centroid], crs=self.crs_utm)

        self.gdf_utm = gdf

        msg = (
            "Arquivo sem coordenadas: criado ponto central no contorno da lavoura "
            "com médias dos campos numéricos."
        )
        self._log(
            NivelLog.WARNING,
            "ponto_central_sem_coords",
            msg,
            mostrar_usuario=True,
        )
        return True, msg



    # ---------------------------------------------------------
    # Etapa 1: carregar CSV
    # ---------------------------------------------------------

    def carregar_csv(self) -> Tuple[bool, str]:
        """Importa o CSV bruto de amostragem, tentando diferentes encodings."""
        if not self.caminho_csv.exists():
            msg = f"Arquivo CSV de amostragem não encontrado: {self.caminho_csv}"
            self._log(NivelLog.ERROR, "carregar_csv", msg, mostrar_usuario=True)
            return False, msg

        encodings_teste = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

        ultima_exc = None
        for enc in encodings_teste:
            try:
                self.df_raw = pd.read_csv(
                    self.caminho_csv,
                    encoding=enc,
                    sep=None,        # infere separador (',' ou ';')
                    engine="python", # necessário quando sep=None
                )
                msg = (
                    f"CSV carregado com sucesso com encoding '{enc}': "
                    f"{len(self.df_raw)} linhas, {len(self.df_raw.columns)} colunas"
                )
                self._log(
                    NivelLog.INFO,
                    "carregar_csv",
                    msg,
                    dados={"encoding_utilizado": enc},
                    mostrar_usuario=True,
                )
                return True, msg
            except UnicodeDecodeError as e:
                ultima_exc = e
                continue
            except Exception as e:
                ultima_exc = e
                continue

        msg = f"Erro ao ler CSV em todos os encodings testados: {ultima_exc}"
        self._log(NivelLog.ERROR, "carregar_csv", msg, mostrar_usuario=True)
        return False, msg


    # ---------------------------------------------------------
    # Etapa 2: detectar campos comuns da plataforma
    # ---------------------------------------------------------

    def _detectar_campos_comuns_plataforma(self) -> Tuple[bool, str]:
        """
        Detecta e valida as colunas comuns da plataforma:

          - 'Data', 'Talhão' são obrigatórias (case/acento-insensitive)
          - 'Cliente', 'Fazenda', 'Gleba' são opcionais

        As colunas reais (com o nome exato do CSV) são armazenadas em:
          self.col_data, self.col_talhao, self.col_cliente, self.col_fazenda, self.col_gleba
        """
        if self.df_raw is None:
            msg = "DataFrame bruto não carregado. Execute carregar_csv() primeiro."
            self._log(NivelLog.ERROR, "campos_comuns", msg, mostrar_usuario=True)
            return False, msg

        colunas_norm: Dict[str, str] = {
            self._normalize_nome_coluna(c): c for c in self.df_raw.columns
        }

        def achar(*nomes_alternativos: str) -> Optional[str]:
            """
            Tenta encontrar a coluna no df usando uma ou mais variantes do nome lógico,
            ex.: achar("Talhão", "Talhao")
            """
            for nome in nomes_alternativos:
                norm = self._normalize_nome_coluna(nome)
                if norm in colunas_norm:
                    return colunas_norm[norm]
            return None

        obrigatorios = {
            "Data": ("Data",),
            "Talhão": ("Talhão", "Talhao"),
        }
        opcionais = {
            "Cliente": ("Cliente",),
            "Fazenda": ("Fazenda",),
            "Gleba": ("Gleba",),
        }

        faltando_obrig = []
        mapping: Dict[str, Optional[str]] = {}

        for campo_logico, variantes in obrigatorios.items():
            real = achar(*variantes)
            mapping[campo_logico] = real
            if real is None:
                faltando_obrig.append(campo_logico)

        for campo_logico, variantes in opcionais.items():
            real = achar(*variantes)
            mapping[campo_logico] = real

        if faltando_obrig:
            msg = (
                "O arquivo não segue o padrão de colunas da plataforma. "
                f"Campos obrigatórios ausentes: {faltando_obrig}"
            )
            self._log(
                NivelLog.ERROR,
                "campos_comuns",
                msg,
                dados={"faltando": faltando_obrig, "colunas": list(self.df_raw.columns)},
                mostrar_usuario=True,
            )
            return False, msg

        self.col_data = mapping["Data"]
        self.col_talhao = mapping["Talhão"]
        self.col_cliente = mapping.get("Cliente")
        self.col_fazenda = mapping.get("Fazenda")
        self.col_gleba = mapping.get("Gleba")

        msg = (
            "Campos comuns detectados: "
            f"Data='{self.col_data}', Talhão='{self.col_talhao}', "
            f"Cliente='{self.col_cliente}', Fazenda='{self.col_fazenda}', "
            f"Gleba='{self.col_gleba}'."
        )
        self._log(NivelLog.INFO, "campos_comuns", msg, mostrar_usuario=True)
        return True, msg


    # ---------------------------------------------------------
    # Etapa 3: detectar lat/lon e ID
    # ---------------------------------------------------------

    def detectar_lat_lon(self) -> Tuple[bool, str]:
        if self.df_raw is None:
            msg = "DataFrame bruto não carregado. Execute carregar_csv() primeiro."
            self._log(
                NivelLog.ERROR,
                "detectar_lat_lon",
                msg,
                mostrar_usuario=True,
            )
            return False, msg

        df = self.df_raw

        candidatos_lat = ["lat", "latitude", "y", "coord_y", "utm_y", "northing"]
        candidatos_lon = ["lon", "long", "longitude", "x", "coord_x", "utm_x", "easting"]

        self.col_lat = None
        self.col_lon = None

        cols_lower = {c.lower(): c for c in df.columns}

        for c in candidatos_lat:
            if c in cols_lower:
                self.col_lat = cols_lower[c]
                break

        for c in candidatos_lon:
            if c in cols_lower:
                self.col_lon = cols_lower[c]
                break

        if self.col_lat is None or self.col_lon is None:
            msg = "Lat e Long não encontrados nas colunas do arquivo."
            self._log(
                NivelLog.WARNING,
                "detectar_lat_lon",
                msg,
                mostrar_usuario=True,
            )
            return False, msg

        s_lat = pd.to_numeric(df[self.col_lat], errors="coerce")
        s_lon = pd.to_numeric(df[self.col_lon], errors="coerce")
        valid = s_lat.notna() & s_lon.notna()
        n_valid = int(valid.sum())
        n_total = len(df)

        if n_valid == 0:
            msg_det = (
                f"Colunas de coordenadas detectadas ('{self.col_lat}', '{self.col_lon}'), "
                "mas todos os valores estão vazios ou inválidos."
            )
            self._log(
                NivelLog.WARNING,
                "detectar_lat_lon",
                msg_det,
                mostrar_usuario=True,
            )

            self.col_lat = None
            self.col_lon = None

            msg = "Lat e Long não encontrados (valores vazios ou inválidos)."
            return False, msg

        self._log(
            NivelLog.INFO,
            "detectar_lat_lon",
            (
                f"Colunas de coordenadas detectadas: lat='{self.col_lat}', lon='{self.col_lon}'. "
                f"{n_valid} pontos com valores válidos de {n_total} linhas."
            ),
            mostrar_usuario=True,
        )

        return True, "Colunas de coordenadas detectadas com sucesso."


    def _detectar_col_id(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detecta uma coluna de ID, se existir.
        Caso contrário, o índice do DataFrame será usado como ID fallback na checagem de pontos fora.
        """
        colunas_lower = {c.lower(): c for c in df.columns}
        for c in ID_CANDIDATOS:
            c_lower = c.lower()
            if c_lower in colunas_lower:
                return colunas_lower[c_lower]
        return None

    # ---------------------------------------------------------
    # Etapa 4: criar GeoDataFrame em EPSG:4326
    # ---------------------------------------------------------

    def criar_gdf_4326(self) -> Tuple[bool, str]:
        """Cria GeoDataFrame em EPSG:4326 a partir das colunas de lat/long em graus decimais."""
        if self.df_raw is None or self.col_lat is None or self.col_lon is None:
            msg = "Dados ou colunas de coordenadas não disponíveis. Execute passos anteriores."
            self._log(NivelLog.ERROR, "criar_gdf_4326", msg, mostrar_usuario=True)
            return False, msg

        try:
            gdf = gpd.GeoDataFrame(
                self.df_raw.copy(),
                geometry=gpd.points_from_xy(
                    self.df_raw[self.col_lon],
                    self.df_raw[self.col_lat],
                ),
                crs="EPSG:4326",
            )
            self.gdf_4326 = gdf

            msg = "GeoDataFrame em EPSG:4326 criado com sucesso."
            self._log(NivelLog.INFO, "criar_gdf_4326", msg)
            return True, msg
        except Exception as e:
            msg = f"Erro ao criar GeoDataFrame 4326: {e}"
            self._log(NivelLog.ERROR, "criar_gdf_4326", msg, mostrar_usuario=True)
            return False, msg

    # ---------------------------------------------------------
    # Etapa 5: reprojetar para UTM
    # ---------------------------------------------------------

    def reprojetar_para_utm(self) -> Tuple[bool, str]:
        """Reprojeta gdf_4326 para o CRS UTM da lavoura."""
        if self.gdf_4326 is None:
            msg = "gdf_4326 não disponível. Execute criar_gdf_4326() primeiro."
            self._log(NivelLog.ERROR, "reprojetar_para_utm", msg, mostrar_usuario=True)
            return False, msg

        try:
            gdf_utm = self.gdf_4326.to_crs(self.crs_utm)
            self.gdf_utm = gdf_utm

            msg = f"GeoDataFrame reprojetado para {self.crs_utm}."
            self._log(NivelLog.INFO, "reprojetar_para_utm", msg)
            return True, msg
        except Exception as e:
            msg = f"Erro ao reprojetar para UTM ({self.crs_utm}): {e}"
            self._log(NivelLog.ERROR, "reprojetar_para_utm", msg, mostrar_usuario=True)
            return False, msg

    # ---------------------------------------------------------
    # Etapa 6: checar se os pontos estão dentro do contorno
    # ---------------------------------------------------------

    def checar_pontos_dentro_contorno(self) -> Tuple[bool, str]:
        """
        Checa se os pontos de amostragem estão dentro do contorno da lavoura.

        Se houver pontos fora, loga os IDs (se existir coluna de ID)
        e interrompe o processamento.
        """
        if self.gdf_utm is None:
            msg = "gdf_utm não disponível. Execute reprojetar_para_utm() primeiro."
            self._log(NivelLog.ERROR, "checar_pontos_contorno", msg, mostrar_usuario=True)
            return False, msg

        geom_union = self.contorno_utm.geometry.unary_union

        dentro_mask = self.gdf_utm.geometry.within(geom_union)
        n_total = len(self.gdf_utm)
        n_dentro = dentro_mask.sum()
        n_fora = n_total - n_dentro

        if n_fora == 0:
            msg = f"Todos os {n_total} pontos estão dentro do contorno da lavoura."
            self._log(NivelLog.INFO, "checar_pontos_contorno", msg, mostrar_usuario=True)
            return True, msg

        df_fora = self.gdf_utm[~dentro_mask].copy()

        if self.col_id is not None and self.col_id in df_fora.columns:
            ids_fora = df_fora[self.col_id].tolist()
        else:
            ids_fora = df_fora.index.tolist()

        msg_usuario = (
            f"{n_fora} pontos estão fora do contorno da lavoura. "
            "Checar os dados e importar novamente."
        )
        msg_tecnico = (
            f"{n_fora} pontos fora de {n_total} no total. "
            f"Ids dos pontos fora: {ids_fora}"
        )

        self._log(
            NivelLog.ERROR,
            "checar_pontos_contorno",
            msg_tecnico,
            dados={"ids_fora": ids_fora},
            mostrar_usuario=True,
        )

        return False, msg_usuario

    # ---------------------------------------------------------
    # Etapa 7: identificar atributos genéricos
    # ---------------------------------------------------------

    def identificar_atributos(self) -> Tuple[bool, str]:
        """
        Identifica colunas de atributos disponíveis para uso posterior.

        Aqui fazemos a filtragem GENÉRICA:
          - removemos colunas de coordenadas (lat, lon)
          - removemos ID
          - removemos geometry
          - removemos campos comuns da plataforma (Data, Cliente, Fazenda, Talhão, Gleba)
          - removemos campos explicitamente NÃO interpoláveis
          - garantimos que as colunas restantes sejam numéricas
            (vírgula decimal -> ponto, to_numeric)

        A filtragem fina (quais de fato entrarão na interpolação, dependendo
        do processo: solo, foliar, etc.) continua sendo feita no pipeline,
        usando COLS_INTERPOLAVEIS para cada processo.
        """
        if self.gdf_utm is None:
            msg = "gdf_utm não disponível. Execute reprojetar_para_utm() primeiro."
            self._log(NivelLog.ERROR, "identificar_atributos", msg, mostrar_usuario=True)
            return False, msg

        cols_excluir = {
            self.col_lat,
            self.col_lon,
            self.col_id,
            "geometry",
        }

        for campo in CAMPOS_COMUNS_PLATAFORMA:
            if campo in self.gdf_utm.columns:
                cols_excluir.add(campo)

        for campo in CAMPOS_NAO_INTERPOLAVEIS:
            if campo in self.gdf_utm.columns:
                cols_excluir.add(campo)

        cols_excluir = {c for c in cols_excluir if c is not None}

        atributos_brutos = [c for c in self.gdf_utm.columns if c not in cols_excluir]

        if not atributos_brutos:
            msg = "Nenhuma coluna candidata a atributo encontrada após filtragem genérica."
            self._log(NivelLog.WARNING, "identificar_atributos", msg, mostrar_usuario=True)
            return False, msg

        atributos_finais: list[str] = []
        for col in atributos_brutos:
            serie_original = self.gdf_utm[col]
            serie_num = self._coagir_para_numerico(serie_original)

            if serie_num.notna().sum() == 0:
                self._log(
                    NivelLog.DEBUG,
                    "identificar_atributos",
                    f"Coluna '{col}' descartada: não pôde ser convertida para numérico.",
                    mostrar_usuario=False,
                )
                continue

            self.gdf_utm[col] = serie_num
            atributos_finais.append(col)

        if not atributos_finais:
            msg = (
                "Nenhuma coluna de atributo numérico encontrada após coerção. "
                "Verifique se o CSV está com separador decimal correto."
            )
            self._log(NivelLog.ERROR, "identificar_atributos", msg, mostrar_usuario=True)
            return False, msg

        self.col_atributos = atributos_finais

        msg = f"Atributos numéricos disponíveis após filtragem genérica: {atributos_finais}"
        self._log(NivelLog.INFO, "identificar_atributos", msg, mostrar_usuario=True)
        return True, msg


    # ---------------------------------------------------------
    # Fluxo principal
    # ---------------------------------------------------------

    def processar_amostragem(
        self,
    ) -> Tuple[bool, str, Optional[gpd.GeoDataFrame], Optional[List[str]]]:
        """
        Executa o fluxo completo de processamento da amostragem:

          1. Carregar CSV
          2. Detectar campos comuns da plataforma
          3. Detectar lat/lon
          4. Criar gdf_4326
          5. Reprojetar para UTM
          6. Checar pontos dentro do contorno
          7. Identificar colunas de atributos (genérico)

        Retorna:
          (sucesso, mensagem_final, gdf_utm, lista_atributos)
        """
        ok, msg = self.carregar_csv()
        if not ok:
            return False, msg, None, None

        ok, msg = self._detectar_campos_comuns_plataforma()
        if not ok:
            return False, msg, None, None

        ok, msg = self.detectar_lat_lon()
        if not ok:
            if "Lat e Long não encontrados" in msg:
                self._log(
                    NivelLog.INFO,
                    "processar_amostragem",
                    "Nenhuma coluna de coordenadas detectada. "
                    "Aplicando fluxo de ponto central com médias dos campos.",
                    mostrar_usuario=True,
                )

                ok_alt, msg_alt = self._criar_ponto_central_sem_coordenadas()
                if not ok_alt:
                    return False, msg_alt, None, None

                ok, msg_chk = self.checar_pontos_dentro_contorno()
                if not ok:
                    return False, msg_chk, None, None

                ok, msg_attr = self.identificar_atributos()
                if not ok:
                    return False, msg_attr, None, None

                msg_final = (
                    "Amostragem sem coordenadas processada com sucesso: "
                    f"{len(self.gdf_utm)} ponto central, "
                    f"{len(self.col_atributos)} atributos genéricos (média por campo)."
                )
                self._log(
                    NivelLog.INFO,
                    "processar_amostragem",
                    msg_final,
                    mostrar_usuario=True,
                )
                return True, msg_final, self.gdf_utm, self.col_atributos

            return False, msg, None, None

        ok, msg = self.criar_gdf_4326()
        if not ok:
            return False, msg, None, None

        ok, msg = self.reprojetar_para_utm()
        if not ok:
            return False, msg, None, None

        ok, msg = self.checar_pontos_dentro_contorno()
        if not ok:
            return False, msg, None, None

        ok, msg = self.identificar_atributos()
        if not ok:
            return False, msg, None, None

        msg_final = (
            f"Amostragem processada com sucesso: "
            f"{len(self.gdf_utm)} pontos, {len(self.col_atributos)} atributos genéricos."
        )
        self._log(
            NivelLog.INFO,
            "processar_amostragem",
            msg_final,
            mostrar_usuario=True,
        )

        return True, msg_final, self.gdf_utm, self.col_atributos
