from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

from .logger import LoggerAgricola, NivelLog


# Colunas comuns fixas da plataforma
CAMPOS_FIXOS = [
    "id_ponto",
    "Cliente",
    "Fazenda",
    "Talhão",
    "Gleba",
    "Lat",
    "Long",
]

# Mapeamento processo -> sufixo de coluna no grid completo
PROCESSO_SUFFIX = {
    "solo": "_solo",
    "prod": "_prod",
    "compac": "_compac",
    "nemat": "_nemat",
    "foliar": "_foliar",
}

# Colunas oficiais de cada processo (nomes da plataforma)
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


class GridCompletoManager:
    """
    Monta/atualiza o grid completo de uma lavoura, no padrão da plataforma:

    - Uma linha por id_ponto (grid estático de pontos)
    - Colunas fixas (sem sufixo): id_ponto, Cliente, Fazenda, Talhão, Gleba, Lat, Long
    - Colunas de Data por processo: Data_solo, Data_prod, Data_compac, Data_nemat, Data_foliar
    - Colunas de atributos com sufixo de processo: ex. 'K_solo', 'K_foliar', 'Kg/Ha_prod'
    - Busca recursiva por {nome_lavoura}*_grid.csv dentro de dir_grids (subpastas csv_solo, csv_prod, etc.)
    """

    def __init__(
        self,
        logger: LoggerAgricola,
        dir_grids: str | Path,
        dir_saida: str | Path,
    ) -> None:
        self.logger = logger
        self.dir_grids = Path(dir_grids)
        self.dir_saida = Path(dir_saida)

        self.dir_saida.mkdir(parents=True, exist_ok=True)

    # -------------------- logging helper --------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)

    # -------------------- helpers de coluna esperada --------------------

    def _colunas_esperadas(self) -> List[str]:
        """
        Retorna a lista completa de colunas esperadas no grid_completo:

        - CAMPOS_FIXOS (sem sufixo)
        - Data_{processo}
        - todos os atributos de cada processo com sufixo
        """
        cols = CAMPOS_FIXOS.copy()

        for proc, suffix in PROCESSO_SUFFIX.items():
            cols.append(f"Data{suffix}")

            for attr in COLS_PROCESSO.get(proc, []):
                cols.append(f"{attr}{suffix}")

        return cols
    
    def normalizar_data_str(self, valor) -> Optional[str]:
        """
        Normaliza datas para o formato ISO 'YYYY-MM-DD'.
    
        Aceita:
          - 'dd/mm/yyyy'
          - 'dd-mm-yyyy'
          - 'yyyy-mm-dd'
          - 'yyyy/mm/dd'
          - datetime / Timestamp
          - pd.NA / None
    
        Retorna:
          - string 'YYYY-MM-DD'
          - None se inválido ou vazio
        """
        if pd.isna(valor):
            return None

        if isinstance(valor, (pd.Timestamp,)):
            return valor.strftime("%Y-%m-%d")
    
        s = str(valor).strip()
    
        if not s:
            return None
    
        try:
            dt = pd.to_datetime(s, dayfirst=True, errors="raise")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    
        try:
            dt = pd.to_datetime(s, dayfirst=False, errors="raise")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    
    # -------------------- helpers de caminho --------------------

    def _caminho_grid_completo(self, nome_lavoura: str) -> Path:
        return self.dir_saida / f"{nome_lavoura}_grid_completo.csv"

    def _carregar_grid_completo(self, nome_lavoura: str) -> Optional[pd.DataFrame]:
        caminho = self._caminho_grid_completo(nome_lavoura)
        if not caminho.exists():
            return None

        df = pd.read_csv(caminho)
        self._log(
            NivelLog.INFO,
            "grid_completo",
            f"Grid completo existente carregado: {caminho}",
        )
        return df

    # -------------------- identificação de processo --------------------

    def _inferir_processo_de_arquivo(self, caminho: Path) -> Optional[str]:
        """
        Infere o processo a partir:
          1) pasta pai 'csv_{processo}' (padrão antigo)
          2) fallback por nome do arquivo:
             - se contém '_prod_' -> processo = 'prod' (yield por projeto_colheita)
        """
        nome_dir = caminho.parent.name.lower()
    
        if nome_dir.startswith("csv_"):
            proc = nome_dir[4:]  # remove 'csv_'
            if proc in PROCESSO_SUFFIX:
                return proc
    
        nome_arq = caminho.name.lower()
        if "_prod_" in nome_arq:
            return "prod"
    
        self._log(
            NivelLog.WARNING,
            "grid_completo_inferir_processo",
            f"Não foi possível inferir processo para arquivo '{caminho}'. "
            f"Pasta pai: '{nome_dir}'. Arquivo será ignorado.",
        )
        return None
    

    # -------------------- listar grids (busca recurisva) --------------------

    def _listar_grids_lavoura(self, nome_lavoura: str) -> List[Path]:
        """
        Busca recursivamente por arquivos {nome_lavoura}*_grid.csv
        dentro de dir_grids/{nome_lavoura} (inclui subpastas csv_{processo}).
        NÃO inclui o grid_completo.
        """
        raiz = self.dir_grids / nome_lavoura
    
        if not raiz.exists():
            self._log(
                NivelLog.WARNING,
                "grid_completo_listar",
                f"Diretório da lavoura não encontrado: {raiz}",
                mostrar_usuario=True,
            )
            return []
    
        arquivos = sorted(
            p
            for p in raiz.rglob("*.csv")
            if p.name.startswith(f"{nome_lavoura}_") and p.name.endswith("_grid.csv")
        )
    
        self._log(
            NivelLog.INFO,
            "grid_completo_listar",
            (
                f"{len(arquivos)} arquivos de grid encontrados para lavoura "
                f"'{nome_lavoura}' em {raiz} (busca recursiva)."
            ),
            dados={"arquivos": [str(a) for a in arquivos]},
            mostrar_usuario=True,
        )
        return arquivos
    

    # -------------------- carregar e padronizar grid unitário --------------------

    def _carregar_grid_unitario_padronizado(
        self,
        caminho: Path,
        processo: str,
    ) -> pd.DataFrame:
        """
        Carrega um grid unitário e padroniza:

        - Garante presença de id_ponto, Lat, Long, Data
        - Garante presença de Cliente, Fazenda, Talhão, Gleba (cria em branco se faltar)
        - Garante TODAS as colunas oficiais do processo (preenchendo com NaN se não houver)
        - Renomeia:
            Data -> Data_{processo}
            atributos -> atributo_{processo}
        - Retorna DataFrame com:
            CAMPOS_FIXOS + Data_{processo} + atributos_{processo}
        """
        df = pd.read_csv(caminho)

        chaves_minimas = ["id_ponto", "Lat", "Long", "Data"]
        faltando = [c for c in chaves_minimas if c not in df.columns]
        if faltando:
            raise ValueError(
                f"Arquivo {caminho.name} não segue o padrão esperado. "
                f"Faltam colunas: {faltando}"
            )

        df["id_ponto"] = pd.to_numeric(df["id_ponto"], errors="coerce").astype("Int64")

        for col in ["Cliente", "Fazenda", "Talhão", "Gleba"]:
            if col not in df.columns:
                df[col] = pd.NA

        cols_oficiais = COLS_PROCESSO.get(processo, [])
        for col in cols_oficiais:
            if col not in df.columns:
                df[col] = pd.NA

        df_base = df[["id_ponto", "Cliente", "Fazenda", "Talhão", "Gleba", "Lat", "Long"]].copy()

        suffix = PROCESSO_SUFFIX[processo]
        data_col_proc = f"Data{suffix}"
        df_data = df[["id_ponto", "Data"]].copy().rename(columns={"Data": data_col_proc})

        df_attr = df[["id_ponto"] + cols_oficiais].copy()
        rename_map = {col: f"{col}{suffix}" for col in cols_oficiais}
        df_attr = df_attr.rename(columns=rename_map)

        df_merged = df_base.merge(df_data, on="id_ponto", how="left")
        df_merged = df_merged.merge(df_attr, on="id_ponto", how="left")

        return df_merged

    # -------------------- montagem do grid completo --------------------

    def montar_grid_completo(
        self,
        nome_lavoura: str,
    ) -> Tuple[bool, str, Optional[pd.DataFrame], Optional[Path]]:
        """
        Atualiza/monta o grid completo da lavoura.

          - Lê grid_completo existente (se houver) como base.
          - Busca {nome_lavoura}*_grid.csv em dir_grids (subpastas csv_{processo}).
          - Para cada arquivo:
              * infere o processo (solo, prod, compac, nemat, foliar)
              * carrega grid unitário padronizado
              * extrai Data_{processo} do arquivo
              * se essa Data_{processo} já existir no grid_completo para esse processo,
                considera o arquivo já integrado e pula
              * caso contrário, integra preenchendo apenas o processo específico,
                a partir da “primeira linha vazia” para aquele processo (estilo tetris)
          - Garante CAMPOS_FIXOS uma única vez
          - Garante colunas de Data e atributos estáticas (criando vazias se necessário)
        """

        self.logger.adicionar_contexto(
            componente="grid_completo",
            lavoura=nome_lavoura,
        )

        df_base = self._carregar_grid_completo(nome_lavoura)

        arquivos = self._listar_grids_lavoura(nome_lavoura)
        if not arquivos and df_base is None:
            msg = (
                f"Nenhum arquivo de grid encontrado para lavoura '{nome_lavoura}' "
                f"em {self.dir_grids}, e não há grid_completo pré-existente."
            )
            self._log(
                NivelLog.ERROR,
                "grid_completo",
                msg,
                mostrar_usuario=True,
            )
            return False, msg, None, None

        if df_base is not None:
            if "id_ponto" not in df_base.columns:
                self._log(
                    NivelLog.WARNING,
                    "grid_completo",
                    "Grid completo existente não possui coluna 'id_ponto'. "
                    "Ele será ignorado e reconstruído do zero.",
                    mostrar_usuario=True,
                )
                df_base = None
            else:
                df_base["id_ponto"] = pd.to_numeric(
                    df_base["id_ponto"], errors="coerce"
                ).astype("Int64")


        datas_existentes_por_proc: Dict[str, set] = {
            proc: set() for proc in PROCESSO_SUFFIX.keys()
        }
        if df_base is not None:
            for proc, suffix in PROCESSO_SUFFIX.items():
                col_data = f"Data{suffix}"
                if col_data in df_base.columns:
                    s = (
                        df_base[col_data]
                        .dropna()
                        .astype("string")
                        .map(self.normalizar_data_str)
                        .dropna()
                    )
                    datas_existentes_por_proc[proc] = set(s.unique())

                    df_base[col_data] = df_base[col_data].astype("string").map(self.normalizar_data_str)


        for caminho in arquivos:
            processo = self._inferir_processo_de_arquivo(caminho)
            if processo is None:
                continue

            try:
                df_unit = self._carregar_grid_unitario_padronizado(caminho, processo)
            except Exception as e:
                self._log(
                    NivelLog.ERROR,
                    "grid_completo",
                    f"Erro ao carregar grid '{caminho.name}': {e}",
                    mostrar_usuario=True,
                )
                continue

            suffix = PROCESSO_SUFFIX[processo]
            col_data_proc = f"Data{suffix}"

            if col_data_proc in df_unit.columns:
                df_unit[col_data_proc] = (
                    df_unit[col_data_proc]
                    .astype("string")
                    .map(self.normalizar_data_str)
                )
            
            if col_data_proc not in df_unit.columns:
                ...
                datas_unit = []
            else:
                datas_unit = (
                    df_unit[col_data_proc]
                    .dropna()
                    .astype("string")
                    .unique()
                    .tolist()
                )

            data_arquivo: Optional[str] = None
            if len(datas_unit) == 0:
                self._log(
                    NivelLog.WARNING,
                    "grid_completo",
                    f"Grid '{caminho.name}' (processo={processo}) não possui nenhuma "
                    f"Data válida em '{col_data_proc}'. Integração sem checagem de duplicata.",
                    mostrar_usuario=True,
                )
            else:
                if len(datas_unit) > 1:
                    self._log(
                        NivelLog.WARNING,
                        "grid_completo",
                        f"Grid '{caminho.name}' (processo={processo}) possui múltiplas "
                        f"datas em '{col_data_proc}': {datas_unit}. "
                        f"Utilizando a primeira para checagem de duplicata.",
                        mostrar_usuario=True,
                    )
                data_arquivo = self.normalizar_data_str(datas_unit[0])

                if (
                    data_arquivo is not None
                    and data_arquivo in datas_existentes_por_proc.get(processo, set())
                ):
                    self._log(
                        NivelLog.INFO,
                        "grid_completo",
                        f"Grid '{caminho.name}' (processo={processo}, Data={data_arquivo}) "
                        "já foi integrado anteriormente. Ignorando.",
                        mostrar_usuario=True,
                    )
                    continue

            if df_base is None:
                df_base = df_unit.copy()
                self._log(
                    NivelLog.INFO,
                    "grid_completo",
                    f"Iniciando grid completo a partir de '{caminho.name}' "
                    f"(processo={processo}).",
                )
            else:
                cols_proc = [col_data_proc] + [
                    f"{attr}{suffix}" for attr in COLS_PROCESSO.get(processo, [])
                ]
                
                for col in cols_proc:
                    if col not in df_base.columns:
                        if col.startswith("Data"):
                            df_base[col] = pd.Series([pd.NA] * len(df_base), dtype="object")
                        else:
                            df_base[col] = pd.Series([pd.NA] * len(df_base))

                for col in CAMPOS_FIXOS:
                    if col not in df_base.columns:
                        df_base[col] = pd.NA

                for _, row in df_unit.iterrows():
                    id_ponto = row["id_ponto"]
                    if pd.isna(id_ponto):
                        continue

                    mask_id = df_base["id_ponto"] == id_ponto

                    if col_data_proc in df_base.columns:
                        mask_vaga = mask_id & df_base[col_data_proc].isna()
                    else:
                        mask_vaga = mask_id

                    if mask_vaga.any():
                        idx = df_base[mask_vaga].index[0]
                        for col in CAMPOS_FIXOS:
                            if col in df_base.columns and col in row.index:
                                df_base.at[idx, col] = row[col]
                        for col in cols_proc:
                            if col in row.index:
                                if col.startswith("Data"):
                                    df_base.at[idx, col] = self.normalizar_data_str(row[col])
                                else:
                                    df_base.at[idx, col] = row[col]
                    else:
                        nova = {}
                        
                        for col in CAMPOS_FIXOS:
                            if col in row.index and pd.notna(row[col]):
                                nova[col] = row[col]
                        
                        for col in cols_proc:
                            if col in row.index and pd.notna(row[col]):
                                if col.startswith("Data"):
                                    nova[col] = self.normalizar_data_str(row[col])
                                else:
                                    nova[col] = row[col]
                        
                        df_base = pd.concat([df_base, pd.DataFrame([nova])], ignore_index=True)

                self._log(
                    NivelLog.INFO,
                    "grid_completo",
                    f"Arquivo '{caminho.name}' integrado no modo TETRIS (processo={processo}).",
                )

            if data_arquivo is not None:
                datas_existentes_por_proc.setdefault(processo, set()).add(data_arquivo)

        if df_base is None:
            msg = (
                f"Não foi possível montar grid completo para lavoura '{nome_lavoura}'. "
                "Nenhum grid unitário válido foi integrado."
            )
            self._log(
                NivelLog.ERROR,
                "grid_completo",
                msg,
                mostrar_usuario=True,
            )
            return False, msg, None, None

        cols_esperadas = self._colunas_esperadas()
        for col in cols_esperadas:
            if col not in df_base.columns:
                df_base[col] = pd.NA

        cols_presentes = list(df_base.columns)

        cols_fixas_presentes = [c for c in CAMPOS_FIXOS if c in cols_presentes]

        cols_proc_ord = []
        for proc, suffix in PROCESSO_SUFFIX.items():
            data_col = f"Data{suffix}"
            if data_col in cols_presentes:
                cols_proc_ord.append(data_col)

            for attr in COLS_PROCESSO.get(proc, []):
                col_attr = f"{attr}{suffix}"
                if col_attr in cols_presentes:
                    cols_proc_ord.append(col_attr)

        cols_core = set(cols_fixas_presentes + cols_proc_ord)
        cols_extras = [c for c in cols_presentes if c not in cols_core]

        df_base = df_base[cols_fixas_presentes + cols_proc_ord + cols_extras]
        
        if "id_ponto" in df_base.columns:
            df_base = df_base.sort_values("id_ponto").reset_index(drop=True)
        
        caminho_saida = self._caminho_grid_completo(nome_lavoura)
        df_base.to_csv(caminho_saida, index=False)
        
        msg_final = (
            f"Grid completo da lavoura '{nome_lavoura}' atualizado. "
            f"Linhas (slots de amostragem x pontos): {len(df_base)}, "
            f"colunas: {len(df_base.columns)}. "
            f"Arquivo: {caminho_saida.name}"
        )
        self._log(
            NivelLog.INFO,
            "grid_completo",
            msg_final,
            mostrar_usuario=True,
        )

        return True, msg_final, df_base, caminho_saida
