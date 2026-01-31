from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Union
import os
import re
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, root_validator
import numpy as np
import pandas as pd
import requests

from facilita_agro.logger import LoggerAgricola, NivelLog
from facilita_agro.processador_lavoura import ProcessadorLavoura
from facilita_agro.pipeline_completa import PipelineAgricola
from facilita_agro.pipeline_yield import PipelineYield
from facilita_agro.grid_completo import GridCompletoManager
from facilita_agro.processador_amostragem_v2 import ProcessadorAmostragemV2
from facilita_agro.blob_storage import upload_blob_file

from facilita_agro.filtro_yield_preprocess import PreprocessadorYield, ConfigPreprocessamentoYield

# ============================================================
# CONFIGURAÇÃO BÁSICA DE DIRETÓRIOS (ajuste conforme servidor)
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# Entradas
DIR_CONTORNOS = BASE_DIR / "data" / "contornos"
DIR_AMOSTRAGEM = BASE_DIR / "data" / "amostragem"
DIR_YIELD = BASE_DIR / "data" / "yield"

# Saídas de raster
DIR_RASTERS_AMOSTRAGEM = BASE_DIR / "outputs" / "rasters_amostragem"
DIR_RASTERS_YIELD = BASE_DIR / "outputs" / "rasters_yield"

# Saída de CSVs (base para todos os processos)
DIR_CSV_BASE = BASE_DIR / "outputs" / "csv"

# Grid completo consolidado
DIR_CSV_GRID_COMPLETO = BASE_DIR / "outputs" / "csv_grid_completo"

# Staging de yield (pré-processado)
DIR_YIELD_TEMP = BASE_DIR / "outputs" / "yield_temp"

PIXEL_SIZE_DEFAULT = 10.0

# ============================================================
# MODELOS Pydantic (entrada/saída da API)
# ============================================================

ProcessoTipo = Literal["solo", "foliar", "compac", "nemat"]


class ProcessarAmostragemRequest(BaseModel):
    lavoura: str = Field(..., description="Nome da lavoura (prefixo dos arquivos de contorno/grade).")
    arquivo_csv: str = Field(..., description="Nome do arquivo CSV de amostragem (dentro de DIR_AMOSTRAGEM).")
    processo: ProcessoTipo = Field(
        ...,
        description="Tipo de processo: 'solo', 'foliar', 'compac' (compactação) ou 'nemat' (nematoide).",
    )
    atualizar_grid_completo: bool = Field(
        default=True,
        description="Se True, atualiza o grid_completo da lavoura após o processamento.",
    )


# Yield em 2 etapas (A: preprocessar; B: interpolar)
class YieldPreprocessRequest(BaseModel):
    lavoura: str = Field(..., description="Nome da lavoura (usado para localizar a pasta e nomear saídas).")
    projeto_colheita: str = Field(..., description="Nome do projeto/colheita (ex.: soja_2025).")


class YieldInterpolarRequest(BaseModel):
    lavoura: str = Field(..., description="Nome da lavoura (prefixo dos arquivos de contorno/grade).")
    projeto_colheita: str = Field(..., description="Nome do projeto/colheita (ex.: soja_2025).")
    atualizar_grid_completo: bool = Field(
        default=True,
        description="Se True, atualiza o grid_completo da lavoura após a interpolação.",
    )


class AtualizarGridCompletoRequest(BaseModel):
    lavoura: str = Field(..., description="Nome da lavoura para atualização do grid_completo.")


class ProcessamentoResponse(BaseModel):
    sucesso: bool
    mensagem: str
    caminho_csv_grid: Optional[str] = None
    caminho_csv_grid_completo: Optional[str] = None
    colunas: Optional[List[str]] = None


class AtributoV2(BaseModel):
    nome: str
    valor: Optional[Union[float, int, str]] = None


class DadoAmostraV2(BaseModel):
    latitude: float
    longitude: float
    ponto_coleta: Optional[str] = None
    profundidade: int
    atributos: List[AtributoV2]


class ProcessarAmostragemV2Request(BaseModel):
    tipo: Literal["gleba", "talhao"]
    id: int
    gerar_csv: bool = Field(default=True, description="Se True, envia dados do grid completo.")
    url_kml: str
    url_grade: str
    processo: ProcessoTipo
    cliente_id: int
    data: str
    fazenda: int
    talhao: int
    gleba: Optional[int] = None
    cultura: Optional[str] = None
    profundidade: Optional[int] = None
    dados: List[DadoAmostraV2]

    @root_validator
    def validar_contexto(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        tipo = values.get("tipo")
        talhao = values.get("talhao")
        gleba = values.get("gleba")
        if not talhao:
            raise ValueError("Campo 'talhao' é obrigatório.")
        if tipo == "gleba" and gleba is None:
            raise ValueError("Campo 'gleba' é obrigatório quando tipo=gleba.")
        return values


class RasterInterpoladoResponse(BaseModel):
    atributo: str
    url: str


class ProcessamentoV2Response(BaseModel):
    sucesso: bool
    mensagem: str
    rasters: List[RasterInterpoladoResponse]
    grid_completo_enviado: bool


class GridCompletoResponse(BaseModel):
    sucesso: bool
    mensagem: str
    caminho_csv_grid_completo: Optional[str] = None
    colunas: Optional[List[str]] = None


class ChecarPerimetroV2Request(BaseModel):
    tipo: Literal["talhao", "gleba"] = Field(..., description="Tipo de área.")
    id: int = Field(..., description="Identificador da área.")


class GerenciarGradeV2Request(BaseModel):
    tipo: Literal["talhao", "gleba"] = Field(..., description="Tipo de área.")
    id: int = Field(..., description="Identificador da área.")
    url_kml: Optional[str] = Field(
        default=None,
        description="URL do KML. Se omitido, será consultado via endpoint do front.",
    )
    url_grade: Optional[str] = Field(
        default=None,
        description="URL da grade existente, se já houver.",
    )


class KmlGradeResponse(BaseModel):
    sucesso: bool
    mensagem: str
    dados: Optional[dict] = None


# ============================================================
# Inicialização FastAPI
# ============================================================

app = FastAPI(
    title="Facilita Agro API",
    description="API de processamento geoespacial para dados agrícolas (amostragem e produtividade).",
    version="0.3.1",
)


# ============================================================
# Helpers
# ============================================================

def _atualizar_grid_completo(logger: LoggerAgricola, lavoura: str) -> Optional[str]:
    """Atualiza grid_completo; retorna caminho se ok, senão None."""
    manager = GridCompletoManager(
        logger=logger,
        dir_grids=DIR_CSV_BASE,          # o manager busca recursivamente em DIR_CSV_BASE/{lavoura}/...
        dir_saida=DIR_CSV_GRID_COMPLETO,
    )
    ok_gc, msg_gc, df_gc, caminho_gc = manager.montar_grid_completo(lavoura)
    if ok_gc and caminho_gc is not None:
        return str(caminho_gc)

    logger.log(
        NivelLog.WARNING,
        "grid_completo",
        f"Falha ao atualizar grid_completo: {msg_gc}",
        mostrar_usuario=True,
    )
    return None


def _slugify_nome(valor: str) -> str:
    valor = valor.strip().lower()
    valor = re.sub(r"[^\w]+", "_", valor, flags=re.UNICODE)
    return valor.strip("_")


def _storage_base_url() -> str:
    base_url = os.getenv("FACILITAGRO_FRONTEND_BASE_URL")
    if not base_url:
        raise ValueError("FACILITAGRO_FRONTEND_BASE_URL não configurada.")
    return base_url.rstrip("/")


def _post_storage(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_storage_base_url()}{endpoint}"
    resposta = requests.post(url, json=payload, timeout=60)
    resposta.raise_for_status()
    try:
        return resposta.json()
    except ValueError:
        return {"status": "ok"}


def _serializar_valor(valor: Any) -> Any:
    if pd.isna(valor):
        return None
    if isinstance(valor, (np.integer, np.floating)):
        return valor.item()
    return valor


def _montar_payload_grid_completo(
    req: ProcessarAmostragemV2Request,
    df_final: pd.DataFrame,
    profundidade: Optional[int],
) -> Dict[str, Any]:
    cols_base = ["id_ponto", "Data", "Cliente", "Fazenda", "Talhão", "Gleba", "Lat", "Long"]
    atributos_cols = [c for c in df_final.columns if c not in cols_base]

    pontos = []
    for _, row in df_final.iterrows():
        atributos = [
            {"nome": col, "valor": _serializar_valor(row[col])}
            for col in atributos_cols
        ]
        pontos.append(
            {
                "id_ponto": _serializar_valor(row.get("id_ponto")),
                "lat": _serializar_valor(row.get("Lat")),
                "long": _serializar_valor(row.get("Long")),
                "data": _serializar_valor(row.get("Data")),
                "cliente": _serializar_valor(row.get("Cliente")),
                "fazenda": _serializar_valor(row.get("Fazenda")),
                "talhao": _serializar_valor(row.get("Talhão")),
                "gleba": _serializar_valor(row.get("Gleba")),
                "profundidade": profundidade,
                "atributos": atributos,
            }
        )

    return {
        "tipo": req.tipo,
        "id": req.id,
        "processo": req.processo,
        "cliente_id": req.cliente_id,
        "data": req.data,
        "fazenda": req.fazenda,
        "talhao": req.talhao,
        "gleba": req.gleba,
        "cultura": req.cultura,
        "profundidade": profundidade,
        "pontos": pontos,
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


# --------------------- AMOSTRAGEM (solo/foliar/compac/nemat) ---------------------

@app.post("/processar-amostragem", response_model=ProcessamentoResponse)
def processar_amostragem(req: ProcessarAmostragemRequest):
    """
    Processa um arquivo de amostragem (solo, foliar, nematoide, compactação),
    interpolando os atributos selecionados, amostrando na grade e gerando
    um CSV {lavoura}_{processo}_<campanha>_grid.csv.

    Se atualizar_grid_completo=True, chama também o módulo de grid_completo.
    """
    lavoura = req.lavoura
    processo = req.processo

    logger = LoggerAgricola(lavoura, salvar_arquivo=True)

    pipeline = PipelineAgricola(
        logger=logger,
        dir_contornos=DIR_CONTORNOS,
        dir_amostragem=DIR_AMOSTRAGEM,
        dir_rasters=DIR_RASTERS_AMOSTRAGEM,
        dir_csv_base=DIR_CSV_BASE,
        pixel_size=PIXEL_SIZE_DEFAULT,
    )

    ok, msg, df_final, caminho_csv = pipeline.executar_pipeline_completa(
        nome_lavoura=lavoura,
        nome_arquivo_csv=req.arquivo_csv,
        processo=processo,
    )

    if not ok or caminho_csv is None:
        raise HTTPException(status_code=400, detail=msg)

    caminho_csv_grid_completo = None
    if req.atualizar_grid_completo:
        caminho_csv_grid_completo = _atualizar_grid_completo(logger, lavoura)

    return ProcessamentoResponse(
        sucesso=True,
        mensagem=msg,
        caminho_csv_grid=str(caminho_csv),
        caminho_csv_grid_completo=caminho_csv_grid_completo,
        colunas=list(df_final.columns) if df_final is not None else None,
    )


@app.post("/processar-amostragem-upload", response_model=ProcessamentoResponse)
async def processar_amostragem_upload(
    lavoura: str = Form(...),
    processo: ProcessoTipo = Form(...),
    atualizar_grid_completo: bool = Form(True),
    arquivo_csv: UploadFile = File(...),
):
    """Versão com upload de arquivo CSV de amostragem."""
    if not arquivo_csv.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="O arquivo enviado deve ser um .csv")

    nome_csv = f"{lavoura}_{processo}.csv"
    caminho_destino = DIR_AMOSTRAGEM / nome_csv
    DIR_AMOSTRAGEM.mkdir(parents=True, exist_ok=True)

    conteudo = await arquivo_csv.read()
    with open(caminho_destino, "wb") as f:
        f.write(conteudo)

    logger = LoggerAgricola(lavoura, salvar_arquivo=True)

    pipeline = PipelineAgricola(
        logger=logger,
        dir_contornos=DIR_CONTORNOS,
        dir_amostragem=DIR_AMOSTRAGEM,
        dir_rasters=DIR_RASTERS_AMOSTRAGEM,
        dir_csv_base=DIR_CSV_BASE,
        pixel_size=PIXEL_SIZE_DEFAULT,
    )

    ok, msg, df_final, caminho_csv_grid = pipeline.executar_pipeline_completa(
        nome_lavoura=lavoura,
        nome_arquivo_csv=nome_csv,
        processo=processo,
    )

    if not ok or caminho_csv_grid is None:
        raise HTTPException(status_code=400, detail=msg)

    caminho_csv_grid_completo = None
    if atualizar_grid_completo:
        caminho_csv_grid_completo = _atualizar_grid_completo(logger, lavoura)

    return ProcessamentoResponse(
        sucesso=True,
        mensagem=msg,
        caminho_csv_grid=str(caminho_csv_grid),
        caminho_csv_grid_completo=caminho_csv_grid_completo,
        colunas=list(df_final.columns) if df_final is not None else None,
    )


# ------------------------- YIELD (A: PREPROCESSAR) -------------------------

@app.post("/yield/preprocessar", response_model=ProcessamentoResponse)
def yield_preprocessar(req: YieldPreprocessRequest):
    """
    Pré-processa (filtra + corrige por balança) TODOS os CSVs encontrados em:
      DIR_YIELD/{lavoura}/{projeto_colheita}/

    Exporta arquivos pré-processados em UTM para:
      DIR_YIELD_TEMP/{lavoura}/{projeto_colheita}/
    """
    lavoura = req.lavoura
    projeto = req.projeto_colheita

    logger = LoggerAgricola(lavoura, salvar_arquivo=True)

    prep = PreprocessadorYield(
        logger=logger,
        config_preprocess=ConfigPreprocessamentoYield(
            dir_yield_temp=str(DIR_YIELD_TEMP),
            formato="gpkg",
        ),
    )

    ok, msg, saidas = prep.preprocessar_pasta_yield(
        dir_yield=str(DIR_YIELD),
        nome_lavoura=lavoura,
        projeto_colheita=projeto,
    )

    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # Aqui ainda não existe "csv_grid" final; devolvemos mensagem + lista em colunas (opcional)
    return ProcessamentoResponse(
        sucesso=True,
        mensagem=msg,
        caminho_csv_grid=None,
        caminho_csv_grid_completo=None,
        colunas=[str(p) for p in (saidas or [])],
    )


@app.post("/yield/preprocessar-upload", response_model=ProcessamentoResponse)
async def yield_preprocessar_upload(
    lavoura: str = Form(...),
    projeto_colheita: str = Form(...),
    arquivo_csv: UploadFile = File(...),
):
    """
    Upload de um CSV de yield para a pasta do projeto:
      salva em DIR_YIELD/{lavoura}/{projeto_colheita}/
      e executa o preprocessamento do projeto (pasta inteira).
    """
    if not arquivo_csv.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="O arquivo enviado deve ser um .csv")

    # Salva no diretório do projeto
    dest_dir = DIR_YIELD / lavoura / projeto_colheita
    dest_dir.mkdir(parents=True, exist_ok=True)
    caminho_destino = dest_dir / arquivo_csv.filename

    conteudo = await arquivo_csv.read()
    with open(caminho_destino, "wb") as f:
        f.write(conteudo)

    logger = LoggerAgricola(lavoura, salvar_arquivo=True)

    prep = PreprocessadorYield(
        logger=logger,
        config_preprocess=ConfigPreprocessamentoYield(
            dir_yield_temp=str(DIR_YIELD_TEMP),
            formato="gpkg",
        ),
    )

    ok, msg, saidas = prep.preprocessar_pasta_yield(
        dir_yield=str(DIR_YIELD),
        nome_lavoura=lavoura,
        projeto_colheita=projeto_colheita,
    )

    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return ProcessamentoResponse(
        sucesso=True,
        mensagem=msg,
        caminho_csv_grid=None,
        caminho_csv_grid_completo=None,
        colunas=[str(p) for p in (saidas or [])],
    )


# ------------------------- YIELD (B: INTERPOLAR) -------------------------

@app.post("/yield/interpolar", response_model=ProcessamentoResponse)
def yield_interpolar(req: YieldInterpolarRequest):
    """
    Interpola yield a partir dos arquivos pré-processados em:
      DIR_YIELD_TEMP/{lavoura}/{projeto_colheita}/

    Saídas:
      - rasters em DIR_RASTERS_YIELD/{lavoura}/{projeto_colheita}/
      - CSV final em DIR_CSV_BASE/{lavoura}/{projeto_colheita}/
    """
    lavoura = req.lavoura
    projeto = req.projeto_colheita

    logger = LoggerAgricola(lavoura, salvar_arquivo=True)

    pipe_yld = PipelineYield(
        logger=logger,
        dir_contornos=DIR_CONTORNOS,
        dir_yield=str(DIR_YIELD),  # mantido por compatibilidade; interpolação lê do staging
        dir_rasters=DIR_RASTERS_YIELD,
        dir_csv_base=DIR_CSV_BASE,
        pixel_size=PIXEL_SIZE_DEFAULT,
        dir_yield_temp=str(DIR_YIELD_TEMP),
    )

    ok, msg, df_final, caminho_csv = pipe_yld.executar_interpolacao_yield(
        nome_lavoura=lavoura,
        projeto_colheita=projeto,
    )

    if not ok or caminho_csv is None:
        raise HTTPException(status_code=400, detail=msg)

    caminho_csv_grid_completo = None
    if req.atualizar_grid_completo:
        caminho_csv_grid_completo = _atualizar_grid_completo(logger, lavoura)

    return ProcessamentoResponse(
        sucesso=True,
        mensagem=msg,
        caminho_csv_grid=str(caminho_csv),
        caminho_csv_grid_completo=caminho_csv_grid_completo,
        colunas=list(df_final.columns) if df_final is not None else None,
    )


# --------------------- GRID COMPLETO (isolado) ---------------------

@app.post("/atualizar-grid-completo", response_model=GridCompletoResponse)
def atualizar_grid_completo(req: AtualizarGridCompletoRequest):
    """
    Atualiza o grid completo de uma lavoura com base em todos os
    {lavoura}*_grid.csv encontrados em outputs/csv/{lavoura}/ (e subpastas).
    """
    lavoura = req.lavoura
    logger = LoggerAgricola(lavoura, salvar_arquivo=True)

    manager = GridCompletoManager(
        logger=logger,
        dir_grids=DIR_CSV_BASE,
        dir_saida=DIR_CSV_GRID_COMPLETO,
    )

    ok, msg, df, caminho = manager.montar_grid_completo(lavoura)
    if not ok or caminho is None:
        raise HTTPException(status_code=400, detail=msg)

    return GridCompletoResponse(
        sucesso=True,
        mensagem=msg,
        caminho_csv_grid_completo=str(caminho),
        colunas=list(df.columns) if df is not None else None,
    )


# --------------------- V2 KML/Grade ---------------------


@app.post("/v2/checar-perimetro", response_model=KmlGradeResponse)
def checar_perimetro_v2(req: ChecarPerimetroV2Request):
    """Carrega o perímetro via URL de KML obtida do front."""
    logger = LoggerAgricola(f"{req.tipo}_{req.id}", salvar_arquivo=True)
    processador = ProcessadorLavoura(
        nome_lavoura=f"{req.tipo}_{req.id}",
        diretorio_base=BASE_DIR,
        logger=logger,
    )

    ok, msg, dados = processador.checar_perimetroV2(req.tipo, req.id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return KmlGradeResponse(sucesso=True, mensagem=msg, dados=dados)


@app.post("/v2/gerenciar-grade", response_model=KmlGradeResponse)
def gerenciar_grade_v2(req: GerenciarGradeV2Request):
    """Valida ou cria grade via URL, com upload ao blob storage quando necessário."""
    logger = LoggerAgricola(f"{req.tipo}_{req.id}", salvar_arquivo=True)
    processador = ProcessadorLavoura(
        nome_lavoura=f"{req.tipo}_{req.id}",
        diretorio_base=BASE_DIR,
        logger=logger,
    )

    ok, msg, dados = processador.checar_perimetroV2(req.tipo, req.id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    url_kml = req.url_kml or dados.get("url_kml")
    url_grade = req.url_grade or dados.get("url_grade")

    if not url_kml:
        raise HTTPException(status_code=400, detail="URL do KML não encontrada.")

    ok_grade, msg_grade, dados_grade = processador.gerenciar_gradeV2(
        req.tipo,
        req.id,
        url_kml,
        url_grade,
    )
    if not ok_grade:
        raise HTTPException(status_code=400, detail=msg_grade)

    return KmlGradeResponse(sucesso=True, mensagem=msg_grade, dados=dados_grade)


# --------------------- V2 Processar Amostragem ---------------------


@app.post("/v2/processar-amostragem", response_model=ProcessamentoV2Response)
def processar_amostragem_v2(req: ProcessarAmostragemV2Request):
    processo = req.processo.lower()
    if processo not in {"solo", "prod", "compac", "nemat", "foliar"}:
        raise HTTPException(status_code=400, detail=f"Processo inválido: '{processo}'.")

    nome_lavoura = f"{req.tipo}_{req.id}"
    logger = LoggerAgricola(nome_lavoura, salvar_arquivo=True)
    logger.log(
        NivelLog.INFO,
        "v2_inicio",
        f"Iniciando processamento V2: tipo={req.tipo}, id={req.id}, processo={processo}",
        mostrar_usuario=True,
    )

    proc_lav = ProcessadorLavoura(
        nome_lavoura=nome_lavoura,
        diretorio_base=BASE_DIR,
        logger=logger,
    )

    logger.log(
        NivelLog.INFO,
        "v2_perimetro",
        f"Baixando perímetro via URL: {req.url_kml}",
        mostrar_usuario=True,
    )
    ok, msg = proc_lav.checar_perimetro_url(req.url_kml)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    logger.log(
        NivelLog.INFO,
        "v2_grade",
        f"Validando grade via URL: {req.url_grade}",
        mostrar_usuario=True,
    )
    ok_grade, msg_grade, _ = proc_lav.gerenciar_gradeV2(
        req.tipo,
        req.id,
        req.url_kml,
        req.url_grade,
    )
    if not ok_grade:
        raise HTTPException(status_code=400, detail=msg_grade)

    logger.log(
        NivelLog.INFO,
        "v2_payload",
        f"Montando amostragem V2 com {len(req.dados)} pontos.",
        mostrar_usuario=True,
    )

    profundidades = sorted({item.profundidade for item in req.dados})
    rasters_enviados: list[RasterInterpoladoResponse] = []
    grid_enviado = False
    rotulo_campanha = None

    for profundidade in profundidades:
        dados_filtrados = [item for item in req.dados if item.profundidade == profundidade]
        logger.log(
            NivelLog.INFO,
            "v2_profundidade",
            f"Processando profundidade={profundidade} com {len(dados_filtrados)} pontos.",
            mostrar_usuario=True,
        )

        proc_am = ProcessadorAmostragemV2(
            nome_lavoura=nome_lavoura,
            dados=[item.dict() for item in dados_filtrados],
            metadados={
                "data": req.data,
                "cliente": req.cliente_id,
                "fazenda": req.fazenda,
                "talhao": req.talhao,
                "gleba": req.gleba,
                "cultura": req.cultura,
            },
            contorno_utm=proc_lav.contorno,
            crs_utm=proc_lav.crs_utm,
            logger=logger,
        )

        logger.log(
            NivelLog.INFO,
            "v2_amostragem",
            "Processando amostragem (fluxo original).",
            mostrar_usuario=True,
        )
        ok_am, msg_am, gdf_utm, atributos = proc_am.processar_amostragem()
        if not ok_am or gdf_utm is None or atributos is None:
            raise HTTPException(status_code=400, detail=msg_am)

        proc_am.gdf_utm = gdf_utm
        proc_am.col_atributos = atributos

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.log(
                NivelLog.INFO,
                "v2_pipeline",
                f"Executando pipeline em diretório temporário: {temp_dir}",
                mostrar_usuario=True,
            )
            pipeline = PipelineAgricola(
                logger=logger,
                dir_contornos=DIR_CONTORNOS,
                dir_amostragem=DIR_AMOSTRAGEM,
                dir_rasters=Path(temp_dir),
                dir_csv_base=Path(temp_dir),
                pixel_size=PIXEL_SIZE_DEFAULT,
            )

            pipeline.logger.adicionar_contexto(
                lavoura=nome_lavoura,
                processo=processo,
                pixel_size=PIXEL_SIZE_DEFAULT,
            )

            rotulo_campanha = pipeline._obter_rotulo_campanha(proc_am)
            logger.log(
                NivelLog.INFO,
                "v2_rotulo_campanha",
                f"Rótulo da campanha definido: {rotulo_campanha}",
                mostrar_usuario=True,
            )
            analise_atributos, attrs_interp = pipeline._analisar_atributos(proc_am, processo)
            if not analise_atributos or not attrs_interp:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Nenhum atributo interpolável pôde ser analisado para processo '{processo}'."
                    ),
                )

            logger.log(
                NivelLog.INFO,
                "v2_interpolar",
                f"Interpolando atributos: {attrs_interp}",
                mostrar_usuario=True,
            )
            caminhos_rasters = pipeline._interpolar_todos_atributos(
                nome_lavoura,
                rotulo_campanha,
                processo,
                proc_lav,
                proc_am,
                analise_atributos,
                attrs_interp,
            )
            if not caminhos_rasters:
                raise HTTPException(status_code=400, detail="Nenhum raster foi gerado.")

            logger.log(
                NivelLog.INFO,
                "v2_amostragem_grade",
                f"Amostrando rasters na grade: {len(caminhos_rasters)} atributos.",
                mostrar_usuario=True,
            )
            df_final = pipeline._amostrar_rasters_na_grade(
                proc_lav,
                proc_am,
                processo,
                caminhos_rasters,
            )

            for attr, caminho_tif in caminhos_rasters.items():
                nome_blob = _slugify_nome(
                    f"{req.tipo}_{req.id}_{processo}_{rotulo_campanha}_{profundidade}_{attr}.tif"
                )
                try:
                    logger.log(
                        NivelLog.INFO,
                        "v2_upload_raster",
                        f"Enviando raster '{attr}' para blob: {nome_blob}",
                        mostrar_usuario=True,
                    )
                    url_blob = upload_blob_file(caminho_tif, nome_blob)
                    logger.log(
                        NivelLog.INFO,
                        "v2_notificar_raster",
                        f"Notificando raster interpolado '{attr}' no storage externo.",
                        mostrar_usuario=True,
                    )
                    _post_storage(
                        "/api/v2/add_raster_interpolados",
                        {
                            "tipo": req.tipo,
                            "id": req.id,
                            "processo": processo,
                            "atributo": attr,
                            "campanha": rotulo_campanha,
                            "url": url_blob,
                            "cliente_id": req.cliente_id,
                            "fazenda": req.fazenda,
                            "talhao": req.talhao,
                            "gleba": req.gleba,
                            "profundidade": profundidade,
                            "data": req.data,
                        },
                    )
                except Exception as exc:
                    logger.log(
                        NivelLog.ERROR,
                        "v2_raster_storage",
                        f"Falha ao enviar raster interpolado: {str(exc)}",
                        mostrar_usuario=True,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="Falha ao enviar raster interpolado para armazenamento externo.",
                    ) from exc
                rasters_enviados.append(
                    RasterInterpoladoResponse(atributo=attr, url=url_blob)
                )

            if req.gerar_csv:
                payload_grid = _montar_payload_grid_completo(req, df_final, profundidade)
                try:
                    logger.log(
                        NivelLog.INFO,
                        "v2_notificar_grid",
                        "Enviando grid completo ao storage externo.",
                        mostrar_usuario=True,
                    )
                    _post_storage("/api/v2/add_interpolacao_grid_completo", payload_grid)
                except Exception as exc:
                    logger.log(
                        NivelLog.ERROR,
                        "v2_grid_storage",
                        f"Falha ao enviar grid completo: {str(exc)}",
                        mostrar_usuario=True,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="Falha ao enviar grid completo para armazenamento externo.",
                    ) from exc
                grid_enviado = True

    return ProcessamentoV2Response(
        sucesso=True,
        mensagem=(
            f"Processamento V2 concluído para '{nome_lavoura}' "
            f"(processo='{processo}', campanha='{rotulo_campanha}')."
        ),
        rasters=rasters_enviados,
        grid_completo_enviado=grid_enviado,
    )
