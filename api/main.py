from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from facilita_agro.logger import LoggerAgricola, NivelLog
from facilita_agro.processador_lavoura import ProcessadorLavoura
from facilita_agro.pipeline_completa import PipelineAgricola
from facilita_agro.pipeline_yield import PipelineYield
from facilita_agro.grid_completo import GridCompletoManager

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
