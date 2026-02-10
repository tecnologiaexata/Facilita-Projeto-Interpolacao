from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Union
import os
import re
import tempfile
import json
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from matplotlib import colors

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
    profundidade: Optional[int] = None
    atributos: List[AtributoV2]


class ProcessarAmostragemV2Request(BaseModel):
    tipo: Literal["gleba", "talhao"]
    id: int
    id_amostragem: int
    safra: str
    gerar_csv: bool = Field(default=True, description="Se True, envia dados do grid completo.")
    url_kml: str
    url_grade: str
    processo: ProcessoTipo
    cliente_id: Optional[int] = None
    data: str
    fazenda: Optional[int] = None
    talhao: int
    gleba: Optional[int] = None
    cultura: Optional[str] = None
    profundidade: Optional[int] = None
    dados: List[DadoAmostraV2]

    @root_validator(skip_on_failure=True)
    def validar_contexto(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        tipo = values.get("tipo")
        talhao = values.get("talhao")
        gleba = values.get("gleba")
        if not talhao:
            raise ValueError("Campo 'talhao' é obrigatório.")
        if tipo == "gleba" and gleba is None:
            return values
        return values


class RateLimitsConfig(BaseModel):
    unit: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    round_step: Optional[float] = None
    round_mode: Literal["nearest"] = "nearest"


class OutsideBufferConfig(BaseModel):
    enabled: bool = True
    width_m: float = 10.0
    mode: Literal["nearest"] = "nearest"


class ProcessingConfigV2Run(BaseModel):
    rate_limits: Optional[RateLimitsConfig] = None
    outside_buffer: Optional[OutsideBufferConfig] = None


class DadoRunV2(BaseModel):
    x: float
    y: float
    valor: float


class RunV2Request(BaseModel):
    processing: Optional[ProcessingConfigV2Run] = None
    atributos: str
    tipo: Literal["gleba", "talhao"]
    id: int
    url_kml: str
    url_grade: str
    processo: str
    cliente_id: Optional[int] = None
    date: str
    fazenda: Optional[int] = None
    talhao: int
    gleba: Optional[int] = None
    cultura: Optional[str] = None
    id_amostragem: int
    safra: str
    profundidade: Optional[int] = None
    data: List[DadoRunV2]


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
    id_amostragem: int = Field(..., description="Identificador da amostragem.")
    safra: str = Field(..., description="Safra (ex.: 2024/2025).")
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


class ConverterGpkgGeoJsonRequest(BaseModel):
    id_referencia: int
    url: str


class ConverterGpkgGeoJsonResponse(BaseModel):
    sucesso: bool
    mensagem: str
    url_geojson: str


class ConverterTifPngRequest(BaseModel):
    id_referencia: int
    url: str
    paleta: List[str]


class ConverterTifPngResponse(BaseModel):
    sucesso: bool
    mensagem: str
    url_png: str


# ============================================================
# Inicialização FastAPI
# ============================================================

app = FastAPI(
    title="Facilita Agro API",
    description="API de processamento geoespacial para dados agrícolas (amostragem e produtividade).",
    version="0.3.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{path:path}")
def options_preflight_handler(path: str) -> Response:
    return Response(status_code=204)

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


def _post_storage(
    endpoint: str,
    payload: Dict[str, Any],
    logger: Optional[LoggerAgricola] = None,
    log_tag: str = "storage_post",
) -> Dict[str, Any]:
    """
    POST no storage externo + logs completos do que foi enviado (estilo console.log).
    Se logger não for passado, ainda faz o POST normalmente.
    """
    url = f"{_storage_base_url()}{endpoint}"

    # 🔍 DEBUG: log do request (URL + BODY)
    try:
        payload_dump = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except Exception:
        payload_dump = str(payload)

    if logger is not None:
        logger.log(
            NivelLog.INFO,
            log_tag,
            f"POST {url}\nPAYLOAD:\n{payload_dump}",
            mostrar_usuario=True,
        )
    else:
        print(f"[DEBUG] POST {url}\nPAYLOAD:\n{payload_dump}")

    resposta = requests.post(url, json=payload, timeout=60)

    # 🔍 DEBUG: log da resposta (status + body) quando não for 2xx
    if resposta.status_code >= 400:
        body_preview = (resposta.text or "")[:4000]
        if logger is not None:
            logger.log(
                NivelLog.ERROR,
                f"{log_tag}_erro",
                f"Resposta HTTP {resposta.status_code} em {url}\nBODY:\n{body_preview}",
                mostrar_usuario=True,
            )
        else:
            print(f"[DEBUG] HTTP {resposta.status_code} {url}\nBODY:\n{body_preview}")

    resposta.raise_for_status()

    try:
        return resposta.json()
    except ValueError:
        return {"status": "ok"}


def _baixar_arquivo_url(url: str, sufixo_padrao: str) -> Path:
    resposta = requests.get(url, timeout=60)
    resposta.raise_for_status()

    sufixo = Path(urlparse(url).path).suffix or sufixo_padrao
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as arquivo_temp:
        arquivo_temp.write(resposta.content)
        return Path(arquivo_temp.name)


def _nome_blob_geojson(url: str, id_referencia: int) -> str:
    path = urlparse(url).path
    base = Path(path).stem or f"referencia_{id_referencia}"
    return _slugify_nome(f"{base}_geojson")


def _nome_blob_png(url: str, id_referencia: int) -> str:
    path = urlparse(url).path
    base = Path(path).stem or f"referencia_{id_referencia}"
    return _slugify_nome(f"{base}_png")


def _validar_paleta(paleta: List[str]) -> List[str]:
    if not paleta:
        raise ValueError("A paleta não pode ser vazia.")
    regex_hex = re.compile(r"^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")
    paleta_limpa: List[str] = []
    for cor in paleta:
        cor_str = cor.strip()
        if not regex_hex.match(cor_str):
            raise ValueError(f"Cor inválida na paleta: '{cor}'.")
        paleta_limpa.append(cor_str)
    return paleta_limpa


def _converter_tif_para_png(caminho_tif: Path, paleta: List[str], caminho_saida: Path) -> None:
    with rasterio.open(caminho_tif) as src:
        dados = src.read(1, masked=True)
        if dados.mask.all():
            raise ValueError("Raster sem pixels válidos para conversão.")

        valores_validos = dados.compressed()
        if valores_validos.size == 0:
            raise ValueError("Raster sem pixels válidos para conversão.")

        vmin = float(np.min(valores_validos))
        vmax = float(np.max(valores_validos))
        if vmin == vmax:
            vmax = vmin + 1.0

        colormap = colors.LinearSegmentedColormap.from_list(
            "custom_palette",
            paleta,
            N=len(paleta),
        )
        normalizador = colors.Normalize(vmin=vmin, vmax=vmax)

        rgba = colormap(normalizador(dados.filled(vmin)))
        rgba[..., 3] = np.where(dados.mask, 0.0, rgba[..., 3])
        rgba_uint8 = (rgba * 255).astype(np.uint8)

        height, width = dados.shape
        with rasterio.open(
            caminho_saida,
            "w",
            driver="PNG",
            width=width,
            height=height,
            count=4,
            dtype="uint8",
        ) as dst:
            for idx in range(4):
                dst.write(rgba_uint8[:, :, idx], idx + 1)


def _serializar_valor(valor: Any) -> Any:
    if pd.isna(valor):
        return None
    if isinstance(valor, (np.integer, np.floating)):
        return valor.item()
    return valor




def _normalizar_processo_v2_run(processo: str) -> str:
    proc = (processo or "").strip().lower()
    aliases = {
        "aplicacao": "solo",
        "aplicação": "solo",
        "solo": "solo",
        "prod": "prod",
        "producao": "prod",
        "produção": "prod",
        "compac": "compac",
        "compactacao": "compac",
        "compactação": "compac",
        "nemat": "nemat",
        "foliar": "foliar",
    }
    proc_norm = aliases.get(proc)
    if proc_norm is None:
        raise HTTPException(status_code=400, detail=f"Processo inválido: '{processo}'.")
    return proc_norm


def _normalizar_tipo_v2_run(tipo: str) -> str:
    tipo_norm = (tipo or "").strip().lower()
    if tipo_norm not in {"talhao", "gleba"}:
        raise HTTPException(status_code=400, detail=f"Tipo inválido: '{tipo}'.")
    return tipo_norm


def _montar_req_processar_v2(req: RunV2Request) -> ProcessarAmostragemV2Request:
    dados_convertidos: list[DadoAmostraV2] = []
    for idx, item in enumerate(req.data, start=1):
        dados_convertidos.append(
            DadoAmostraV2(
                latitude=item.y,
                longitude=item.x,
                ponto_coleta=f"P{idx}",
                profundidade=req.profundidade,
                atributos=[AtributoV2(nome=req.atributos, valor=item.valor)],
            )
        )

    return ProcessarAmostragemV2Request(
        tipo=_normalizar_tipo_v2_run(req.tipo),
        id=req.id,
        id_amostragem=req.id_amostragem,
        safra=req.safra,
        gerar_csv=True,
        url_kml=req.url_kml,
        url_grade=req.url_grade,
        processo=_normalizar_processo_v2_run(req.processo),
        cliente_id=req.cliente_id,
        data=req.date,
        fazenda=req.fazenda,
        talhao=req.talhao,
        gleba=req.gleba,
        cultura=req.cultura,
        profundidade=req.profundidade,
        dados=dados_convertidos,
    )


def _aplicar_rate_limits_em_df(df: pd.DataFrame, atributo: str, cfg: Optional[RateLimitsConfig]) -> pd.DataFrame:
    if cfg is None or atributo not in df.columns:
        return df

    serie = pd.to_numeric(df[atributo], errors="coerce")

    if cfg.min is not None:
        serie = serie.clip(lower=cfg.min)
    if cfg.max is not None:
        serie = serie.clip(upper=cfg.max)

    if cfg.round_step and cfg.round_step > 0:
        if cfg.round_mode != "nearest":
            raise HTTPException(status_code=400, detail="round_mode inválido. Use 'nearest'.")
        step = float(cfg.round_step)
        serie = (np.round(serie / step) * step).astype(float)

    df[atributo] = serie
    return df

def _montar_payload_grid_completo(
    req: ProcessarAmostragemV2Request,
    df_final: pd.DataFrame,
) -> Dict[str, Any]:
    cols_base = [
        "id_ponto",
        "Data",
        "Cliente",
        "Fazenda",
        "Talhão",
        "Gleba",
        "Profundidade",
        "Lat",
        "Long",
    ]
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
                "profundidade": _serializar_valor(row.get("Profundidade")),
                "atributos": atributos,
            }
        )

    return {
        "tipo": req.tipo,
        "id": req.id,
        "processo": req.processo,
        "id_amostragem": req.id_amostragem,
        "safra": req.safra,
        "cliente_id": req.cliente_id,
        "data": req.data,
        "fazenda": req.fazenda,
        "talhao": req.talhao,
        "gleba": req.gleba,
        "cultura": req.cultura,
        "profundidade": req.profundidade,
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
        req.id_amostragem,
        req.safra,
        url_kml,
        url_grade,
    )
    if not ok_grade:
        raise HTTPException(status_code=400, detail=msg_grade)

    return KmlGradeResponse(sucesso=True, mensagem=msg_grade, dados=dados_grade)


# --------------------- V2 Converter GPKG em GeoJSON ---------------------

@app.post("/v2/converterGPKGemGeoJson", response_model=ConverterGpkgGeoJsonResponse)
def converter_gpkg_geojson(req: ConverterGpkgGeoJsonRequest):
    logger = LoggerAgricola(f"converter_gpkg_{req.id_referencia}", salvar_arquivo=True)
    logger.log(
        NivelLog.INFO,
        "v2_geojson_inicio",
        f"Convertendo GPKG via URL: {req.url}",
        mostrar_usuario=True,
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            caminho_gpkg = _baixar_arquivo_url(req.url, ".gpkg")
            try:
                gdf = gpd.read_file(caminho_gpkg)
            finally:
                caminho_gpkg.unlink(missing_ok=True)

            if gdf.crs is None:
                raise ValueError("CRS não encontrado no arquivo GPKG.")

            gdf = gdf.to_crs(epsg=4326)

            caminho_geojson = Path(temp_dir) / f"{_nome_blob_geojson(req.url, req.id_referencia)}.geojson"
            gdf.to_file(caminho_geojson, driver="GeoJSON")

            nome_blob = caminho_geojson.name
            logger.log(
                NivelLog.INFO,
                "v2_geojson_upload",
                f"Enviando GeoJSON para blob: {nome_blob}",
                mostrar_usuario=True,
            )
            url_geojson = upload_blob_file(caminho_geojson, nome_blob)

        payload = {
            "id_referencia": req.id_referencia,
            "url_geojson": url_geojson,
        }
        logger.log(
            NivelLog.INFO,
            "v2_geojson_notificar",
            "Notificando GeoJSON no storage externo.",
            mostrar_usuario=True,
        )
        try:
            _post_storage(
                "/api/v2/addGeoJson",
                payload,
                logger=logger,
                log_tag="v2_payload_geojson",
            )
        except requests.HTTPError as exc:
            logger.log(
                NivelLog.WARNING,
                "v2_geojson_notificar_erro",
                f"Falha ao notificar storage externo: {exc}",
                mostrar_usuario=True,
            )

    except Exception as exc:
        logger.log(
            NivelLog.ERROR,
            "v2_geojson_erro",
            f"Falha ao converter/enviar GeoJSON: {str(exc)}",
            mostrar_usuario=True,
        )
        raise HTTPException(
            status_code=502,
            detail="Falha ao converter/enviar GeoJSON.",
        ) from exc

    return ConverterGpkgGeoJsonResponse(
        sucesso=True,
        mensagem="GeoJSON convertido e enviado com sucesso.",
        url_geojson=url_geojson,
    )


# --------------------- V2 Converter TIF em PNG ---------------------

@app.post("/v2/converterTifemPng", response_model=ConverterTifPngResponse)
def converter_tif_png(req: ConverterTifPngRequest):
    logger = LoggerAgricola(f"converter_tif_{req.id_referencia}", salvar_arquivo=True)
    logger.log(
        NivelLog.INFO,
        "v2_tif_inicio",
        f"Convertendo TIF via URL: {req.url}",
        mostrar_usuario=True,
    )

    try:
        paleta = _validar_paleta(req.paleta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            caminho_tif = _baixar_arquivo_url(req.url, ".tif")
            try:
                caminho_png = Path(temp_dir) / f"{_nome_blob_png(req.url, req.id_referencia)}.png"
                _converter_tif_para_png(caminho_tif, paleta, caminho_png)
            finally:
                caminho_tif.unlink(missing_ok=True)

            nome_blob = caminho_png.name
            logger.log(
                NivelLog.INFO,
                "v2_tif_upload",
                f"Enviando PNG para blob: {nome_blob}",
                mostrar_usuario=True,
            )
            url_png = upload_blob_file(caminho_png, nome_blob)

        payload = {
            "id_referencia": req.id_referencia,
            "url": url_png,
        }
        logger.log(
            NivelLog.INFO,
            "v2_tif_notificar",
            "Notificando PNG no storage externo.",
            mostrar_usuario=True,
        )
        try:
            _post_storage(
                "/api/v2/addPNGdoTif",
                payload,
                logger=logger,
                log_tag="v2_payload_png",
            )
        except requests.HTTPError as exc:
            logger.log(
                NivelLog.WARNING,
                "v2_tif_notificar_erro",
                f"Falha ao notificar storage externo: {exc}",
                mostrar_usuario=True,
            )

    except Exception as exc:
        logger.log(
            NivelLog.ERROR,
            "v2_tif_erro",
            f"Falha ao converter/enviar PNG: {str(exc)}",
            mostrar_usuario=True,
        )
        raise HTTPException(
            status_code=502,
            detail="Falha ao converter/enviar PNG.",
        ) from exc

    return ConverterTifPngResponse(
        sucesso=True,
        mensagem="PNG convertido e enviado com sucesso.",
        url_png=url_png,
    )


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
        req.id_amostragem,
        req.safra,
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
    proc_am = ProcessadorAmostragemV2(
        nome_lavoura=nome_lavoura,
        dados=[item.dict() for item in req.dados],
        metadados={
            "data": req.data,
            "cliente": req.cliente_id,
            "fazenda": req.fazenda,
            "talhao": req.talhao,
            "gleba": req.gleba,
            "cultura": req.cultura,
            "profundidade": req.profundidade,
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

        rasters_enviados: list[RasterInterpoladoResponse] = []
        for attr, caminho_tif in caminhos_rasters.items():
            nome_base = _slugify_nome(
                f"{req.tipo}_{req.id}_{processo}_{rotulo_campanha}_{attr}"
            )
            nome_blob = f"{nome_base}.tif"
            try:
                logger.log(
                    NivelLog.INFO,
                    "v2_upload_raster",
                    f"Enviando raster '{attr}' para blob: {nome_blob}",
                    mostrar_usuario=True,
                )
                url_blob = upload_blob_file(caminho_tif, nome_blob)

                # ✅ PAYLOAD EM VARIÁVEL + LOG COMPLETO (estilo console.log)
                payload_raster = {
                    "tipo": req.tipo,
                    "id": req.id,
                    "processo": processo,
                    "atributo": attr,
                    "campanha": rotulo_campanha,
                    "url": url_blob,
                    "id_amostragem": req.id_amostragem,
                    "safra": req.safra,
                    "cliente_id": req.cliente_id,
                    "data": req.data,
                    "fazenda": req.fazenda,
                    "talhao": req.talhao,
                    "gleba": req.gleba,
                    "profundidade": req.profundidade,
                }

                logger.log(
                    NivelLog.INFO,
                    "v2_notificar_raster",
                    f"Notificando raster interpolado '{attr}' no storage externo.",
                    mostrar_usuario=True,
                )

                # ✅ Agora o _post_storage já loga URL + BODY + ERRO (se houver)
                _post_storage(
                    "/api/v2/add_raster_interpolados",
                    payload_raster,
                    logger=logger,
                    log_tag="v2_payload_raster",
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

            rasters_enviados.append(RasterInterpoladoResponse(atributo=attr, url=url_blob))

        grid_enviado = False
        if req.gerar_csv:
            payload_grid = _montar_payload_grid_completo(req, df_final)
            try:
                logger.log(
                    NivelLog.INFO,
                    "v2_notificar_grid",
                    "Enviando grid completo ao storage externo.",
                    mostrar_usuario=True,
                )

                # ✅ Também loga payload do grid (muito útil!)
                _post_storage(
                    "/api/v2/add_interpolacao_grid_completo",
                    payload_grid,
                    logger=logger,
                    log_tag="v2_payload_grid",
                )

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


@app.post("/v2/run", response_model=ProcessamentoV2Response)
def run_v2(req: RunV2Request):
    """
    Compatibilidade com contrato V2/run:
      - valida pontos dentro do KML/perímetro
      - força distribuição/interpolação com base na grade enviada
      - aplica rate-limits no atributo solicitado
    """
    req_convertido = _montar_req_processar_v2(req)
    resp = processar_amostragem_v2(req_convertido)

    if not req.processing or not req.processing.rate_limits:
        return resp

    nome_lavoura = f"{req_convertido.tipo}_{req_convertido.id}"
    logger = LoggerAgricola(nome_lavoura, salvar_arquivo=True)

    csv_dir = DIR_CSV_BASE / nome_lavoura
    if not csv_dir.exists():
        return resp

    arquivos = sorted(csv_dir.rglob(f"*_{req_convertido.processo}_*_grid.csv"), reverse=True)
    if not arquivos:
        return resp

    caminho_csv = arquivos[0]
    df = pd.read_csv(caminho_csv)
    df = _aplicar_rate_limits_em_df(df, req.atributos, req.processing.rate_limits)
    df.to_csv(caminho_csv, index=False)

    logger.log(
        NivelLog.INFO,
        "v2_run_rate_limits",
        f"Rate limits aplicados no CSV final ({caminho_csv.name}) para atributo '{req.atributos}'.",
        mostrar_usuario=True,
    )

    return resp
