"""
logger.py - Sistema de logs estruturados para o pipeline agrícola
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path


class NivelLog(Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"        # Para desenvolvedores
    INFO = "INFO"          # Informações de fluxo normal
    WARNING = "WARNING"    # Avisos não críticos
    ERROR = "ERROR"        # Erros que não necessariamente derrubam o processo
    CRITICAL = "CRITICAL"  # Erros críticos (pipeline quebra)
    USUARIO = "USUARIO"    # Mensagens pensadas para o usuário final


class LoggerAgricola:
    """Logger estruturado para processamento agrícola"""

    def __init__(
        self,
        nome_lavoura: str,
        usuario_id: Optional[str] = None,
        salvar_arquivo: bool = False,
    ) -> None:
        self.nome_lavoura = nome_lavoura
        self.usuario_id = usuario_id
        self.salvar_arquivo = salvar_arquivo

        self.logs = []          # logs completos (para dev/diagnóstico)
        self.logs_usuario = []  # logs simplificados (para exibir na UI)
        self.contexto: Dict[str, Any] = {}

        if self.salvar_arquivo:
            self.diretorio_logs = Path("logs")
            self.diretorio_logs.mkdir(exist_ok=True)
        else:
            self.diretorio_logs = None

    # ---------------------------------------------------------
    # Métodos principais
    # ---------------------------------------------------------

    def log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        """
        Registra log estruturado.

        Args:
            nivel: Nível do log (DEBUG, INFO, WARNING, ERROR, CRITICAL, USUARIO)
            etapa: Nome da etapa do processamento (ex: "importacao_csv")
            mensagem: Mensagem descritiva
            dados: Dados adicionais em formato dicionário
            mostrar_usuario: Se True, também registra log simplificado para o usuário
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "nivel": nivel.value,
            "etapa": etapa,
            "lavoura": self.nome_lavoura,
            "usuario": self.usuario_id,
            "mensagem": mensagem,
            "dados": dados or {},
        }

        if self.contexto:
            log_entry["contexto"] = self.contexto.copy()

        self.logs.append(log_entry)

        if nivel.value in ["INFO", "WARNING", "ERROR", "CRITICAL", "USUARIO"]:
            self._log_console(log_entry)

        if mostrar_usuario or nivel == NivelLog.USUARIO:
            self.log_usuario(etapa, mensagem, nivel=nivel.value)

    def log_usuario(self, etapa: str, mensagem: str, nivel: str = "INFO") -> None:
        """Registra log simplificado apenas para o usuário final."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "etapa": etapa,
            "mensagem": mensagem,
            "nivel": nivel,
        }
        self.logs_usuario.append(entry)
        print(f"👤 [{etapa}] {mensagem}")

    def resetar(self) -> None:
        """
        Limpa logs e contexto atuais.
        """
        self.logs.clear()
        self.logs_usuario.clear()
        self.contexto.clear()


    # ---------------------------------------------------------
    # Saída / exportação
    # ---------------------------------------------------------

    def exportar_logs(self) -> Dict[str, Any]:
        """
        Exporta logs em formato estruturado (para salvar ou retornar via API).
        """
        # contagem por nível
        niveis: Dict[str, int] = {}
        etapas: Dict[str, int] = {}

        for log in self.logs:
            niveis[log["nivel"]] = niveis.get(log["nivel"], 0) + 1
            etapas[log["etapa"]] = etapas.get(log["etapa"], 0) + 1

        return {
            "lavoura": self.nome_lavoura,
            "usuario": self.usuario_id,
            "total_logs": len(self.logs),
            "total_logs_usuario": len(self.logs_usuario),
            "estatisticas_niveis": niveis,
            "estatisticas_etapas": etapas,
            "tempo_inicio": self.logs[0]["timestamp"] if self.logs else None,
            "tempo_fim": self.logs[-1]["timestamp"] if self.logs else None,
            "logs": self.logs,
            "logs_usuario": self.logs_usuario,
        }

    def salvar_arquivo_log(self) -> Optional[Path]:
        """
        Salva logs completos em arquivo JSON, se salvar_arquivo=True.
        Retorna caminho do arquivo salvo, ou None se não salvar.
        """
        if not self.salvar_arquivo or self.diretorio_logs is None:
            return None

        nome_arquivo = f"log_{self.nome_lavoura}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        caminho = self.diretorio_logs / nome_arquivo

        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.exportar_logs(), f, indent=2, ensure_ascii=False)

        return caminho

    # ---------------------------------------------------------
    # Contexto e console
    # ---------------------------------------------------------

    def adicionar_contexto(self, **kwargs: Any) -> None:
        """
        Adiciona informações de contexto aos logs (ex: ano, tipo_amostragem, etc.).
        """
        self.contexto.update(kwargs)

    def _log_console(self, log_entry: Dict[str, Any]) -> None:
        """Imprime log colorido no console/terminal (para dev)."""
        nivel = log_entry["nivel"]

        cores = {
            "INFO": "\033[94m",        # Azul
            "WARNING": "\033[93m",     # Amarelo
            "ERROR": "\033[91m",       # Vermelho
            "CRITICAL": "\033[91;1m",  # Vermelho negrito
            "USUARIO": "\033[92m",     # Verde
        }

        cor = cores.get(nivel, "\033[0m")
        reset = "\033[0m"

        print(f"{cor}[{nivel}] {log_entry['etapa']}: {log_entry['mensagem']}{reset}")
