# -*- coding: utf-8 -*-
"""
logger.py

Logger simples para o pipeline
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class LoggerConfig:
    job_dir: str
    filename: str = "pipeline.log"
    subdir: str = "tmp/logs"
    to_console: bool = True
    to_file: bool = True
    level: str = "INFO"  # "DEBUG" | "INFO" | "WARNING" | "ERROR"


_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


class PipelineLogger:
    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg
        self.level = cfg.level.upper().strip()
        if self.level not in _LEVEL_ORDER:
            self.level = "INFO"

        self.filepath: Optional[str] = None
        if cfg.to_file:
            p = Path(cfg.job_dir) / cfg.subdir
            p.mkdir(parents=True, exist_ok=True)
            self.filepath = str(p / cfg.filename)

    def _ok_level(self, level: str) -> bool:
        return _LEVEL_ORDER[level] >= _LEVEL_ORDER[self.level]

    def _emit(self, level: str, msg: str):
        if not self._ok_level(level):
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{level}] {ts} | {msg}"

        if self.cfg.to_console:
            print(line)

        if self.filepath and self.cfg.to_file:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def debug(self, msg: str): self._emit("DEBUG", msg)
    def info(self, msg: str): self._emit("INFO", msg)
    def warning(self, msg: str): self._emit("WARNING", msg)
    def error(self, msg: str): self._emit("ERROR", msg)

    def get_log_path(self) -> Optional[str]:
        return self.filepath
