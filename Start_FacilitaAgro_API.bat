@echo off
setlocal enabledelayedexpansion
title FacilitaAgro - API

REM ============================================================
REM Configurações rápidas (edite se quiser)
REM ============================================================
set "CONDA_ENV=agro"
set "PORT=8000"

REM ============================================================
REM 1) Ir para a pasta do projeto (mesma pasta deste .bat)
REM ============================================================
cd /d "%~dp0"

REM Se o .bat estiver dentro de uma subpasta, descomente e ajuste:
REM cd /d "%~dp0.."

REM ============================================================
REM 2) Ativar conda (tenta Anaconda padrão; se falhar, tenta conda no PATH)
REM ============================================================
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else (
  echo [AVISO] Nao encontrei activate.bat em anaconda3/miniconda3 no USERPROFILE.
  echo         Vou tentar usar o conda do PATH...
)

REM ============================================================
REM 3) Ativar o ambiente
REM ============================================================
call conda activate %CONDA_ENV%
if errorlevel 1 (
  echo [ERRO] Falha ao ativar o ambiente conda "%CONDA_ENV%".
  echo       Ajuste CONDA_ENV no topo do .bat ou crie o ambiente.
  pause
  exit /b 1
)

REM ============================================================
REM 4) Iniciar a API
REM ============================================================
echo.
echo =====================================================
echo Iniciando a API FacilitaAgro...
echo Endereco da API:        http://127.0.0.1:%PORT%
echo Documentacao (Swagger): http://127.0.0.1:%PORT%/docs
echo =====================================================
echo.

REM Obs: este comando assume que existe o pacote api/ com api/main.py
uvicorn api.main:app --host 0.0.0.0 --port %PORT% --reload

echo.
echo API finalizada.
pause
