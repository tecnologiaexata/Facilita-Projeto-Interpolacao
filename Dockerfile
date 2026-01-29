FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências de sistema (geoespacial)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    proj-bin \
    libproj-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Diretórios "contrato" do pipeline
RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/outputs/yield_temp \
    /app/outputs/csv \
    /app/outputs/csv_grid_completo \
    /app/outputs/rasters_amostragem \
    /app/outputs/rasters_yield

# Variáveis de ambiente (documentação viva)
ENV DATA_DIR=/app/data
ENV OUTPUTS_DIR=/app/outputs

# Dependências Python
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código da aplicação
COPY facilita_agro /app/facilita_agro
COPY api /app/api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

