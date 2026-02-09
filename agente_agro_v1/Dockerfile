FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md /app/

RUN pip install --upgrade pip && pip install .

COPY agente_agro/ /app/agente_agro/

RUN mkdir -p /app/agente_agro/audit /app/agente_agro/outputs

EXPOSE 8000

CMD ["uvicorn", "agente_agro.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
