# Agente Agro (Facilita Agro) — API + Inteligência Agronômica

Este repositório entrega o módulo **Agente de Inteligência Agronômica** em formato de **API**, com pipelines analíticos hard-coded.

A API expõe dois endpoints principais:

- `POST /run/temporal`  
  Roda o pipeline temporal (consistência entre safras, quando aplicável) e retorna `report_text` + `bundle_json`.

- `POST /run/individual`  
  Separa por `Safra_prod` e roda o pipeline intra-safra para cada safra, retornando uma lista com `report_text` + `bundle_json` por safra.

> **Saída do sistema**: `report_text` (humano) + `bundle_json` (estruturado para integração) + `audit_path` (JSON exportado pelo observability).

---

## Estrutura esperada

Na raiz do projeto:

```
.
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── README.md
├── agente_agro/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── pipelines/
│   │   ├── runner.py
│   │   ├── io_checks.py
│   │   ├── insight_engine.py
│   │   ├── recommendation_engine.py
│   │   ├── temporal_engine.py
│   │   ├── safety_quality.py
│   │   ├── collinearity.py
│   │   └── observability.py
│   ├── outputs/
│   └── audit/
```

---

## Como chamar os endpoints

Os endpoints recebem um **CSV** via upload (`multipart/form-data`).

### Temporal

```bash
curl -X POST "http://127.0.0.1:8000/run/temporal" \
  -F "file=@/caminho/grid_completo.csv" \
  -F "talhao=Talhao_001"
```

### Individual (por safra)

```bash
curl -X POST "http://127.0.0.1:8000/run/individual" \
  -F "file=@/caminho/grid_completo.csv" \
  -F "talhao=Talhao_001"
```

---

## Campos importantes na resposta

- `report_text`: texto final do relatório
- `bundle_json`: saída estruturada 
- `audit_path`: caminho do audit exportado localmente pelo observability
- `warnings`: lista de alertas 

---

## Observações de integração (plataforma)

1) O módulo **não entrega “produto final”**; ele entrega **diagnóstico e recomendações**.
2) O `bundle_json` contém as recomendações e evidências generadas.
3) O `audit` permite rastreabilidade (debug, QA, auditoria interna).

---

