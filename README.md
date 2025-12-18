# 📘 Facilita Agro — API & Pipelines Geoespaciais

## Visão Geral

O **Facilita Agro** é um conjunto de pipelines geoespaciais e uma API para processamento automatizado de dados agrícolas, com foco em **padronização, rastreabilidade e escalabilidade operacional**.

O sistema processa:

- Amostragens de **solo**
- Amostragens **foliares**
- Amostragens de **compactação**
- Amostragens de **nematoides**
- Mapas de **produtividade (yield)**

Todos os pipelines seguem a mesma filosofia:

> 🔹 **O cliente define os diretórios de entrada e saída**  
> 🔹 **O Facilita Agro define a lógica interna de organização e processamento**

Isso permite integração direta com diferentes plataformas, ERPs ou rotinas de TI.

---

## 🔁 Fluxo Geral dos Pipelines

Cada processo passa, conceitualmente, pelas etapas:

1. Leitura de dados tabulares (CSV)
2. Validação e padronização de colunas
3. Conversão espacial e reprojeção
4. Interpolação (IDW ou Kriging, com decisão automática)
5. Geração de rasters GeoTIFF
6. Amostragem dos rasters na grade da lavoura
7. Exportação de CSVs por processo/campanha
8. Atualização incremental do **grid completo** da lavoura

---

## 📁 Estrutura do Projeto (lógica, não caminhos fixos)

```
facilitaagro/
│
├── facilita_agro/          # Pipelines e lógica geoespacial
│   ├── pipeline_completa.py
│   ├── pipeline_yield.py
│   ├── filtro_yield_preprocess.py
│   ├── grid_completo.py
│   ├── interpolador.py
│   ├── processador_amostragem.py
│   └── ...
│
├── api/                    # API FastAPI
│   └── main.py
│
├── data/                   # Diretórios de entrada (definidos pelo cliente)
│
├── outputs/                # Diretórios de saída (definidos pelo cliente)
│
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

> ⚠️ Os nomes reais dos diretórios (`data`, `outputs`, etc.) são **parâmetros da API/pipelines**, não imposições do projeto.

---

## 🌱 Organização dos Outputs (padrão lógico)

### 🔹 Princípio geral
Todos os outputs são organizados por:

```
<diretório_de_saida> /
└── <nome_lavoura> /
    └── <processo ou campanha> /
```

---

## 📦 Yield (Produtividade)

### Etapa A — Pré-processamento

Entrada esperada:
```
<dir_yield>/<nome_lavoura>/<projeto_colheita>/*.csv
```

Saída (staging):
```
<dir_saida_yield_temp>/<nome_lavoura>/<projeto_colheita>/
```

### Etapa B — Interpolação

Saídas finais:
```
<dir_rasters>/<nome_lavoura>/<projeto_colheita>/
<dir_csv>/<nome_lavoura>/<projeto_colheita>/
```

---

## 🌾 Amostragens (solo, foliar, compac, nemat)

CSV:
```
<dir_csv>/<nome_lavoura>/csv_<processo>/
```

Rasters:
```
<dir_rasters>/<nome_lavoura>/<processo>_<data>/
```

---

## 🧩 Grid Completo

Busca recursiva em:
```
<dir_csv>/<nome_lavoura>/
```

Saída:
```
<dir_saida_grid>/<nome_lavoura>_grid_completo.csv
```

---

## 🌐 API — Visão Geral

- `GET /health`
- `POST /yield/preprocessar`
- `POST /yield/preprocessar-upload`
- `POST /yield/interpolar`
- `POST /processar-amostragem-upload`
- `POST /atualizar-grid-completo`

---

## 🚀 Pronto para Produção

- Pipelines modulares
- API estável
- Yield multi-colhedora suportado
- Grid completo consistente
- Docker-ready
