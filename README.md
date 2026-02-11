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
- `POST /v2/processar-amostragem`
- `POST /atualizar-grid-completo`

---

## 🛠️ Troubleshooting do endpoint `POST /v2/processar-amostragem`

Se o endpoint receber dados e **não gerar raster** para alguns tipos de processo, o comportamento esperado do backend é:

1. O payload é convertido em DataFrame e os atributos são abertos dinamicamente (não há "delete" silencioso no payload bruto).
2. O processador remove campos que **nunca são interpolados** (ex.: `Data`, `Talhão`, `Profundidade`, `Ponto`, etc.).
3. Só ficam atributos com conteúdo numérico válido.
4. Por fim, o pipeline prioriza os atributos oficiais por processo (`COLS_INTERPOLAVEIS`), mas também aceita atributos numéricos extras fora da lista oficial.

Na prática: diferenças de nomenclatura não bloqueiam mais a interpolação, desde que o valor seja numérico válido.

### Atributos aceitos para interpolação por processo

- `solo`: `pH H2O`, `pH CaCl`, `pH SMP`, `P meh`, `P res`, `P total`, `Na`, `K`, `S`, `Ca`, `Mg`, `Al`, `H + Al`, `MO`, `CO`, `B`, `Cu`, `Fe`, `Mn`, `Zn`, `SB`, `t`, `T`, `V %`, `m %`, `Ca/Mg`, `Ca/K`, `Mg/K`, `(Ca+Mg)/K`, `Ca/t`, `Mg/t`, `Ca/T`, `Mg/T`, `K/T`, `(H+Al)/T`, `(Ca+Mg)/T`, `(Ca+Mg+K)/T`, `Argila`, `Silte`, `Areia Total`, `Areia Grossa`, `Areia Fina`, `Cascalho`.
- `foliar`: `N`, `P`, `K`, `Ca`, `Mg`, `S`, `B`, `Cu`, `Fe`, `Mn`, `Zn`, `Massa Fresca`, `Massa Seca`.
- `compac`: `Kpa`.
- `nemat`: `Quantidade`, `U/100cc`.
- `prod`: `Kg/Ha` (presente no pipeline, mas hoje o schema da API restringe o request para `solo|foliar|compac|nemat`).

### Motivos comuns para "não gerar"

- Coordenadas inválidas/vazias (`latitude`/`longitude`).
- Pontos fora do perímetro da lavoura.
- Atributo enviado como texto não conversível para número.
- Em versões antigas, nome de atributo diferente da convenção podia bloquear interpolação; agora atributos numéricos extras também são aceitos.

---

## 🚀 Pronto para Produção

- Pipelines modulares
- API estável
- Yield multi-colhedora suportado
- Grid completo consistente
- Docker-ready



```
docker compose stop facilita
docker compose rm -f facilita
docker compose build --no-cache facilita
docker compose up -d facilita
docker compose logs -f --tail=200 facilita
```
