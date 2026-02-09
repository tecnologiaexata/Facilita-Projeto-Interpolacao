# Mapas de Aplicação — API 

Este repositório entrega:
- Um pipeline agronômico/geoespacial para gerar mapas de aplicação (dose raster -> prescrição vetorial -> export para monitor + PDF).
- Uma API FastAPI para integração.

## Rodando com Docker (Linux container)

Na raiz do projeto:

```bash
docker build -t mapas-aplicacao:0.1 .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/outputs:/app/outputs" \
  mapas-aplicacao:0.1
```

API:
- http://localhost:8000
Docs:
- http://localhost:8000/docs

## Endpoint principal

### POST /run (multipart/form-data)
Campos:
- config_json (string JSON)
- boundary_kml (arquivo .kml)
- soil_table (arquivo .csv)

Retorno:
- job_id
- outputs (artefatos por recomendação)
- downloads (URLs para baixar ZIP/PDF)

## Exemplo (cURL)

```bash
curl -X POST "http://localhost:8000/run" \
  -F 'config_json={
    "meta":{"field_name":"pivo3"},
    "recommendation":{"selection":["calcario"],"params":{"calcario":{"V_desejado":90,"PRNT":75}}},
    "processing":{
      "grid":{"pixel_size_m":10},
      "rate_limits":{"unit":"t/ha","min":1.0,"max":6.0,"round_step":0.5,"round_mode":"nearest"},
      "outside_buffer":{"enabled":true,"width_m":10,"mode":"nearest","fixed_rate":{"unit":"t/ha","value":null}}
    },
    "interpolation":{"method":"auto"},
    "export":{"format":"all","output_package":{"name_prefix":"export_final"}},
    "outputs":{"keep_intermediate":false},
    "report":{"enabled":true}
  }' \
  -F "boundary_kml=@data/contornos/pivo3.kml" \
  -F "soil_table=@data/amostragem/pivo3.csv"
```

## Integração (NextJS)

```js
const fd = new FormData();
fd.append("config_json", JSON.stringify(config));
fd.append("boundary_kml", fileKml);
fd.append("soil_table", fileCsv);

const res = await fetch("http://SEU_HOST:8000/run", { method: "POST", body: fd });
const out = await res.json();
```

## Observações
- Logo do relatório é fixo em `support_files/imagens_relatorio/logo_exata.png`.
- Recomenda-se enviar `meta.field_name` no JSON para identificar o talhão.
