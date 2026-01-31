from __future__ import annotations

from pathlib import Path
import os

import requests


def upload_blob_file(caminho_arquivo: Path, nome_arquivo: str) -> str:
    """Envia um arquivo para o blob storage da Vercel e retorna a URL pública."""
    token = os.getenv("NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN")
    if not token:
        raise ValueError("Variável NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN não configurada.")

    endpoint = f"https://blob.vercel-storage.com/{nome_arquivo}"
    with open(caminho_arquivo, "rb") as arquivo:
        resposta = requests.put(
            endpoint,
            data=arquivo,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
            timeout=60,
        )
    resposta.raise_for_status()

    url_blob = resposta.headers.get("x-vercel-blob-url")
    if url_blob:
        return url_blob

    try:
        dados = resposta.json()
    except ValueError:
        dados = {}

    return dados.get("url") or dados.get("downloadUrl") or endpoint

